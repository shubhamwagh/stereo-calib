import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple, Union

from stereo_calib.utils.vis_utils import stereo_rectify_and_combine_images
from stereo_calib.utils.calibration_utils import CameraCalibrationData, StereoCalibrationData


def calculate_reprojection_error_per_image(obj_pts: np.ndarray,
                                           image_pts: np.ndarray,
                                           rvec: np.ndarray,
                                           tvec: np.ndarray,
                                           camera_matrix: np.ndarray,
                                           dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates the reprojection error for a single image using camera calibration parameters.

    Args:
        obj_pts: Array containing object points for the image.
        image_pts: Array containing image points for the image.
        rvec: Rotation vector for the image.
        tvec: Translation vector for the image.
        camera_matrix: Camera matrix containing intrinsic parameters.
        dist_coeffs: Distortion coefficients.

    Returns:
        Tuple containing:
            - Array of projected points.
            - Array of error points (difference between projected and actual image points).
            - Root mean square (RMS) reprojection error for the image.
    """
    projected_pts, _ = cv2.projectPoints(objectPoints=obj_pts,
                                         rvec=rvec,
                                         tvec=tvec,
                                         cameraMatrix=camera_matrix,
                                         distCoeffs=dist_coeffs)
    error_pts = image_pts - projected_pts
    rms_error_per_image = np.sqrt(np.mean(np.sum(error_pts.squeeze(axis=1) ** 2, axis=1)))
    return projected_pts, error_pts, rms_error_per_image


def calculate_reprojection_errors_for_all_images(all_obj_pts: List[np.ndarray],
                                                 all_image_pts: List[np.ndarray],
                                                 camera_calib_results: CameraCalibrationData) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Calculates the re-projection errors for all images in a set of object points and corresponding
    image points using camera calibration results.

    Args:
        all_obj_pts: List of arrays containing object points for each image.
        all_image_pts: List of arrays containing image points for each image.
        camera_calib_results: CameraCalibrationData object containing calibration results.

    Returns:
        Tuple of three lists:
            - List of arrays containing reprojected points for each image.
            - List of arrays containing reprojected error points for each image.
            - List of re-projection errors for each image.
    """
    assert len(all_obj_pts) == len(camera_calib_results.rotation_vectors) == len(
        camera_calib_results.translation_vectors)
    assert len(all_obj_pts) == len(all_image_pts)
    reprojected_error_pts = []
    reprojection_errors = []
    reprojected_pts = []

    for id, obj_pts in enumerate(all_obj_pts):
        projected_pts, error_pts, rms_error_per_image = calculate_reprojection_error_per_image(
            obj_pts=obj_pts,
            image_pts=all_image_pts[id],
            rvec=camera_calib_results.rotation_vectors[id],
            tvec=camera_calib_results.translation_vectors[id],
            camera_matrix=camera_calib_results.camera_matrix,
            dist_coeffs=camera_calib_results.dist_coeffs
        )
        reprojected_pts.append(projected_pts)
        reprojection_errors.append(rms_error_per_image)
        reprojected_error_pts.append(error_pts)
    return reprojected_pts, reprojected_error_pts, reprojection_errors


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Args:
    - image: Input image (grayscale).
    - clip_limit: Threshold for contrast limiting.
    - grid_size: Size of grid for histogram equalization.

    Returns:
    - Output image after applying CLAHE.
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE to the input image
    clahe_image = clahe.apply(image)

    return clahe_image


def compute_disparity(left_rectified_image: np.ndarray,
                      right_rectified_image: np.ndarray,
                      min_disparity: int = 8,
                      num_disparity: int = 16 * 7) -> np.ndarray:
    """
    Computes the disparity map from rectified left and right stereo images using the
    Semi-Global Block Matching (SGBM) algorithm followed by a Weighted Least Squares (WLS) filter.

    Args:
        left_rectified_image: Rectified left stereo image.
        right_rectified_image: Rectified right stereo image.
        min_disparity: Minimum possible disparity value.
        num_disparity: Number of disparity levels.

    Returns:
        np.ndarray: Raw disparity map of type np.float32.
    """
    num_channels = 3 if len(left_rectified_image.shape) == 3 else 1

    window_size = 5
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparity,  # max_disp has to be divisible by 16
        blockSize=window_size,
        P1=8 * num_channels * window_size * window_size,
        P2=32 * num_channels * window_size * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

    # WLS filter Parameters
    lmbda = 80000
    sigma = 1.3
    radius = 7
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    wls_filter.setDepthDiscontinuityRadius(radius)

    left_disp = stereo_left.compute(left_rectified_image, right_rectified_image)
    right_disp = stereo_right.compute(right_rectified_image, left_rectified_image)

    disparity = wls_filter.filter(left_disp, left_rectified_image, None, right_disp)
    disparity = disparity.clip(0.0)
    disparity = disparity.astype(np.float32)
    return disparity


def compute_depth_map(disparity: np.ndarray, disparity_to_depth_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the depth map from a disparity map and a disparity-to-depth transformation matrix.

    Args:
        disparity: Disparity map of type np.float32.
        disparity_to_depth_matrix: Disparity-to-depth transformation matrix.

    Returns:
        np.ndarray: Depth map of type np.float32.
    """
    img_3d = cv2.reprojectImageTo3D(disparity=disparity / 16.0,
                                    Q=disparity_to_depth_matrix,
                                    handleMissingValues=True,
                                    ddepth=cv2.CV_32F)

    depth_map = img_3d[:, :, 2]
    cv_outlier_depth = 10000
    depth_map[depth_map == cv_outlier_depth] = 0.0
    return depth_map.astype(np.float32)


def to_disparity_image(raw_disparity: np.ndarray,
                       min_disparity: int = 8,
                       num_disparity: int = 16 * 7) -> np.ndarray:
    """
    Converts raw disparity values to a disparity image.

    Args:
        raw_disparity (np.ndarray): Raw disparity map of type np.float32.
        min_disparity (int): Minimum possible disparity value.
        num_disparity (int): Number of disparity levels.

    Returns:
        np.ndarray: Disparity image.
    """
    disparity_img = raw_disparity / 16.0
    disparity_img = 255.0 * (disparity_img - min_disparity) / num_disparity
    return disparity_img


def save_rectified_stereo_images_with_disparity_and_depth_maps(left_images_path: List[str],
                                                               right_images_path: List[str],
                                                               stereo_calib_results: StereoCalibrationData,
                                                               output_dir: Union[str, Path]) -> None:
    """
    Saves rectified stereo images along with their disparity and depth maps.

    Args:
        left_images_path (List[str]): List of paths to left stereo images.
        right_images_path (List[str]): List of paths to right stereo images.
        stereo_calib_results (StereoCalibrationData): Stereo calibration results.
        output_dir (Union[str, Path]): Directory to save the output images.

    Returns:
        None
    """
    assert len(left_images_path) == len(right_images_path)

    stereo_rectified_path = Path(output_dir) / "stereo_rectified"
    disparity_map_path = Path(output_dir) / "disparity_map"
    raw_depth_map_path = Path(output_dir) / "raw_depth_map"
    depth_map_img_path = Path(output_dir) / "depth_map_img"

    stereo_rectified_path.mkdir(exist_ok=True, parents=True)
    disparity_map_path.mkdir(exist_ok=True, parents=True)
    raw_depth_map_path.mkdir(exist_ok=True, parents=True)
    depth_map_img_path.mkdir(exist_ok=True, parents=True)

    count = 0
    for left_img_path, right_img_path in tqdm(zip(left_images_path, right_images_path),
                                              desc="Saving rectified stereo images, disparity maps, and depth maps",
                                              total=len(left_images_path)):
        left_img = cv2.imread(left_img_path, 0)
        right_img = cv2.imread(right_img_path, 0)

        height, width = left_img.shape[:2]

        stereo_recitifed_img = stereo_rectify_and_combine_images(left_img, right_img, stereo_calib_results)
        cv2.imwrite(str(stereo_rectified_path.joinpath(f'{count:02d}.png')), stereo_recitifed_img)

        left_rectified_img, right_rectified_img = stereo_recitifed_img[:, :width], stereo_recitifed_img[:, width:]

        min_disparity = 8
        num_disparity = 16 * 7
        raw_disparity = compute_disparity(left_rectified_img,
                                          right_rectified_img,
                                          min_disparity=min_disparity,
                                          num_disparity=num_disparity)

        disparity_img = to_disparity_image(raw_disparity, min_disparity, num_disparity).astype(np.uint8)
        disparity_img = cv2.convertScaleAbs(disparity_img, 1)
        cv2.imwrite(str(disparity_map_path.joinpath(f'{count:02d}.png')), disparity_img)

        depth_map = compute_depth_map(raw_disparity, stereo_calib_results.perspective_transformation_matrix_Q)
        depth_map_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=255 / np.max(depth_map)),
                                          cv2.COLORMAP_PLASMA)

        cv2.imwrite(str(depth_map_img_path.joinpath(f'{count:02d}.png')), depth_map_img)
        cv2.imwrite(str(raw_depth_map_path.joinpath(f'{count:02d}.tiff')), depth_map)

        count += 1
    logger.success("Successfully saved all rectified stereo images, disparity maps, and depth maps!")
    return
