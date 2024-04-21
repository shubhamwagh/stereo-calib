import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

from stereo_calib.utils.calibration_utils import StereoCalibrationData


def draw_markers(img_points, img):
    """
    Draws circular markers on an image at the specified points.

    Args:
        img_points (List[np.ndarray]): List of arrays containing points coordinates.
        img (np.ndarray): The image on which to draw the markers.

    Returns:
        None
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for point in img_points:
        point = (int(point[0, 0]), int(point[0, 1]))
        cv2.circle(img, point, radius=5, color=(0, 255, 0), thickness=-1)  # GREEN

    cv2.imshow('Image with Points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_epilines(img_rect_1, img_rect_2, pts1, pts2, F):
    """
    Draws epipolar lines on two images based on the corresponding points and the fundamental matrix
    relating the two images.

    Args:
        img_rect_1, img_rect_2: The two images between which the epipolar lines will be drawn.
        pts1, pts2: Corresponding points in img_rect_1 and img_rect_2. Both should be Nx2 numpy arrays.
        F: The fundamental matrix relating img_rect_1 and img_rect_2.

    Returns:
        None
    """

    # Convert points to homogeneous coordinates
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)

    # Draw the lines on the images
    r, c = img_rect_1.shape[:2]
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img_rect_1 = cv2.line(img_rect_1, (x0, y0), (x1, y1), color, 1)
        img_rect_1 = cv2.circle(img_rect_1, tuple(pt1), 5, color, -1)
    r, c = img_rect_2.shape[:2]
    for r, pt1, pt2 in zip(lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img_rect_2 = cv2.line(img_rect_2, (x0, y0), (x1, y1), color, 1)
        img_rect_2 = cv2.circle(img_rect_2, tuple(pt2), 5, color, -1)

    # Display the images with epipolar lines
    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img_rect_1)
    plt.title('Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_rect_2)
    plt.title('Image 2'), plt.xticks([]), plt.yticks([])
    plt.show()


def plot_object_points(obj_pts: List[np.ndarray]) -> None:
    """
     Plots object points in 3D space.

     Args:
         obj_pts (List[np.ndarray]): List of arrays containing object points.

     Returns:
         None
     """
    objp = np.squeeze(obj_pts, axis=1)
    X = objp[:, 0]
    Y = objp[:, 1]
    Z = objp[:, 2]
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("Object Coordinates of Corners")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_left_and_right_detected_corners(left_corner_pts: np.ndarray,
                                         right_corner_pts: np.ndarray,
                                         image_size: Tuple[int, int]) -> None:
    """
    Plots the detected corner points from left and right images on a single plot.

    Args:
        left_corner_pts (List[np.ndarray]): array containing corner points in the left image.
        right_corner_pts (List[np.ndarray]): array containing corner points in the right image.
        image_size (Tuple[int, int]): Tuple containing the width and height of the images.

    Returns:
        None
    """
    if len(left_corner_pts.shape) == 3:
        left_corner_pts = np.squeeze(left_corner_pts, axis=1)
    if len(right_corner_pts.shape) == 3:
        right_corner_pts = np.squeeze(right_corner_pts, axis=1)
    image_width, image_height = image_size

    plt.figure()
    plt.plot(left_corner_pts[:, 0], left_corner_pts[:, 1], marker='o', linestyle='None', color='blue',
             label='Left Image Corners')
    plt.plot(right_corner_pts[:, 0], right_corner_pts[:, 1], marker='o', linestyle='None', color='orange',
             label='Right Image Corners')
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.gca().invert_yaxis()  # Set origin to be the top left
    plt.xlabel("X Pixel Coords")
    plt.ylabel("Y Pixel Coords")
    plt.title("CharucoBoard Corners (left/right images)")
    plt.legend()


def plot_reprojection_errors(reprojection_error: List[np.ndarray], camera_side: Optional[str] = None) -> None:
    """
    Plot reprojection errors.

    Args:
        reprojection_error (List[np.ndarray]): List of arrays containing reprojection errors for each camera.
        camera_side (Optional[str]): Optional string specifying the side of the camera used
        (e.g., 'Left', 'Right'). Defaults to None.

    Returns:
        None
    """
    plt.figure()
    for errors in reprojection_error:
        plt.plot(errors[:, :, 0], errors[:, :, 1], "x")
        plt.ioff()
    if camera_side is not None:
        plt.title(f"{camera_side} Camera Reprojection Errors (pixels)")
    else:
        plt.title("Camera Reprojection Errors (pixels)")
    plt.xlabel("X Errors (pixels)")
    plt.ylabel("Y Errors (pixels)")
    # plt.show()


def plot_left_right_errors(left_errors: List[float], right_errors: List[float]) -> None:
    """
    Plot errors for left and right images.

    Args:
        left_errors (List[float]): List of RMS errors for each left image
        right_errors (List[float]): List of RMS errors for each right image

    Returns:
        None
    """
    plt.figure()
    plt.plot(left_errors, right_errors, "x")
    plt.title("Images RMS Errors (closer to 0 is better)")
    plt.xlabel("Left Image Errors")
    plt.ylabel("Right Image Errors")
    plt.xlim([0, max(left_errors) + 0.1])
    plt.ylim([0, max(right_errors) + 0.1])
    # Add image numbers to the points
    [plt.annotate(index, (left_error, right_error)) for index, left_error, right_error in
     zip(range(0, len(left_errors)), left_errors, right_errors)]
    # plt.show()


def plot_reprojected_over_detected_points(detected_pts: np.ndarray,
                                          reprojected_pts: np.ndarray,
                                          image_size: Tuple[int, int],
                                          camera_side: Optional[str] = None) -> None:
    """
    Plot detected and reprojected points over an image.

    Args:
        detected_pts (np.ndarray): array containing detected points or markers
        reprojected_pts (np.ndarray): array containing reprojected points corresponding to the
        detected points, with the same format as detected_pts.
        image_size (Tuple[int, int]): Tuple containing the width and height of the image.
        camera_side (Optional[str]): Optional string specifying the side of the camera used
        (e.g., 'Left', 'Right'). Defaults to None.

    Returns:
        None
    """
    if len(detected_pts.shape) == 3:
        detected_pts = np.squeeze(detected_pts, axis=1)
    if len(reprojected_pts.shape) == 3:
        reprojected_pts = np.squeeze(reprojected_pts, axis=1)
    image_width, image_height = image_size

    plt.figure()
    plt.plot(detected_pts[:, 0], detected_pts[:, 1], marker='o', linestyle='None', color='blue',
             label='Detected points')
    plt.plot(reprojected_pts[:, 0], reprojected_pts[:, 1], marker='o', linestyle='None', color='orange',
             label='Reprojected points')
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.gca().invert_yaxis()  # Set origin to be the top left
    plt.xlabel("X Pixel Coords")
    plt.ylabel("Y Pixel Coords")
    if camera_side is not None:
        plt.title(f"{camera_side} CharucoBoard Corners (reprojected over detected)")
    else:
        plt.title("CharucoBoard Corners (reprojected over detected)")
    plt.legend()
    # plt.show()


def plot_disparity_map(disparity_map: np.ndarray, min_disparity: int = 8, num_disparity: int = 16 * 7) -> None:
    """
    Plots the disparity map.

    Args:
        disparity_map (np.ndarray): The input disparity map.
        min_disparity (int): The minimum disparity value. Default is 8.
        num_disparity (int): The number of disparity levels. Default is 16 * 7.

    Returns:
        None
    """
    disparity = disparity_map / 16.0
    disparity = 255.0 * (disparity - min_disparity) / num_disparity
    plt.figure()
    plt.title('Disparity map')
    plt.imshow(disparity, cmap='gray')
    plt.colorbar()
    # plt.show()


def plot_depth_map(depth_map: np.ndarray) -> None:
    """
    Plots the depth map.

    Args:
        depth_map (np.ndarray): The input depth map.

    Returns:
        None
    """
    plt.figure()
    plt.title('Depth map (m)')
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar()
    # plt.show()


def combine_stereo_images(left_img: np.ndarray,
                          right_img=np.ndarray,
                          display: bool = False,
                          draw_lines: bool = False) -> np.ndarray:
    """
    Combines the left and right stereo images into a single image.

    Args:
        left_img (np.ndarray): Left stereo image.
        right_img (np.ndarray): Right stereo image.
        display (bool): Whether to display the combined stereo image. Default is False.
        draw_lines (bool): Whether to draw horizontal lines on the combined stereo image. Default is False.

    Returns:
        np.ndarray: Joined combined stereo image.
    """
    joined = np.concatenate([left_img, right_img], axis=1)

    if draw_lines:
        # Draw horizontal lines
        height, width = joined.shape[:2]
        cv2.line(joined, (0, 100), (width, 100), (0, 255, 0), 2)
        cv2.line(joined, (0, 300), (width, 300), (0, 255, 0), 2)
        cv2.line(joined, (0, 500), (width, 500), (0, 255, 0), 2)
        cv2.line(joined, (0, 700), (width, 700), (0, 255, 0), 2)

    if display:
        plt.figure()
        plt.title("Combined (left/right) image")
        plt.imshow(cv2.cvtColor(joined, cv2.COLOR_BGR2RGB)) if len(joined.shape) == 3 else plt.imshow(joined,
                                                                                                      cmap='gray')
    return joined


def undistort_and_combine_stereo_images(left_img: np.ndarray,
                                        right_img: np.ndarray,
                                        stereo_calib_results: StereoCalibrationData,
                                        display: bool = False,
                                        draw_lines: bool = False) -> np.ndarray:
    """
    Undistorts the given left and right stereo images using the provided stereo calibration results,
    and then combines them into a single image.

    Args:
        left_img (np.ndarray): Left stereo image.
        right_img (np.ndarray): Right stereo image.
        stereo_calib_results (StereoCalibrationData): Stereo calibration results containing undistort maps.
        display (bool): Whether to display the undistorted stereo image. Default is False.
        draw_lines (bool): Whether to draw horizontal lines on the undistorted stereo image. Default is False.

    Returns:
        np.ndarray: Joined undistorted stereo image.
    """

    left_calib = stereo_calib_results.left_camera_calibration_data
    right_calib = stereo_calib_results.right_camera_calibration_data
    undistorted_left = cv2.remap(left_img, left_calib.undistort_map_x, left_calib.undistort_map_y,
                                 interpolation=cv2.INTER_LINEAR)
    undistorted_right = cv2.remap(right_img, right_calib.undistort_map_x, right_calib.undistort_map_y,
                                  interpolation=cv2.INTER_LINEAR)
    joined_undistort = np.concatenate([undistorted_left, undistorted_right], axis=1)

    if draw_lines:
        # Draw horizontal lines
        height, width = joined_undistort.shape[:2]
        cv2.line(joined_undistort, (0, 100), (width, 100), (0, 255, 0), 2)
        cv2.line(joined_undistort, (0, 300), (width, 300), (0, 255, 0), 2)
        cv2.line(joined_undistort, (0, 500), (width, 500), (0, 255, 0), 2)
        cv2.line(joined_undistort, (0, 700), (width, 700), (0, 255, 0), 2)

    if display:
        plt.figure()
        plt.title("Undistorted (left/right) image")
        plt.imshow(cv2.cvtColor(joined_undistort, cv2.COLOR_BGR2RGB)) if len(
            joined_undistort.shape) == 3 else plt.imshow(joined_undistort, cmap='gray')
    return joined_undistort


def stereo_rectify_and_combine_images(left_img: np.ndarray,
                                      right_img: np.ndarray,
                                      stereo_calib_results: StereoCalibrationData,
                                      display: bool = False,
                                      draw_lines: bool = False) -> np.ndarray:
    """
    Rectifies the given left and right stereo images using the provided stereo calibration results,
    and then combines them into a single image.

    Args:
        left_img (np.ndarray): Left stereo image.
        right_img (np.ndarray): Right stereo image.
        stereo_calib_results (StereoCalibrationData): Stereo calibration results containing rectify maps.
        display (bool): Whether to display the rectified stereo image. Default is False.
        draw_lines (bool): Whether to draw horizontal lines on the rectified stereo image. Default is False.

    Returns:
        np.ndarray: Joined rectified stereo image.
    """

    rect_left = cv2.remap(left_img, stereo_calib_results.left_camera_calibration_data.stereo_rectify_map_x,
                          stereo_calib_results.left_camera_calibration_data.stereo_rectify_map_y,
                          interpolation=cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, stereo_calib_results.right_camera_calibration_data.stereo_rectify_map_x,
                           stereo_calib_results.right_camera_calibration_data.stereo_rectify_map_y,
                           interpolation=cv2.INTER_LINEAR)
    joined_rect = np.concatenate([rect_left, rect_right], axis=1)

    if draw_lines:
        # Draw horizontal lines
        height, width = joined_rect.shape[:2]
        cv2.line(joined_rect, (0, 100), (width, 100), (0, 255, 0), 2)
        cv2.line(joined_rect, (0, 300), (width, 300), (0, 255, 0), 2)
        cv2.line(joined_rect, (0, 500), (width, 500), (0, 255, 0), 2)
        cv2.line(joined_rect, (0, 700), (width, 700), (0, 255, 0), 2)

    if display:
        plt.figure()
        plt.title("Stereo rectified image")
        plt.imshow(cv2.cvtColor(joined_rect, cv2.COLOR_BGR2RGB)) if len(joined_rect.shape) == 3 else plt.imshow(
            joined_rect, cmap='gray')
    return joined_rect
