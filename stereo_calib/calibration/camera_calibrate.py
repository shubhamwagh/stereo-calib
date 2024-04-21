import logging

import cv2
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Union, Optional, List, Tuple

from stereo_calib.charuco import CharucoBoard
from stereo_calib.utils import CameraCalibrationData, StereoCalibrationData


class StereoCalibration:
    """
    Class for performing stereo calibration using Charuco boards.

    Args:
        data_path (Union[str, Path]): Path to the directory containing left and right camera images.
        charuco_board (CharucoBoard): Charuco board object used for calibration.

    Attributes:
        data_path (Path): Path to the directory containing calibration images.
        charuco_board (CharucoBoard): Charuco board object used for calibration.
        left_images_path (List[str]): List of paths to left camera images.
        right_images_path (List[str]): List of paths to right camera images.
        stereo_obj_points (List[np.ndarray]): List to store stereo object points.
        stereo_charuco_points_l (List[np.ndarray]): List to store stereo charuco points in left image plane.
        stereo_charuco_ids_l (List[np.ndarray]): List to store stereo charuco ids in left image.
        stereo_charuco_points_r (List[np.ndarray]): List to store stereo charuco points in right image plane.
        stereo_charuco_ids_r (List[np.ndarray]): List to store stereo charuco ids in right image.
        frame_size (Optional[Tuple[int, int]]): Size of the calibration images.
        criteria (Tuple[int, int, float]): Termination criteria for calibration.
        stereo_criteria (Tuple[int, int, float]): Termination criteria for stereo calibration.
        left_camera_calib_results (Optional[CameraCalibrationData]): Results of left camera calibration.
        right_camera_calib_results (Optional[CameraCalibrationData]): Results of right camera calibration.
        best_calib_images_indices (List[int]): Indices of the best calibration images.
        recalibrate (bool): Flag indicating whether recalibration is required.
        max_allowable_rms_error (float): Maximum allowable RMS error for calibration.
    """

    def __init__(self, data_path: Union[str, Path], charuco_board: CharucoBoard):
        """
         Initialize StereoCalibration object.

         Args:
             data_path (Union[str, Path]): Path to the directory containing left and right camera images.
             charuco_board (CharucoBoard): Charuco board object used for calibration.
         """
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.charuco_board = charuco_board

        self.left_images_path = self.load_images(self.data_path.joinpath("left", "*.png"))
        self.right_images_path = self.load_images(self.data_path.joinpath("right", "*.png"))
        assert len(self.left_images_path) == len(self.right_images_path)

        # List to store stereo object points, charuco points and charuco ids from all the images.
        self.stereo_obj_points: List[np.ndarray] = []  # 3d point in real world space
        self.stereo_charuco_points_l: List[np.ndarray] = []  # stereo charuco points in left image plane.
        self.stereo_charuco_ids_l: List[np.ndarray] = []  # stereo charuco ids in left image
        self.stereo_charuco_points_r: List[np.ndarray] = []  # stereo charuco points in rightimage plane.
        self.stereo_charuco_ids_r: List[np.ndarray] = []  # stereo charuco ids in right image

        self.frame_size: Optional[Tuple[int, int]] = None
        self._min_points: int = 5

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
        self.stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 1e-6)

        self.left_camera_calib_results: Optional[CameraCalibrationData] = None
        self.right_camera_calib_results: Optional[CameraCalibrationData] = None
        self.best_calib_images_indices: List[int] = []
        self.recalibrate: bool = False
        self.max_allowable_rms_error = 0.4
        self.process_images()

    @staticmethod
    def load_images(directory_path: Union[str, Path]) -> List[str]:
        """
        Load images from a directory.

        Args:
            directory_path (Union[str, Path]): Path to the directory containing images.

        Returns:
            List[str]: List of paths to the images.
        """
        images = sorted(glob.glob(str(directory_path)), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return images

    def init_camera_matrix(self):
        """
        Initialize camera matrix.

        Returns:
            np.ndarray: Initial camera matrix.
        """
        fx, fy = 570.0, 570.0
        height, width = self.frame_size
        cx, cy = width / 2.0, height / 2.0

        initial_camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return initial_camera_matrix

    def process_images(self):
        """
         Process calibration images to extract charuco points and ids.
        """
        params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(self.charuco_board.aruco_dict, params)

        for img_left_path, img_right_path in tqdm(zip(self.left_images_path, self.right_images_path),
                                                  total=len(self.left_images_path),
                                                  desc="Processing calibration images"):
            img_l = cv2.imread(img_left_path, cv2.IMREAD_COLOR)
            img_r = cv2.imread(img_right_path, cv2.IMREAD_COLOR)

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            if self.frame_size is None:
                self.frame_size = gray_l.shape
            else:
                assert self.frame_size == gray_l.shape == gray_r.shape

            corners_l, ids_l, rejected_img_points_l = aruco_detector.detectMarkers(gray_l)
            corners_r, ids_r, rejected_img_points_r = aruco_detector.detectMarkers(gray_r)

            if ids_l is not None and ids_r is not None:
                retval_l, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids_l,
                                                                                                 gray_l,
                                                                                                 self.charuco_board.board)
                retval_r, charuco_corners_r, charuco_ids_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids_r,
                                                                                                 gray_r,
                                                                                                 self.charuco_board.board)
                if charuco_corners_l is None or charuco_corners_r is None:
                    continue

                if retval_l > self._min_points and retval_r > self._min_points:
                    obj_pts_l, img_pts_l = cv2.aruco.getBoardObjectAndImagePoints(self.charuco_board.board,
                                                                                  charuco_corners_l,
                                                                                  charuco_ids_l)
                    obj_pts_r, img_pts_r = cv2.aruco.getBoardObjectAndImagePoints(self.charuco_board.board,
                                                                                  charuco_corners_r,
                                                                                  charuco_ids_r)

                    pts_l = {tuple(a): tuple(b) for a, b in zip(obj_pts_l[:, 0], img_pts_l[:, 0])}
                    pts_r = {tuple(a): tuple(b) for a, b in zip(obj_pts_r[:, 0], img_pts_r[:, 0])}
                    ids_l = {tuple(a): b for a, b in zip(obj_pts_l[:, 0], charuco_ids_l[:, 0])}
                    ids_r = {tuple(a): b for a, b in zip(obj_pts_r[:, 0], charuco_ids_r[:, 0])}
                    common_pts = set(pts_l.keys()) & set(pts_r.keys())

                    obj = np.zeros((len(common_pts), 1, 3), dtype=np.float32)
                    left_corners = np.zeros((len(common_pts), 1, 2), dtype=np.float32)
                    right_corners = np.zeros((len(common_pts), 1, 2), dtype=np.float32)
                    left_corner_ids = np.zeros((len(common_pts), 1), dtype=np.int32)
                    right_corner_ids = np.zeros((len(common_pts), 1), dtype=np.int32)
                    for i, pts in enumerate(common_pts):
                        obj[i] = pts
                        left_corners[i] = np.reshape(pts_l[pts], (1, 2))
                        right_corners[i] = np.reshape(pts_r[pts], (1, 2))
                        left_corner_ids[i] = ids_l[pts]
                        right_corner_ids[i] = ids_r[pts]

                    self.stereo_obj_points.append(obj)
                    self.stereo_charuco_points_l.append(left_corners)
                    self.stereo_charuco_points_r.append(right_corners)
                    self.stereo_charuco_ids_l.append(left_corner_ids)
                    self.stereo_charuco_ids_r.append(right_corner_ids)

        assert len(self.stereo_obj_points) == len(self.stereo_charuco_points_l) == len(
            self.stereo_charuco_points_r) == len(
            self.stereo_charuco_ids_l) == len(self.stereo_charuco_ids_r)

        init_camera_matrix = self.init_camera_matrix()

        # left camera calibration
        self.left_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_l,
                                                               charuco_ids=self.stereo_charuco_ids_l,
                                                               initial_camera_matrix=init_camera_matrix,
                                                               criteria=self.criteria)

        # right camera calibration
        self.right_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_r,
                                                                charuco_ids=self.stereo_charuco_ids_r,
                                                                initial_camera_matrix=init_camera_matrix,
                                                                criteria=self.criteria)

        if self.left_camera_calib_results.rms_reprojection_error > self.max_allowable_rms_error or \
                self.right_camera_calib_results.rms_reprojection_error > self.max_allowable_rms_error:
            self.recalibrate = True
            self.select_best_calib_images()

    def calibrate_camera(self,
                         charuco_points: List[np.ndarray],
                         charuco_ids: List[np.ndarray],
                         initial_camera_matrix: np.ndarray,
                         criteria: Tuple[int, int, float]) -> CameraCalibrationData:
        """
        Calibrate a camera.

        Args:
            charuco_points (List[np.ndarray]): Charuco points detected in the calibration images.
            charuco_ids (List[np.ndarray]): Charuco IDs detected in the calibration images.
            initial_camera_matrix (np.ndarray): Initial camera matrix.
            criteria (Tuple[int, int, float]): Termination criteria for calibration.

        Returns:
            CameraCalibrationData: Results of camera calibration.
        """
        flags = 0
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_RATIONAL_MODEL

        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=charuco_points,
            charucoIds=charuco_ids,
            board=self.charuco_board.board,
            imageSize=self.frame_size,
            cameraMatrix=initial_camera_matrix,
            distCoeffs=None,
            flags=flags,
            criteria=criteria)

        image_width, image_height = self.frame_size[::-1]

        undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(cameraMatrix=camera_matrix,
                                                                       distCoeffs=dist_coeffs,
                                                                       R=None,
                                                                       newCameraMatrix=None,
                                                                       size=self.frame_size[::-1],
                                                                       m1type=cv2.CV_16SC2)

        return CameraCalibrationData(rms_reprojection_error=retval,
                                     camera_matrix=camera_matrix,
                                     dist_coeffs=dist_coeffs,
                                     rotation_vectors=list(rvecs),
                                     translation_vectors=list(tvecs),
                                     undistort_map_x=undistort_map_x,
                                     undistort_map_y=undistort_map_y,
                                     image_width=image_width,
                                     image_height=image_height,
                                     image_dim=(image_width, image_height))

    def select_best_calib_images(self, keep_best_ratio: float = 0.75) -> None:
        """
        Select the best calibration images based on reprojection error.

        Args:
            keep_best_ratio (float): Ratio of best images to keep.
        """
        left_reprojection_errors = self.calculate_reprojection_error(obj_pts=self.stereo_obj_points,
                                                                     image_pts=self.stereo_charuco_points_l,
                                                                     camera_calib_results=self.left_camera_calib_results)
        right_reprojection_errors = self.calculate_reprojection_error(obj_pts=self.stereo_obj_points,
                                                                      image_pts=self.stereo_charuco_points_r,
                                                                      camera_calib_results=self.right_camera_calib_results)

        mean_errors = (np.array(left_reprojection_errors) + np.array(right_reprojection_errors)) / 2.0
        mean_errors = mean_errors.tolist()
        mean_errors_sorted_indices = sorted(range(len(mean_errors)), key=lambda i: mean_errors[i])
        threshold_index = int(len(mean_errors_sorted_indices) * keep_best_ratio)
        self.best_calib_images_indices = sorted(mean_errors_sorted_indices[:threshold_index])

    @staticmethod
    def calculate_reprojection_error(obj_pts: List[np.ndarray],
                                     image_pts: List[np.ndarray],
                                     camera_calib_results: CameraCalibrationData) -> List[float]:
        """
        Calculate reprojection error.

        Args:
            obj_pts (List[np.ndarray]): Object points.
            image_pts (List[np.ndarray]): Image points.
            camera_calib_results (CameraCalibrationData): Results of camera calibration.

        Returns:
            List[float]: List of reprojection errors.
        """
        reprojected_errors: List[float] = [0.0] * len(obj_pts)
        for id, obj in enumerate(obj_pts):
            projected_pts, _ = cv2.projectPoints(objectPoints=obj,
                                                 rvec=camera_calib_results.rotation_vectors[id],
                                                 tvec=camera_calib_results.translation_vectors[id],
                                                 cameraMatrix=camera_calib_results.camera_matrix,
                                                 distCoeffs=camera_calib_results.dist_coeffs)
            error_pts = image_pts[id] - projected_pts
            rms_error_per_image = np.sqrt(np.mean(np.sum(error_pts.squeeze(axis=1) ** 2, axis=1)))
            reprojected_errors[id] = rms_error_per_image
        return reprojected_errors

    def recalibrate_cameras(self):
        """
        Recalibrate cameras using the best calibration images.
        """
        self.left_images_path = [self.left_images_path[i] for i in self.best_calib_images_indices]
        self.right_images_path = [self.right_images_path[i] for i in self.best_calib_images_indices]
        self.stereo_obj_points = [self.stereo_obj_points[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_points_l = [self.stereo_charuco_points_l[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_ids_l = [self.stereo_charuco_ids_l[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_points_r = [self.stereo_charuco_points_r[i] for i in self.best_calib_images_indices]
        self.stereo_charuco_ids_r = [self.stereo_charuco_ids_r[i] for i in self.best_calib_images_indices]

        self.left_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_l,
                                                               charuco_ids=self.stereo_charuco_ids_l,
                                                               initial_camera_matrix=self.left_camera_calib_results.camera_matrix,
                                                               criteria=self.criteria)

        self.right_camera_calib_results = self.calibrate_camera(charuco_points=self.stereo_charuco_points_r,
                                                                charuco_ids=self.stereo_charuco_ids_r,
                                                                initial_camera_matrix=self.right_camera_calib_results.camera_matrix,
                                                                criteria=self.criteria)

    def calibrate(self) -> StereoCalibrationData:
        """
        Perform stereo calibration.

        Returns:
            StereoCalibrationData: Results of stereo calibration.
        """
        logger.info("Starting Calibration")
        flags = 0
        flags |= cv2.CALIB_RATIONAL_MODEL

        if self.recalibrate:
            self.recalibrate_cameras()
            flags |= cv2.CALIB_FIX_INTRINSIC
        else:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        ret_stereo, new_camera_matrix_l, new_dist_coeffs_l, new_camera_matrix_r, new_dist_coeffs_r, rot, trans, \
            essential_matrix, fundamental_matrix = cv2.stereoCalibrate(objectPoints=self.stereo_obj_points,
                                                                       imagePoints1=self.stereo_charuco_points_l,
                                                                       imagePoints2=self.stereo_charuco_points_r,
                                                                       cameraMatrix1=self.left_camera_calib_results.camera_matrix,
                                                                       distCoeffs1=self.left_camera_calib_results.dist_coeffs,
                                                                       cameraMatrix2=self.right_camera_calib_results.camera_matrix,
                                                                       distCoeffs2=self.right_camera_calib_results.dist_coeffs,
                                                                       imageSize=self.frame_size,
                                                                       criteria=self.stereo_criteria,
                                                                       flags=flags)

        np.testing.assert_array_equal(new_camera_matrix_l, self.left_camera_calib_results.camera_matrix)
        np.testing.assert_array_equal(new_dist_coeffs_l, self.left_camera_calib_results.dist_coeffs)

        np.testing.assert_array_equal(new_camera_matrix_r, self.right_camera_calib_results.camera_matrix)
        np.testing.assert_array_equal(new_dist_coeffs_r, self.right_camera_calib_results.dist_coeffs)

        # stereo rectification
        rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = cv2.stereoRectify(
            cameraMatrix1=new_camera_matrix_l,
            distCoeffs1=new_dist_coeffs_l,
            cameraMatrix2=new_camera_matrix_r,
            distCoeffs2=new_dist_coeffs_r,
            imageSize=self.frame_size[::-1],
            R=rot,
            T=trans,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)

        stereo_rectify_map_l_x, stereo_rectify_map_l_y = cv2.initUndistortRectifyMap(cameraMatrix=new_camera_matrix_l,
                                                                                     distCoeffs=new_dist_coeffs_l,
                                                                                     R=rect_l,
                                                                                     newCameraMatrix=proj_matrix_l,
                                                                                     size=self.frame_size[::-1],
                                                                                     m1type=cv2.CV_16SC2)

        stereo_rectify_map_r_x, stereo_rectify_map_r_y = cv2.initUndistortRectifyMap(cameraMatrix=new_camera_matrix_r,
                                                                                     distCoeffs=new_dist_coeffs_r,
                                                                                     R=rect_r,
                                                                                     newCameraMatrix=proj_matrix_r,
                                                                                     size=self.frame_size[::-1],
                                                                                     m1type=cv2.CV_16SC2)

        self.left_camera_calib_results.stereo_rectify_map_x = stereo_rectify_map_l_x
        self.left_camera_calib_results.stereo_rectify_map_y = stereo_rectify_map_l_y
        self.right_camera_calib_results.stereo_rectify_map_x = stereo_rectify_map_r_x
        self.right_camera_calib_results.stereo_rectify_map_y = stereo_rectify_map_r_y

        logger.success("Calibration complete!")
        logger.info(f"Stereo Calibration result: RMS error = {ret_stereo:.4f}")
        self.log_calib_info()

        return StereoCalibrationData(rms_stereo_reprojection_error=ret_stereo,
                                     left_camera_calibration_data=self.left_camera_calib_results,
                                     right_camera_calibration_data=self.right_camera_calib_results,
                                     left_camera_rectification_transform=rect_l,
                                     right_camera_rectification_transform=rect_r,
                                     rot=rot,
                                     trans=trans,
                                     essential_matrix=essential_matrix,
                                     fundamental_matrix=fundamental_matrix,
                                     projection_matrix_left=proj_matrix_l,
                                     projection_matrix_right=proj_matrix_r,
                                     perspective_transformation_matrix_Q=Q)

    def log_calib_info(self):
        logger.info(f"Left Camera Calibration: RMS error = {self.left_camera_calib_results.rms_reprojection_error:.4f}")
        logger.info(
            f"Right Camera Calibration: RMS error = {self.right_camera_calib_results.rms_reprojection_error:.4f}")

        logger.info(f"Left Camera Calibration Matrix = {self.left_camera_calib_results.camera_matrix}")
        logger.info(f"Right Camera Calibration Matrix = {self.right_camera_calib_results.camera_matrix}")
