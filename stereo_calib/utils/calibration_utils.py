import json
import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from json import JSONEncoder
from loguru import logger
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from typing import List, Tuple, Union, Optional


@dataclass_json
@dataclass
class CameraCalibrationData:
    """
    A data class to hold camera calibration information.

    Attributes:
        rms_reprojection_error (float): The root mean square (RMS) reprojection error.
        camera_matrix (np.ndarray): The camera matrix.
        dist_coeffs (np.ndarray): The distortion coefficients.
        rotation_vectors (List[np.ndarray]): List of rotation vectors for each calibration image.
        translation_vectors (List[np.ndarray]): List of translation vectors for each calibration image.
        image_width (int): The width of the calibration image.
        image_height (int): The height of the calibration image.
        image_dim (Tuple[int, int]): The dimensions of the calibration image [width, height].
        undistort_map_x (Optional[np.ndarray]): Optional map for undistortion in x-direction.
        undistort_map_y (Optional[np.ndarray]): Optional map for undistortion in y-direction.
        stereo_rectify_map_x (Optional[np.ndarray]): Optional map for stereo rectification in x-direction.
        stereo_rectify_map_y (Optional[np.ndarray]): Optional map for stereo rectification in y-direction.
    """
    rms_reprojection_error: float
    camera_matrix: np.ndarray = field(metadata=config(decoder=np.asarray))
    dist_coeffs: np.ndarray = field(metadata=config(decoder=np.asarray))
    rotation_vectors: List[np.ndarray] = field(metadata=config(decoder=lambda l: [np.asarray(item) for item in l]))
    translation_vectors: List[np.ndarray] = field(metadata=config(decoder=lambda l: [np.asarray(item) for item in l]))
    image_width: int
    image_height: int
    image_dim: Tuple[int, int]
    undistort_map_x: Optional[np.ndarray] = None
    undistort_map_y: Optional[np.ndarray] = None
    stereo_rectify_map_x: Optional[np.ndarray] = None
    stereo_rectify_map_y: Optional[np.ndarray] = None


# Define data class for stereo calibration results
@dataclass_json
@dataclass
class StereoCalibrationData:
    """
    A data class to hold stereo calibration information.

    Attributes:
        rms_stereo_reprojection_error (float): The root mean square (RMS) stereo reprojection error.
        left_camera_calibration_data (CameraCalibrationData): Calibration data for the left camera.
        right_camera_calibration_data (CameraCalibrationData): Calibration data for the right camera.
        projection_matrix_left (np.ndarray): Projection matrix for the left camera.
        projection_matrix_right (np.ndarray): Projection matrix for the right camera.
        left_camera_rectification_transform (np.ndarray): Rectification transform for the left camera.
        right_camera_rectification_transform (np.ndarray): Rectification transform for the right camera.
        rot (np.ndarray): The rotation matrix between the two cameras.
        trans (np.ndarray): The translation vector between the two cameras.
        essential_matrix (np.ndarray): The essential matrix.
        fundamental_matrix (np.ndarray): The fundamental matrix.
        perspective_transformation_matrix_Q (np.ndarray): Perspective transformation matrix.
    """
    rms_stereo_reprojection_error: float
    left_camera_calibration_data: CameraCalibrationData
    right_camera_calibration_data: CameraCalibrationData
    projection_matrix_left: np.ndarray = field(metadata=config(decoder=np.asarray))
    projection_matrix_right: np.ndarray = field(metadata=config(decoder=np.asarray))
    left_camera_rectification_transform: np.ndarray = field(metadata=config(decoder=np.asarray))
    right_camera_rectification_transform: np.ndarray = field(metadata=config(decoder=np.asarray))
    rot: np.ndarray = field(metadata=config(decoder=np.asarray))
    trans: np.ndarray = field(metadata=config(decoder=np.asarray))
    essential_matrix: np.ndarray = field(metadata=config(decoder=np.asarray))
    fundamental_matrix: np.ndarray = field(metadata=config(decoder=np.asarray))
    perspective_transformation_matrix_Q: np.ndarray = field(metadata=config(decoder=np.asarray))


class CustomJSONEncoder(JSONEncoder):
    """
    A custom JSON encoder to handle special data types like numpy arrays, lists, and tuples.
    Overrides the default() method of JSONEncoder to handle custom serialization.
    Methods:
        default(obj): Override of the default() method to handle custom serialization.
    Attributes:
        None
    """

    def default(self, obj):
        """
        Override the default() method of JSONEncoder to handle custom serialization.
        Args:
            obj: The object to be serialized.
        Returns:
            Serialized JSON-compatible representation of the object.
        Raises:
            None
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.default(item) for item in obj]
        if isinstance(obj, tuple):
            return list(obj)
        # If not a special type, use default serialization
        return JSONEncoder.default(self, obj)


def save_calibration_data(data: StereoCalibrationData, file_path: Union[str, Path]) -> None:
    """
    Save stereo calibration data to a JSON file.
    Args:
        data (StereoCalibrationData): Stereo calibration data to be saved.
        file_path (Union[str, Path]): File path where the calibration data will be saved.

    Returns:
        None
    """
    logger.info("Saving calibration results ...")

    data_to_save = deepcopy(data)

    # setting undistort and rectify map to None and only save useful calib data
    for camera_data in [data_to_save.left_camera_calibration_data, data_to_save.right_camera_calibration_data]:
        camera_data: CameraCalibrationData
        camera_data.undistort_map_x = None
        camera_data.undistort_map_y = None
        camera_data.stereo_rectify_map_x = None
        camera_data.stereo_rectify_map_y = None

    data_dict = data_to_save.to_dict()

    if isinstance(file_path, str):
        file_path = Path(file_path)

    file_path.mkdir(exist_ok=True, parents=True)
    with open(str(file_path.joinpath("calibration_results.json")), 'w') as file:
        json.dump(data_dict, file, cls=CustomJSONEncoder, indent=4)
    logger.success("Calibration results successfully saved!")


def load_stereo_rectify_maps(data: StereoCalibrationData) -> StereoCalibrationData:
    """
    Load stereo rectify maps and update the StereoCalibrationData object.

    This function loads stereo rectify maps for each camera and updates the given StereoCalibrationData
    object with the computed rectify maps.

    Args:
        data (StereoCalibrationData): Stereo calibration data containing camera calibration information.

    Returns:
        StereoCalibrationData: Updated stereo calibration data object with stereo rectify maps.
    """
    for camera_data, rect, projection_mat in zip(
            [data.left_camera_calibration_data, data.right_camera_calibration_data],
            [data.left_camera_rectification_transform, data.right_camera_rectification_transform],
            [data.projection_matrix_left, data.projection_matrix_right]):
        camera_data: CameraCalibrationData
        rect: np.ndarray
        projection_mat: np.ndarray

        stereo_rectify_map_x, stereo_rectify_map_y = cv2.initUndistortRectifyMap(cameraMatrix=camera_data.camera_matrix,
                                                                                 distCoeffs=camera_data.dist_coeffs,
                                                                                 R=rect,
                                                                                 newCameraMatrix=projection_mat,
                                                                                 size=camera_data.image_dim,
                                                                                 m1type=cv2.CV_16SC2)

        camera_data.stereo_rectify_map_x = stereo_rectify_map_x
        camera_data.stereo_rectify_map_y = stereo_rectify_map_y

    return data


def load_calibration_data(file_path: Union[str, Path]) -> StereoCalibrationData:
    """
    Load stereo calibration data from a JSON file.

    This function loads stereo calibration data from a JSON file at the specified file path,
    and returns the corresponding StereoCalibrationData object.

    Args:
        file_path (Union[str, Path]): File path to the JSON file containing stereo calibration data.

    Returns:
        StereoCalibrationData: Stereo calibration data loaded from the JSON file.

    Raises:
        FileNotFoundError: If the specified file path does not exist or is not a file.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if file_path.is_file() and file_path.exists():
        with open(file_path, 'r') as file:
            data = json.load(file)
        stereo_calib_data = StereoCalibrationData.from_dict(data)
        stereo_calib_data = load_stereo_rectify_maps(stereo_calib_data)
        return stereo_calib_data
    else:
        raise FileNotFoundError(f"File '{str(file_path)}' not found or is not a file.")
