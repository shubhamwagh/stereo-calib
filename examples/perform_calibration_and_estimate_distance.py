import sys
import cv2
import json
import argparse
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from typing import Dict

from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C
from stereo_calib.calibration import StereoCalibration
from stereo_calib import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo Calibration")
    parser.add_argument("--data-path", type=str, help="Path to input data folder")
    parser.add_argument("--output-path", type=str, help="Path to output results folder")
    return parser.parse_args()


def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def estimate_distance_between_camera_and_image_centre(calib: StereoCalibration,
                                                      calib_data: utils.StereoCalibrationData) -> Dict[str, float]:
    centre_distances: Dict[str, float] = {}
    for left_img_path, right_img_path in tqdm(zip(calib.left_images_path, calib.right_images_path),
                                              total=len(calib.left_images_path),
                                              desc="Estimating distances between camera and image center"):
        left_img = cv2.imread(left_img_path, 0)
        right_img = cv2.imread(right_img_path, 0)

        height, width = left_img.shape[:2]

        rectified_img = utils.stereo_rectify_and_combine_images(left_img, right_img, calib_data)
        left_rectified_img, right_rectified_img = rectified_img[:, :width], rectified_img[:, width:]

        disparity = utils.compute_disparity(left_rectified_img,
                                            right_rectified_img,
                                            min_disparity=8,
                                            num_disparity=16 * 7)
        depth_map = utils.compute_depth_map(disparity, calib_data.perspective_transformation_matrix_Q)

        centre_distance = depth_map[height // 2, width // 2]
        centre_distances[
            str(Path(left_img_path).stem + '_' + Path(right_img_path).stem)] = centre_distance * 1000.0  # in mm
    return centre_distances


def main():
    args = parse_args()

    # Input data path
    data_path = Path(args.data_path).resolve() if args.data_path else Path(__file__).resolve().parent.parent / "dataset"

    # Calibration results data path
    output_dir = Path(args.output_path).resolve() if args.output_path else \
        Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)

    charuco_board = CharucoBoard(charuco_data=CharucoBoardData(aruco_dict=C.ARUCO_DICT,
                                                               squares_vertically=C.SQUARES_VERTICALLY,
                                                               squares_horizontally=C.SQUARES_HORIZONTALLY,
                                                               square_length=C.SQUARE_LENGTH,
                                                               marker_length=C.MARKER_LENGTH))

    calib: StereoCalibration = StereoCalibration(data_path=data_path, charuco_board=charuco_board)
    calib_results: utils.StereoCalibrationData = calib.calibrate()

    utils.save_calibration_data(calib_results, output_dir)
    logger.info(f"Calibration results saved to {str(output_dir)}.")

    distance_json_path = output_dir / "centre_image_distances_in_mm.json"
    distances_dict = estimate_distance_between_camera_and_image_centre(calib, calib_data=calib_results)
    save_json(distances_dict, str(distance_json_path))
    logger.success("Successfully computed distances (mm) between camera and image centre!")
    logger.info(f"Saved computed distances (mm) between camera and image centre to {str(distance_json_path)}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(0)
