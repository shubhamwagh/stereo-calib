import sys
import argparse
from pathlib import Path
from loguru import logger

from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C
from stereo_calib.calibration import StereoCalibration
from stereo_calib import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo Calibration")
    parser.add_argument("--data-path", type=str, help="Path to input data folder")
    parser.add_argument("--output-path", type=str, help="Path to output results folder")
    return parser.parse_args()


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

    utils.save_rectified_stereo_images_with_disparity_and_depth_maps(calib.left_images_path,
                                                                     calib.right_images_path,
                                                                     calib_results,
                                                                     output_dir)
    utils.save_calibration_data(calib_results, output_dir)

    logger.info(f"Calibration results saved to {str(output_dir)}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(0)
