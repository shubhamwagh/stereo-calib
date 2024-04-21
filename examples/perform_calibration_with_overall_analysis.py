import sys
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from stereo_calib.charuco import CharucoBoard, CharucoBoardData
from stereo_calib.charuco import CharucoConfig as C
from stereo_calib.calibration import StereoCalibration
from stereo_calib import utils

plt.ioff()


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo Calibration")
    parser.add_argument("--data-path", type=str, help="Path to input data folder")
    parser.add_argument("--output-path", type=str, help="Path to output results folder")
    return parser.parse_args()


def overall_calibration_analysis(calib: StereoCalibration, calib_data: utils.StereoCalibrationData) -> None:
    overall_errors = {}
    for camera_data, image_pts, camera_side in zip(
            [calib_data.left_camera_calibration_data, calib.right_camera_calib_results],
            [calib.stereo_charuco_points_l, calib.stereo_charuco_points_r],
            ['Left', 'Right']):
        projected_pts, reprojection_error, rms_errors = utils.calculate_reprojection_errors_for_all_images(
            all_obj_pts=calib.stereo_obj_points,
            all_image_pts=image_pts,
            camera_calib_results=camera_data)

        utils.plot_reprojection_errors(reprojection_error, camera_side)
        overall_errors[camera_side] = rms_errors

    utils.plot_left_right_errors(left_errors=overall_errors['Left'], right_errors=overall_errors['Right'])
    # plt.show()


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

    # Perform calibration
    calib: StereoCalibration = StereoCalibration(data_path=data_path, charuco_board=charuco_board)
    calib_results: utils.StereoCalibrationData = calib.calibrate()

    # Perform post calibration overall analysis
    logger.info("Performing post calibration analysis ...")
    overall_calibration_analysis(calib, calib_results)
    logger.success("Analysis successfully completed!")

    # save calibration data
    utils.save_calibration_data(calib_results, output_dir)
    logger.info(f"Calibration results saved to {str(output_dir)}.")

    # display all plots
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(0)
