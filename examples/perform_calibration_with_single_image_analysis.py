import sys
import cv2
import argparse
import numpy as np
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


def selected_image_analysis(calib: StereoCalibration, calib_data: utils.StereoCalibrationData) -> None:
    image_id = np.random.randint(0, len(calib.right_images_path))
    left_image_path = calib.left_images_path[image_id]
    right_image_path = calib.right_images_path[image_id]
    logger.info(f"Performing post calibration analysis on images {left_image_path} and {right_image_path}")

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(calib.right_images_path[image_id], cv2.IMREAD_GRAYSCALE)
    height, width = left_image.shape[:2]

    # plot detected left and right markers
    utils.plot_left_and_right_detected_corners(left_corner_pts=calib.stereo_charuco_points_l[image_id],
                                               right_corner_pts=calib.stereo_charuco_points_r[image_id],
                                               image_size=(width, height))

    # calculate and plot reprojection errors for selected image
    for camera_data, image_pts, camera_side in zip(
            [calib_data.left_camera_calibration_data, calib_data.right_camera_calibration_data],
            [calib.stereo_charuco_points_l, calib.stereo_charuco_points_r],
            ["Left", "Right"]):
        camera_data: utils.CameraCalibrationData
        reprojection_pts, _, _ = utils.calculate_reprojection_error_per_image(
            obj_pts=calib.stereo_obj_points[image_id],
            image_pts=image_pts[image_id],
            rvec=camera_data.rotation_vectors[image_id],
            tvec=camera_data.translation_vectors[image_id],
            camera_matrix=camera_data.camera_matrix,
            dist_coeffs=camera_data.dist_coeffs
        )
        utils.plot_reprojected_over_detected_points(detected_pts=image_pts[image_id],
                                                    reprojected_pts=reprojection_pts,
                                                    image_size=camera_data.image_dim,
                                                    camera_side=camera_side)

    # rectify image and plot
    utils.combine_stereo_images(left_image, right_image, display=True, draw_lines=False)

    rectified_img = utils.stereo_rectify_and_combine_images(left_image,
                                                            right_image,
                                                            calib_data,
                                                            display=True,
                                                            draw_lines=False)

    left_rect_img, right_rect_img = rectified_img[:, :width], rectified_img[:, width:]

    # Plot disparity map and depth map
    disparity = utils.compute_disparity(left_rect_img, right_rect_img)
    depth_map = utils.compute_depth_map(disparity,
                                        disparity_to_depth_matrix=calib_data.perspective_transformation_matrix_Q)
    utils.plot_depth_map(depth_map)
    utils.plot_disparity_map(disparity)
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

    # Perform post calibration analysis on randomly selected stereo pair
    logger.info("Performing post calibration analysis on randomly selected image ...")
    selected_image_analysis(calib, calib_results)
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
