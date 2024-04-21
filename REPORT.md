<h1 align="center">
  <br>
  Stereo Calib
  <br>
</h1>

<h4 align="center">Calibration for Stereo Cameras</h4>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#intrinsic-parameter-estimation">Intrinsic Parameter Estimation</a> •
  <a href="#extrinsic-parameter-estimation">Extrinsic Parameter Estimation</a> •
  <a href="#references">References</a> 
</p>

## Introduction

In this report, we focus on the calibration and analysis of a horizontal stereo camera, including the
estimation of intrinsic and extrinsic parameters, rectification of images, and the application of stereo rectification
to generate a disparity and depth map.

The process encompasses:

* **Intrinsic Parameter Estimation**: Determining the internal characteristics of the camera, such as focal length,
  optical centre and distortion coefficients, which are crucial for converting camera coordinates to pixel coordinates.
* **Extrinsic Parameter Estimation**: Determining the rotation
  matrix ($R$) and translation matrix ($T$) of the left camera coordinate system with respect to the right camera coordinate
  system.
* **Image Rectification**: Aligning the optical axes of the two cameras to ensure accurate stereo vision by making the
  images line-aligned.
* **Disparity Map Calculation**: Computing the disparity map, which represents the difference in the x-coordinate of the
  same point in the left and right images, essential for depth estimation.
* **Depth Map Calculation**: Utilizing the disparity map and the baseline to compute the depth map, providing depth
  information for each pixel in the image.

## Dataset

The dataset provided includes **39 stereo image pairs** captured by the wide angle stereo camera system, featuring
synchronized left and right images. These images contain Charuco calibration patterns, a combination of Chessboard and
ArUco markers, which are essential for the calibration process.

## Intrinsic Parameter Estimation

### Justification of Camera Model

Given the dataset's origin from a wide-angle stereo camera system, the rational polynomial distortion
model (`cv2.CALIB_RATIONAL_MODEL`) is selected. This distortion camera model is known for its ability to capture complex
distortions, both radial and tangential, common in real-world imaging. This model extends the capabilities of simpler
models like the pinhole camera
model by incorporating additional distortion parameters, aiming to achieve higher calibration accuracy and enhancing
image quality.

### Equations

* **Intrinsic Parameters**: The camera matrix ($K$) includes the focal lengths ($f_x$) and ($f_y$) along the x and y
  axes, and
  the principal point coordinates (($c_x$, $c_y$)).

```math
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
```

* **Distortion Parameters**: Radial distortion coefficients ($k_1$, $k_2$, $k_3$) and tangential distortion
  coefficients ($p_1$, $p_2$) are used to correct for image distortions caused by lens imperfections.
* **Rational Model**: The rational model extends the distortion model by introducing additional coefficients ($k_4$,
  $k_5$, $k_6$) to accommodate higher-order distortions, enhancing the model's ability to correct for complex lens
  distortions.
* **Optimisation**: The calibration process optimises both intrinsic and distortion parameters using iterative
  algorithms (based on least-squares method) like Levenberg-Marquardt to minimize reprojection error. We
  use `cv2.aruco.calibrateCameraCharuco` that leverages Charuco detected corners to ensure accurate parameter
  estimation.

For robust calibration, it is recommended to detect a minimum of 5 to 10 corners (points) per image on the Charuco
board.
Given the dataset consisting of 39 stereo pairs of images, detecting more points per image will enhance the accuracy and
robustness of the camera calibration. Additionally, we ensure the utilization of Charuco corner points that are commonly
visible by both cameras, thereby reinforcing the consistency and reliability of the calibration process.

