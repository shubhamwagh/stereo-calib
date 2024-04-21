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
  <a href="#results">Results</a> •
  <a href="#references">References</a> 
</p>

## Introduction

In this report, we focus on the calibration and analysis of a horizontal stereo camera, including the
estimation of intrinsic and extrinsic parameters, rectification of images, and the application of stereo rectification
to generate a disparity and depth map.

The process encompasses:

* **Intrinsic Parameter Estimation**: Determining the internal characteristics of the camera, such as focal length,
  optical centre and distortion coefficients, which are crucial for converting camera coordinates to pixel coordinates.
* **Extrinsic Parameter Estimation**: Converting world coordinates to camera coordinates, including the rotation
  matrix (R) and translation matrix (T).
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

* **Intrinsic Parameters**: The camera matrix ($K$) includes the focal lengths ($f_x$) and ($f_y$) along the x and y axes, and
  the principal point coordinates (($c_x$, $c_y$)).
$K = \begin{bmatrix} f_x & 0 & c_x \\\ 0 & f_y & c_y \\\ 0 & 0 & 1 \end{bmatrix}$
