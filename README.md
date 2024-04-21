<h1 align="center">
  <br>
  Stereo Calib
  <br>
</h1>

<h4 align="center">Calibration for Stereo Cameras</h4>

<p align="center">
  <a href="#description">Description</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#system-details">System Details</a> •
  <a href="#results">Results</a> •
  <a href="#references">References</a> 
</p>

## Description

Stereo Calib is a project dedicated to performing stereo camera calibration using Charuco boards, followed by
rectification, disparity map generation, and depth map estimation.

## Getting Started

### Setup

* Install [poetry](https://python-poetry.org/docs/#installation)
* Navigate to the project directory: `cd stereo-calib`
* Install necessary dependencies:

```commandline
poetry config virtualenvs.in-project true                  
poetry install 
```

### Perform Calibration : Save Rectified Images, Disparity and Depth Maps

To conduct stereo camera calibration, execute the following command:

```commandline
poetry run python -m examples.perform_calibration --data-path "./dataset" 
```

This script generates rectified stereo images along with disparity and depth maps, saving the calibration results to a
**results** folder.

### Perform Calibration: Estimate Distance Between Camera And Image Centre

To execute stereo camera calibration and estimate the distance between the camera and the image center, run the
following command:

```commandline
poetry run python -m examples.perform_calibration_and_estimate_distance --data-path "./dataset"
```

This scripts saves calibration results to **results** folder outputs the computed distances in millimeters to a JSON
file.

### Perform Calibration: Overall Calibration Analysis

To execute stereo camera calibration with overall calibration analysis, run the following
command:

```commandline
poetry run python -m examples.perform_calibration_with_overall_analysis --data-path "./dataset"             
```

This script conducts post-calibration analysis, including calculating and plotting reprojection errors and root mean
square (RMS) errors for both left and right camera views. Finally, it saves the calibration results and displays the
analysis plots.

### Perform Calibration: Single Image Analysis

To execute stereo camera calibration with single image analysis, run the following command:

```commandline
poetry run python -m examples.perform_calibration_with_single_image_analysis --data-path "./dataset"
```

This script randomly selects a stereo image pair, analyzes it by plotting detected markers, calculating and plotting
reprojection errors, and visualizing rectified images, disparity maps, and depth maps. Finally, it saves the calibration
results and displays the analysis plots.