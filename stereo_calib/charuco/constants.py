import cv2
from typing import Tuple

# Aruco marker dictionary
ARUCO_DICT: int = cv2.aruco.DICT_4X4_250

# Number of squares in the vertical and horizontal directions on the calibration board
SQUARES_VERTICALLY: int = 16
SQUARES_HORIZONTALLY: int = 31

# Length of each square on the calibration board and length of the markers (in metres)
SQUARE_LENGTH: float = 0.04933
MARKER_LENGTH: float = 0.03846

# Size of the calibration board image
BOARD_IMAGE_SIZE: Tuple[int, int] = (2100, 1100)

# Size of the margin in pixels
MARGIN_PX: int = 0
