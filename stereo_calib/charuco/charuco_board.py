import cv2
from dataclasses import dataclass

from stereo_calib.charuco import constants as CharucoConfig


@dataclass
class CharucoBoardData:
    """
    A data class to hold Charuco board configuration.

    Attributes:
        aruco_dict (int): Aruco dictionary type.
        squares_vertically (int): Number of squares along the y-axis.
        squares_horizontally (int): Number of squares along the x-axis.
        square_length (float): Length of each square in meters.
        marker_length (float): Length of each marker in meters.
        board_image_size (tuple): Desired board image size in pixels (width, height).
        margin_px (int): Size of the margin in pixels.
    """
    aruco_dict: int = CharucoConfig.ARUCO_DICT
    squares_vertically: int = CharucoConfig.SQUARES_VERTICALLY
    squares_horizontally: int = CharucoConfig.SQUARES_HORIZONTALLY
    square_length: float = CharucoConfig.SQUARE_LENGTH
    marker_length: float = CharucoConfig.MARKER_LENGTH
    board_image_size: tuple = CharucoConfig.BOARD_IMAGE_SIZE
    margin_px: int = CharucoConfig.MARGIN_PX


class CharucoBoard:
    """
    A class to generate Charuco boards.

    Attributes:
        charuco_data (CharucoBoardData): Configuration data for the Charuco board.
        aruco_dict: Initialized Aruco dictionary.
        board: Generated Charuco board.
    """

    def __init__(self, charuco_data: CharucoBoardData):
        """
        Initializes the CharucoBoard instance.

        Parameters:
            charuco_data (CharucoBoardData): Configuration data for the Charuco board.
        """
        self.charuco_data = charuco_data
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(charuco_data.aruco_dict)
        board_size = (charuco_data.squares_horizontally, charuco_data.squares_vertically)
        self.board = cv2.aruco.CharucoBoard(board_size,
                                            charuco_data.square_length,
                                            charuco_data.marker_length,
                                            self.aruco_dict)
        self.board.setLegacyPattern(True)

    def generate_board_image(self):
        """
        Generates a Charuco board image.

        Returns:
            np.ndarray: The generated Charuco board image.
        """
        return self.board.generateImage(self.charuco_data.board_image_size, marginSize=self.charuco_data.margin_px)


if __name__ == "__main__":
    board = CharucoBoard(CharucoBoardData())
    img = board.generate_board_image()
    cv2.imshow("board", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
