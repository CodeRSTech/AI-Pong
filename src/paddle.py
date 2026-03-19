from rectangle import Rectangle


class Paddle(Rectangle):
    """
    A class representing a paddle object in a game.
    Based on the Rectangle class.

    Attributes:
    - pos_x: X-coordinate of the paddle.
    - pos_y: Y-coordinate of the paddle.
    """

    def __init__(self, left, top, width, height):
        """
        Initialize the Paddle object with the given position and size.

        Args:
        left (int): The left coordinate of the paddle.
        top (int): The top coordinate of the paddle.
        width (int): The width of the paddle.
        height (int): The height of the paddle.
        """
        super().__init__(left, top, width, height)

    def update_variables(self) -> None:
        """
        Update the center position of the paddle based on its current position.
        """
        self.center = (self.pos_x, self.pos_y)
