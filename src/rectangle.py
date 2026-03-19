from pygame import Rect, draw
from pygame.color import Color

BLACK = Color(0, 0, 0)


class Rectangle(Rect):
    """
    A class representing a rectangle object in a game.
    Inherits from the Rect class in pygame.

    Attributes:

    - color (pygame.Color): Color of the rectangle.

    - pos_x (float): X-coordinate of the center of the rectangle.

    - pos_y (float): Y-coordinate of the center of the rectangle.
    """

    def __init__(self, left, top, width, height):
        """
        Initialize the Rectangle object with the given position and size.

        Args:
        left (int): The left coordinate of the rectangle.
        top (int): The top coordinate of the rectangle.
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.
        """
        super().__init__(left, top, width, height)
        self.color = BLACK
        self.pos_x, self.pos_y = self.center

    def render_to(self, screen) -> None:
        """
        Render the rectangle on the screen if it is visible.

        Args:
        screen: The screen where the rectangle will be rendered.
        """
        draw.rect(screen, self.color, self)
