# src/paddle.py
"""
Paddle entity rendered as a rectangle.
"""

from components.rectangle import Rectangle


class Paddle(Rectangle):
    """
    Player/CPU paddle (inherits basic rectangle behavior and rendering).
    """

    def __init__(self, left, top, width, height):
        super().__init__(left, top, width, height)

    def update_variables(self) -> None:
        """
        Keep center in sync with current position.
        """
        self.center = (self.pos_x, self.pos_y)


