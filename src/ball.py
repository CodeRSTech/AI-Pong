import pygame

from rectangle import Rectangle
from pygame import Vector2, draw


class Ball(Rectangle):
    """
    A class representing a ball object in a game.
    Inherits from the Rectangle class.
    """

    def __init__(self, left, top, width, height):
        """
        Initialize the Ball object with the given position and size.

        Args:
        left (int): The left coordinate of the ball.
        top (int): The top coordinate of the ball.
        width (int): The width of the ball.
        height (int): The height of the ball.
        """
        super().__init__(left, top, width, height)
        self.speed = Vector2(0, 0)

    def update_variables(self) -> None:
        """
        Update the position of the ball based on its speed.
        """
        self.pos_x += self.speed.x
        self.pos_y += self.speed.y
        self.center = (self.pos_x, self.pos_y)

    def set_speed(self, value: Vector2) -> None:
        """
        Set the speed of the ball.

        Args:
        value (Vector2): The speed vector of the ball.
        """
        self.speed = Vector2(value)

    def render_to(self, screen: pygame.Surface) -> None:
        """
        Render the ball on the screen.

        Args:
        screen (pygame.Surface): The screen where the ball will be rendered.
        """
        draw.circle(
            screen,
            self.color,
            (self.pos_x, self.pos_y),
            self.height / 2)

    def is_going_up(self) -> bool:
        """
        Check if the ball is moving upwards.

        Returns:
        bool: True if the ball is moving upwards, False otherwise.
        """
        return self.speed.y < 0

    def is_going_down(self) -> bool:
        """
        Check if the ball is moving downwards.

        Returns:
        bool: True if the ball is moving downwards, False otherwise.
        """
        return not self.is_going_up()

    def flip_y(self) -> None:
        """
        Reverse the vertical direction of the ball's speed.
        """
        self.speed.y *= -1

    def flip_x(self) -> None:
        """
        Reverse the horizontal direction of the ball's speed.
        """
        self.speed.x *= -1
