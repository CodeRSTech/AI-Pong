# src/ball.py
"""
Ball entity with 2D velocity and Arcade rendering.
"""

import arcade

from components.geometry import Vec2
from components.rectangle import Rectangle


class Ball(Rectangle):
    """
    A simple ball represented as a circle; inherits rectangle bounds for AABB.
    """

    def __init__(self, left: float, top: float, width: float, height: float):
        """
        Args:
            left: Left edge X (y-down space).
            top: Top edge Y (y-down space).
            width: Circle diameter.
            height: Circle diameter.
        """
        super().__init__(left, top, width, height)
        self.speed = Vec2(0, 0)
        self.hits = 0
        self.last_hit_by_cpu = False
        self.last_hit_by_player = False

    def update_variables(self) -> None:
        """
        Integrate velocity into position.
        """
        self.pos_x += self.speed.x
        self.pos_y += self.speed.y
        self.center = (self.pos_x, self.pos_y)

    def set_speed(self, value: Vec2) -> None:
        """
        Assign a new 2D speed vector to the ball.
        """
        self.speed = Vec2(value.x, value.y)

    def render_to(self, window_height: float) -> None:
        """
        Draw as a filled circle using Arcade (convert y-down center to y-up).

        Args:
            window_height: Current window height, used to flip y-axis for Arcade.
        """
        y_up_center = window_height - self.pos_y
        radius = self.height / 2
        arcade.draw_circle_filled(
            self.pos_x,
            y_up_center,
            radius,
            self.color
        )
        arcade.draw_circle_outline(self.pos_x, y_up_center, radius, arcade.color.BLACK)

    def is_going_up(self) -> bool:
        """
        Returns True if vertical velocity is upwards in y-down space.
        """
        return self.speed.y < 0

    def is_going_down(self) -> bool:
        """
        Returns True if vertical velocity is downwards in y-down space.
        """
        return not self.is_going_up()

    def flip_y(self) -> None:
        """Invert vertical velocity."""
        self.speed.y *= -1

    def flip_x(self) -> None:
        """Invert horizontal velocity."""
        self.speed.x *= -1
