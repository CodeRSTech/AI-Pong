# src/rectangle.py
"""
A minimal rectangle with AABB collision and Arcade rendering support.

Note on coordinates:
- The simulation/game logic uses a y-down coordinate system (origin top-left) to
  mirror the original project behavior.
- Arcade renders in y-up space (origin bottom-left). We flip y at draw time.
"""

from components.colors import BLACK
import arcade


class Rectangle:
    """
    Simple axis-aligned rectangle.

    Attributes:
        pos_x (float): X coordinate of center in y-down space.
        pos_y (float): Y coordinate of center in y-down space.
        width (float)
        height (float)
        color (RGB tuple)
    """

    def __init__(self, left: float, top: float, width: float, height: float):
        """
        Initialize with a pygame-like signature: left, top, width, height.

        Args:
            left: Left edge X (y-down space).
            top: Top edge Y (y-down space).
            width: Rectangle width.
            height: Rectangle height.
        """
        self.width = float(width)
        self.height = float(height)
        self.pos_x = float(left + width / 2)
        self.pos_y = float(top + height / 2)
        self.color = BLACK

    # Geometry properties (computed from center)
    @property
    def left(self) -> float:
        return self.pos_x - self.width / 2

    @property
    def right(self) -> float:
        return self.pos_x + self.width / 2

    @property
    def top(self) -> float:
        # Top edge in y-down coordinates (smaller y)
        return self.pos_y - self.height / 2

    @property
    def bottom(self) -> float:
        # Bottom edge in y-down coordinates (larger y)
        return self.pos_y + self.height / 2

    @property
    def center(self) -> tuple[float, float]:
        return (self.pos_x, self.pos_y)

    @center.setter
    def center(self, value: tuple[float, float]) -> None:
        self.pos_x, self.pos_y = float(value[0]), float(value[1])

    @property
    def centerx(self) -> float:
        return self.pos_x

    @property
    def centery(self) -> float:
        return self.pos_y

    def colliderect(self, other: "Rectangle") -> bool:
        """
        Axis-aligned bounding-box collision with another rectangle (y-down space).
        """
        return not (
            self.right < other.left or
            self.left > other.right or
            self.bottom < other.top or
            self.top > other.bottom
        )

    def update_variables(self) -> None:
        """
        Hook for subclasses that update position; kept for interface compatibility.
        """
        # No-op for base rectangle
        pass

    def render_to(self, window_height: float) -> None:
        """
        Draw as a filled rectangle using Arcade (convert y-down to y-up).

        Args:
            window_height: Current window height, used to flip y-axis for Arcade.
        """
        # Convert y-down edges to y-up edges
        left = self.left
        right = self.right
        top_y_up = window_height - self.top
        bottom_y_up = window_height - self.bottom

        # Define corners in CCW order
        points = [
            (left,  top_y_up),     # top-left
            (right, top_y_up),     # top-right
            (right, bottom_y_up),  # bottom-right
            (left,  bottom_y_up),  # bottom-left
        ]
        arcade.draw_polygon_filled(points, self.color)