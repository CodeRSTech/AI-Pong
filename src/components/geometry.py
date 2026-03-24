# src/geometry.py
"""
Minimal 2D vector used as a replacement for pygame.Vector2.
"""

import numpy as np


class Vec2:
    """
    Minimal 2D vector for this project:
    - rotate_ip(degrees): in-place rotation (degrees, counter-clockwise)
    - angle_to(other): signed angle to another vector (degrees)
    - magnitude(): vector length
    """

    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def rotate_ip(self, degrees: float) -> None:
        rad = np.deg2rad(degrees)
        c, s = np.cos(rad), np.sin(rad)
        x, y = self.x, self.y
        self.x = x * c - y * s
        self.y = x * s + y * c

    def angle_to(self, other: "Vec2") -> float:
        dot = self.x * other.x + self.y * other.y
        mag = self.magnitude() * other.magnitude()
        if mag == 0.0:
            return 0.0
        cosang = max(min(dot / mag, 1.0), -1.0)
        ang = np.rad2deg(np.arccos(cosang))
        # Determine sign using z-component of 2D cross product
        cross_z = self.x * other.y - self.y * other.x
        return -ang if cross_z < 0 else ang

    def magnitude(self) -> float:
        return float(np.hypot(self.x, self.y))

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)
