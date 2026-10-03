import pytest

from src.components import Ball, Vec2
from src.components.rectangle import Rectangle
from src.utils.functions import squash


def test_rectangle_edges_use_y_down_space():
    r = Rectangle(10, 20, 30, 40)
    assert (r.left, r.right, r.top, r.bottom) == (10, 40, 20, 60)
    assert r.center == (25, 40)


def test_rectangle_collision():
    a = Rectangle(0, 0, 10, 10)
    assert a.collide_rect(Rectangle(5, 5, 10, 10))
    assert not a.collide_rect(Rectangle(20, 20, 5, 5))


def test_ball_flip_and_motion():
    ball = Ball(10, 10, 5, 5)
    assert (ball.speed.x, ball.speed.y) == (0, 0)
    ball.speed = Vec2(2, -3)
    ball.flip_y()
    ball.flip_x()
    assert (ball.speed.x, ball.speed.y) == (-2, 3)
    x, y = ball.pos_x, ball.pos_y
    ball.update_variables()
    assert (ball.pos_x, ball.pos_y) == (x - 2, y + 3)
    assert ball.is_going_down()


def test_vec2_rotation_and_magnitude():
    v = Vec2(1, 0)
    v.rotate_ip(90)
    assert v.x == pytest.approx(0, abs=1e-9)
    assert v.y == pytest.approx(1)
    assert v.magnitude() == pytest.approx(1)
    assert Vec2(1, 0).angle_to(Vec2(0, 1)) == pytest.approx(90)


def test_squash_is_bounded_and_monotonic():
    assert squash(0, 10) == 0
    assert squash(1000, 10) == pytest.approx(10)
    assert squash(1, 10) < squash(2, 10)
