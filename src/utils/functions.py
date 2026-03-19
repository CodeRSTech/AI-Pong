# src/functions.py
"""
Game helper functions:
- genetic crossover utility
- game object factories
- ball spin/skew on paddle hit
- math helpers
"""
import time
from copy import deepcopy
import numpy as np
from numpy.random import random

from components.ball import Ball
from components.geometry import Vec2
from components.paddle import Paddle
from utils.logger import logger


class ActivationsMismatchError(SyntaxError):
    """
    Custom exception class for activations mismatch error in neural network layers.
    """

    def __init__(self, layer_index, activation1, activation2):
        self.layer_index = layer_index
        self.activation1 = activation1
        self.activation2 = activation2
        super().__init__()

    def __str__(self):
        return f"Activations mismatch in layer {self.layer_index}: '{self.activation1}' and '{self.activation2}'"

def two_point_crossover(player_1, player_2):
    """
    Two-point crossover for PyTorch-based NeuralNet.
    """

    from ga.player import IndividualPlayer
    from ga.network import NeuralNet
    import torch

    new_player = IndividualPlayer()
    new_net = NeuralNet()

    net1 = player_1.neural_net
    net2 = player_2.neural_net

    for layer1, layer2 in zip(net1.layers, net2.layers):

        linear1 = layer1[0]
        linear2 = layer2[0]

        w1 = linear1.weight.data.clone()
        w2 = linear2.weight.data.clone()
        b1 = linear1.bias.data.clone()
        b2 = linear2.bias.data.clone()

        out_features, in_features = w1.shape

        # Choose crossover rows (neurons)
        cross_points = sorted(torch.randperm(out_features)[:2].tolist())

        new_w = w1.clone()
        new_b = b1.clone()

        for i in range(out_features):
            if cross_points[0] <= i < cross_points[1]:
                new_w[i] = w2[i]
                new_b[i] = b2[i]

        # Detect activation type from second module
        activation_module = layer1[1]
        if isinstance(activation_module, torch.nn.ReLU):
            activation = "relu"
        elif isinstance(activation_module, torch.nn.Tanh):
            activation = "tanh"
        else:
            activation = "binary"

        new_net.add_layer(
            size=in_features,
            output_size=out_features,
            activation=activation
        )

        # Set weights
        new_layer_linear = new_net.layers[-1][0]
        new_layer_linear.weight.data = new_w
        new_layer_linear.bias.data = new_b

    new_player.neural_net = new_net
    return new_player

def skew_ball_direction(ball, paddle, is_cpu=False) -> None:
    """
    Skew/tilt the ball direction based on hit position on paddle.

    The farther from the paddle center, the stronger the skew.

    Args:
        ball: Ball object.
        paddle: Paddle object.
        is_cpu: If True, invert the skew (for the top paddle).
    """
    deviation = ball.centerx - paddle.centerx
    influence = quantify(deviation, paddle.width / 2)
    horizon = Vec2(1, 0)
    # Only skew if angle with horizon is significant
    if abs(ball.speed.angle_to(horizon)) > 30:
        rotation = influence * 45
        if is_cpu:
            rotation *= -1
    else:
        rotation = 0
    ball.speed.rotate_ip(rotation)


def quantify(value, factor) -> float:
    """
    Squash 'value' using tanh(value/factor) and scale it back by 'factor'.
    """
    y = np.tanh(value / factor) * factor
    return y


def timeit(func):
    """
    A decorator to time a function's execution and print the duration.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} \t=\t {(end_time - start_time) * 1000:.5f} ms")
        return result

    return wrapper


def create_paddle(screen_width, screen_height, color, is_cpu=False) -> Paddle:
    """
    Create a paddle positioned at top (CPU) or bottom (Player).

    Args:
        screen_width: Width of the screen.
        screen_height: Height of the screen.
        color: RGB tuple.
        is_cpu: If True, place at top; else bottom.
    """
    left = screen_width // 2
    top = 0 if is_cpu else screen_height - 10
    paddle = Paddle(left, top, 80, 10)
    paddle.color = color
    paddle.pos_x = left
    return paddle


def create_ball() -> Ball:
    """
    Create a new Ball object with random velocity and color.
    """
    ball = Ball(150, 150, 10, 10)
    ball.speed = Vec2(12, 0)
    ball.speed.rotate_ip(random() * 360)
    ball.color = (
        int(random() * 255),
        int(random() * 255),
        int(random() * 255)
    )
    return ball
