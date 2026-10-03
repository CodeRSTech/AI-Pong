# src/functions.py
"""
Game helper functions:
- genetic crossover utility
- game object factories
- ball spin/skew on paddle hit
- math helpers
"""
import time

import numpy as np
from numpy.random import random

from src.components import Ball, Paddle, Vec2
from src.utils import logger


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
    Crossover contiguous neurons, including each selected neuron's outgoing
    weights in the following layer.
    """

    from copy import deepcopy

    from src.ga.player import IndividualPlayer
    import torch

    net1 = player_1.neural_net
    net2 = player_2.neural_net
    if not net1.layers or len(net1.layers) != len(net2.layers):
        raise ValueError("Parents must have matching neural-network architectures.")

    for layer1, layer2 in zip(net1.layers, net2.layers):
        linear1, linear2 = layer1[0], layer2[0]
        if linear1.weight.shape != linear2.weight.shape or type(layer1[1]) is not type(layer2[1]):
            raise ValueError("Parents must have matching neural-network architectures.")

    total_neurons = sum(layer[0].out_features for layer in net1.layers)
    cross_points = sorted(torch.randperm(total_neurons + 1)[:2].tolist())
    inherited_neurons = set(range(cross_points[0], cross_points[1]))

    new_player = IndividualPlayer()
    new_net = deepcopy(net1)
    neuron_offset = 0
    previous_mask = None

    with torch.no_grad():
        for layer1, layer2, child_layer in zip(net1.layers, net2.layers, new_net.layers):
            parent1_linear = layer1[0]
            parent2_linear = layer2[0]
            child_linear = child_layer[0]
            out_features = parent1_linear.out_features
            selected = [
                neuron_offset + neuron_index in inherited_neurons
                for neuron_index in range(out_features)
            ]
            row_mask = torch.tensor(selected, dtype=torch.bool, device=child_linear.weight.device)

            child_linear.weight.copy_(parent1_linear.weight)
            child_linear.bias.copy_(parent1_linear.bias)
            child_linear.weight[row_mask] = parent2_linear.weight[row_mask]
            child_linear.bias[row_mask] = parent2_linear.bias[row_mask]

            if previous_mask is not None:
                column_mask = torch.tensor(
                    previous_mask, dtype=torch.bool, device=child_linear.weight.device
                )
                child_linear.weight[:, column_mask] = parent2_linear.weight[:, column_mask]

            previous_mask = selected
            neuron_offset += out_features

    new_net.last_input = None
    new_net.last_activations = None
    new_net.last_output_raw = None
    new_net.last_output_binary = None
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
    displacement_x = ball.center_x - paddle.center_x
    influence = displacement_x // (paddle.width / 2)
    horizon = Vec2(1, 0)

    # Only skew if angle with horizon is significant
    if abs(ball.speed.angle_to(horizon)) > 15:
        rotation = influence * 30
        if is_cpu:
            rotation *= -1
    else:
        rotation = 0
    ball.speed.rotate_ip(rotation)

    new_angle = abs(ball.speed.angle_to(horizon))

    if new_angle == 0:
        ball.speed.y = 0.2

    # FIXME: Following implementation (partially working) increases ball speed
    #  depending upon how far from centre it hit the paddle.
    #  Key issue here is that there isn't a working method to reset the speed
    #  once the ball angle re-adjusts after a fresh hit.
    # speed_multiplier = 1.0
    #if new_angle < 20:
    #    difference = 20 - new_angle
    #    speed_multiplier = (difference + 1) / 10.0
    # ball.speed.scale_ip(speed_multiplier)


def squash(value, factor) -> float:
    """
    Squash 'value' using tanh(value/factor) and scale it back by 'factor'.
    """
    y = np.tanh(value / factor) * factor
    return float(y)


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


def create_paddle(screen_width, screen_height, color, is_cpu=False, width=80) -> Paddle:
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
    paddle = Paddle(left, top, width, 10)
    paddle.color = color
    paddle.pos_x = left
    return paddle


def create_ball() -> Ball:
    """
    Create a new Ball object with random velocity and color.
    """
    ball = Ball(150, 150, 12, 12)
    ball.speed = Vec2(2, 0)
    ball.speed.rotate_ip(random() * 360)
    ball.color = (
        int(random() * 255),
        int(random() * 255),
        int(random() * 255)
    )
    return ball
