from copy import deepcopy
import numpy as np
import pygame as pg
from numpy.random import random
from ball import Ball
from paddle import Paddle


class ActivationsMismatchError(SyntaxError):
    """
    Custom exception class for activations mismatch error in neural network layers.
    """

    def __init__(self, layer_index, activation1, activation2):
        self.layer_index = layer_index
        self.activation1 = activation1
        self.activation2 = activation2
        super.__init__()

    def __str__(self):
        return f"Activations mismatch in layer {self.layer_index}: '{self.activation1}' and '{self.activation2}'"


def binary(z_array) -> np.ndarray:
    """
    Compute the binary step activation function for an array.

    Args:
    z_array (numpy.array): Input array.

    Returns:
    numpy.ndarray: Result of applying the binary step function to the input array.
    """
    threshold = 0.8
    result = z_array >= threshold

    return result


def relu(z_array) -> np.ndarray:
    """
    Compute the ReLU activation function for an array.

    Args:
    z_array (numpy.array): Input array.

    Returns:
    numpy.ndarray: Result of applying the ReLU function to the input array.
    """
    result = []
    for zi in z_array[0]:
        result.append(max(zi, 0.0))
    return np.array([result])


def two_point_crossover(player_1, player_2):
    """
    Perform two-point crossover between two neural network players.

    Args:
    player_1 (IndividualPlayer): First player for crossover.
    player_2 (IndividualPlayer): Second player for crossover.

    Returns:
    IndividualPlayer: New player resulting from the crossover operation.
    """
    from network import NeuralNet
    from player import IndividualPlayer

    # Create a new IndividualPlayer, NeuralNet object to store the result
    new_player = IndividualPlayer()
    neural_net = NeuralNet()

    net_1 = deepcopy(player_1.neural_net)
    net_2 = deepcopy(player_2.neural_net)

    # How many layers
    num_layers = len(net_1.layers)

    for layer_index in range(num_layers):
        # Layers --------------------------
        layer_1 = net_1.layers[layer_index]
        layer_2 = net_2.layers[layer_index]

        # Activations --------------------------
        activation_1 = layer_1.activation
        activation_2 = layer_2.activation

        if activation_1 != activation_2:
            raise ActivationsMismatchError(layer_index, activation_1, activation_2)

        # Weights --------------------------
        weights_1 = layer_1.weights.transpose()
        weights_2 = layer_2.weights.transpose()

        # Biases --------------------------
        biases_1 = layer_1.biases[0]
        biases_2 = layer_2.biases[0]

        # Rows, Columns length --------------------------
        num_rows, num_cols = weights_1.shape

        # Determine cross-over points --------------------------
        cross_points = sorted(np.random.choice(num_rows, 2, replace=False))

        new_weights = []
        new_biases = []

        for i in range(num_rows):
            if cross_points[0] <= i < cross_points[1]:
                new_weights.append(weights_2[i])
                new_biases.append(biases_2[i])
            else:
                new_weights.append(weights_1[i])
                new_biases.append(biases_1[i])

        weights = np.array(new_weights).transpose()
        biases = np.array([new_biases])

        neural_net.add_layer(size=num_cols, output_size=num_rows, activation=activation_1,
                             weights=weights,
                             biases=biases)

    new_player.neural_net = neural_net
    return new_player


def create_ball() -> Ball:
    """
    Create a new Ball object with random attributes.

    Returns:
    Ball: A new Ball object.
    """
    ball = Ball(150, 150, 10, 10)
    ball.speed = pg.Vector2(4, 0)
    ball.speed.rotate_ip(random() * 360)
    ball.color = [random() * 255,
                  random() * 255,
                  random() * 255]
    return ball


def create_paddle(screen_width, screen_height, ball_color, is_cpu=False) -> Paddle:
    """
    Create a new Paddle object with specified attributes.

    Args:
    screen_width (int): Width of the screen.
    screen_height (int): Height of the screen.
    ball_color (list): RGB color values for the paddle.
    is_cpu (bool): Flag to determine if the paddle is controlled by the CPU.

    Returns:
    Paddle: A new Paddle object.
    """
    left = screen_width // 2
    top = 0 if is_cpu else screen_height - 10
    paddle = Paddle(left, top, 100, 10)
    paddle.color = ball_color
    paddle.pos_x = left

    return paddle


def skew_ball_direction(ball, paddle, is_cpu=False) -> None:
    """
    Adjust the direction of the ball based on the position of the paddle.

    Args:
    ball (Ball): The ball object.
    paddle (Paddle): The paddle object.
    is_cpu (bool): Flag to determine if the paddle is controlled by the CPU.
    """
    deviation = ball.centerx - paddle.centerx
    influence = quantify(deviation, paddle.width / 2)
    horizon = pg.Vector2(1, 0)
    if abs(ball.speed.angle_to(horizon)) > 30:  # if ball makes more than 30 deg angle with horizon
        rotation = influence * 45
        if is_cpu:
            rotation *= -1
    else:
        rotation = 0
    ball.speed.rotate_ip(rotation)


def quantify(value, factor) -> float:
    """
    Quantify a value using tanh on value/factor and multiply it by a factor.

    Args:
    value: The value to be quantified.
    factor: The factor to be used in quantification.

    Returns:
    float: The quantified value.
    """
    y = np.tanh(value / factor) * factor
    return y
