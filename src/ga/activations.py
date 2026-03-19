# src/activations.py
"""
Activation functions for the neural network.

Separated to avoid circular imports between variables/layer/functions.
"""

import numpy as np


def binary(z_array) -> np.ndarray:
    """
    Binary step function (thresholded at 0.8).
    Returns a boolean array of the same shape.
    """
    threshold = 0.8
    return z_array >= threshold


def relu(z_array) -> np.ndarray:
    """
    ReLU activation for a 2D array (expects shape (1, N)).
    """
    result = []
    for zi in z_array[0]:
        result.append(max(zi, 0.0))
    return np.array([result])
