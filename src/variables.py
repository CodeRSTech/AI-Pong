# src/variables.py
"""
Global configuration and activation function lookup.
"""

from numpy import tanh
from ga.activations import binary, relu


VARIABLES = {
    'WIDTH': 476,
    'HEIGHT': 600,
    'FPS': 20,  # Logical updates per second (used to set Arcade update rate)
    'TIME_OUT': 60,  # Seconds per generation; -1 for infinite
    'SPEED': 6,  # Multiplier for the update rate
    'PANEL_WIDTH': 640
}

activation_functions = {
    'binary': binary,
    'tanh': tanh,
    'relu': relu,
}
