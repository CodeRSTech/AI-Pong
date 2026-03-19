from functions import binary, relu
from numpy import tanh
from pygame.color import Color

VARIABLES = {
    'WIDTH': 400,
    'HEIGHT': 500,
    'FPS': 288,
    'TIME_OUT': 4,
    'SPEED': 2
}

activation_functions = {
    'binary': binary,
    'tanh': tanh,
    'relu': relu,
}

BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
RED = Color(200, 0, 0)
GREEN = Color(0, 200, 0)
BLUE = Color(0, 0, 200)
YELLOW = Color(255, 255, 0)
GRAY = Color(100, 100, 100)
LIGHT_GRAY = Color(150, 150, 150)
OFF_WHITE = Color(220, 220, 180)
