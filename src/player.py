import numpy as np

from network import NeuralNet
from pygame import Vector2
import uuid
from functions import quantify


class IndividualPlayer:
    """
    Class for an Individual (AI) in the Population.
    An individual consists of:

    - a Neural Network (instance of NeuralNet) or the brain

    - a record of its scores

    - age

    At the end of an epoch, the individual age must be increased by one
    """

    def __init__(self):
        """
        Create an AI player.
        """
        self.uid = None
        self.scores = None
        self.age = 0
        self.reset_defaults()
        self.neural_net = NeuralNet()
        self.neural_net.add_layer(5, 8, activation='relu')
        self.neural_net.add_layer(8, 3, activation='tanh')
        self.neural_net.add_layer(3, 2, activation='binary')

    def reset_defaults(self) -> None:
        """
        Reset the default values for the individual player.
        """
        self.uid = uuid.uuid4()
        self.age = 0
        self.scores = {
            'player': 0,
            'cpu': 0,
            'moves_made': 0,
            'overall_fitness': 0,
            'hits': 0,
        }

    def think(self, inputs_list) -> np.ndarray:
        """
        Make decisions based on the inputs provided.

        Args:
        inputs_list (list): List of inputs for decision-making.

        Returns:
        list: Output decisions based on Neural Network prediction.
        """
        outputs = self.neural_net.predict(inputs_list)
        return outputs

    def look(self, zone) -> np.ndarray:
        """
        Generate inputs based on the current state of the game.

        Args:
        zone (PlayZone): The Play-zone container in which the player exists.

        Returns:
        list: List of inputs based on the speed of the ball and positions of the ball and paddles.
        """
        ball = zone.ball
        ai_paddle = zone.ai_paddle
        ball_distance = Vector2(ball.center) - Vector2(ai_paddle.center)
        inputs_list = np.array([
            ball_distance.x,
            ball.speed.magnitude(),
            ai_paddle.pos_x,
            ball.pos_x,
            ball.pos_y,
        ])
        return inputs_list

    def execute_move(self, zone) -> bool:
        """
        Perform a move based on the Neural Network predictions.

        Args:
        zone (PlayZone): The Play-zone container where the move will be executed.
        """
        player_paddle = zone.ai_paddle
        speed = zone.speed
        inputs_list = self.look(zone)
        thought = self.think(inputs_list)
        move_left = thought[0][0]
        move_right = thought[0][1]

        if move_right and move_left:
            return False

        move_distance = speed * 5.0

        if move_left and player_paddle.pos_x > 0:
            distance_left = player_paddle.pos_x
            distance_limit = 0
            if distance_left <= move_distance:
                player_paddle.pos_x = distance_limit
            else:
                player_paddle.pos_x -= move_distance
            self.scores['moves_made'] += 1
        elif move_right and player_paddle.pos_x < zone.WIDTH:
            distance_right = zone.WIDTH - player_paddle.pos_x
            distance_limit = zone.WIDTH
            if distance_right <= move_distance:
                player_paddle.pos_x = distance_limit
            else:
                player_paddle.pos_x += move_distance
            self.scores['moves_made'] += 1

    def calculate_fitness(self) -> float:
        """
        Calculate the fitness of the Individual Player.

        Apart from score, number of hits and number of moves made are also taken into consideration.

        Returns:
        float: Fitness value calculated based on the `scores` dictionary.
        """
        score = self.scores

        fitness = (1000 * score['player']
                   - 990 * score['cpu']
                   + quantify(score['hits'], 100)
                   + quantify(score['moves_made'], 10))
        self.scores['overall_fitness'] = fitness
        return fitness

    def increase_age(self) -> int:
        """
        Increase the age of the individual player by one.

        Some players may stay on top through a few generations.

        Returns:
        int: Updated age of the player.
        """
        self.age += 1
        return self.age

    @property
    def __str__(self) -> str:
        """
        Return a string representation of the individual player.

        Returns:
        str: String representation of the player with UID and overall fitness.
        """
        return 'Player = {0}; fitness = {1}'.format(self.uid, self.scores['overall_fitness'])
