# src/player.py
"""
AI Player driven by a tiny neural network.
"""
import math
import uuid
import numpy as np
from components.geometry import Vec2
from ga.network import NeuralNet
from utils.logger import logger
from utils.functions import quantify, timeit


class IndividualPlayer:
    """
    Class for an Individual (AI) in the Population.
    """

    def __init__(self):
        self.fit_scores = None
        self.uid = None
        self.scores = None
        self.age = 0
        self.streak = 0
        self.reset_defaults()
        self.neural_net = NeuralNet()
        # Keep the original architecture
        self.neural_net.add_layer(7, 10, activation='tanh')
        self.neural_net.add_layer(10, 6, activation='tanh')
        self.neural_net.add_layer(6, 2, activation='binary')

    
    
    def think(self, inputs_list) -> np.ndarray:
        """
        Feed-forward decision using the neural network.
        """
        outputs = self.neural_net.predict(inputs_list)
        return outputs

    def reset_scores(self) -> None:
        """
        Specifically reset game-play stats without wiping UID or Age.
        """
        self.scores = {
            'Player': 0,
            'CPU': 0,
            'left_moves': 0,
            'right_moves': 0,
            'hits': 0,
            'streak': 0,
            'fitness': 0,
        }

    def reset_defaults(self) -> None:
        """
        Reset stats and assign new UID for brand new offspring/pupils.
        """
        self.uid = uuid.uuid4()
        self.age = 0
        self.reset_scores()  # Call the new helper here

    @staticmethod
    def look(zone) -> np.ndarray:
        ball = zone.ball
        ai_paddle = zone.ai_paddle

        ball_distance_x = ball.pos_x - ai_paddle.pos_x
        ball_distance_y = ball.pos_y - ai_paddle.pos_y

        # Normalize inputs relative to zone dimensions
        inputs_list = np.array([
            ball_distance_x / zone.WIDTH,
            ball_distance_y / zone.HEIGHT,
            (ai_paddle.pos_x * 2.0 - zone.WIDTH) / zone.WIDTH,  # Paddle position normalized
            (ball.pos_x * 2.0 - zone.WIDTH) / zone.WIDTH,  # Ball X normalized
            (ball.pos_y * 2.0 - zone.HEIGHT) / zone.HEIGHT,  # Ball Y normalized
            ball.speed.x / ball.speed.magnitude(),
            ball.speed.y / ball.speed.magnitude()
        ])
        return inputs_list

    def execute_move(self, zone) -> bool:
        """
        Perform a move based on the Neural Network predictions.
        """
        player_paddle = zone.ai_paddle
        speed = zone.speed
        inputs_list = self.look(zone)
        thought = self.think(inputs_list)

        # Read the thresholded binary outputs directly
        move_left = bool(thought[0][0])
        move_right = bool(thought[0][1])

        # Conflicting move: ignore motion, keep counters unchanged
        if move_left and move_right:
            return False

        move_distance = max(2.0, float(speed))

        half = player_paddle.width / 2.0
        min_x = half
        max_x = zone.WIDTH - half

        intended_dir = 0
        if move_left:
            self.scores['left_moves'] += 1
            intended_dir = -1
        elif move_right:
            self.scores['right_moves'] += 1
            intended_dir = 1

        # Apply movement (clamped).
        if intended_dir == -1:
            player_paddle.pos_x = max(min_x, player_paddle.pos_x - move_distance)
        elif intended_dir == 1:
            player_paddle.pos_x = min(max_x, player_paddle.pos_x + move_distance)

        return True

    def calculate_fitness(self) -> float:
        score = self.scores

        left_moves = score['left_moves']
        right_moves = score['right_moves']
        total_moves = left_moves + right_moves

        fit_scores = {
            'Player': (1000 if (left_moves >0 ) and (right_moves > 0) else 0) * score['Player'],
            'CPU': -990 * score['CPU'],
            'Hits': quantify(score['hits'], 100),
            'Left-Moves': quantify(left_moves, 10),
            'Right-Moves': quantify(right_moves, 10),
        }

        fitness = sum(fit_scores.values())

        fit_scores['Fitness'] = int(fitness)

        self.scores['fitness'] = int(fitness)
        self.fit_scores = fit_scores
        return fitness

    def increase_age(self) -> int:
        """Increase age by one."""
        self.age += 1
        return self.age

    def add_streak(self) -> None:
        """Update the streak counter."""
        self.streak += 1
        self.scores['streak'] = max(self.scores['streak'], self.streak)

    def reset_streak(self) -> None:
        """Reset the streak counter."""
        self.streak = 0

    @property
    def __str__(self) -> str:
        """Readable identity."""
        return 'Player = {0}; fitness = {1}'.format(self.uid, self.scores['fitness'])

    def __repr__(self):
        return 'Player = {0}; fitness = {1}'.format(self.uid, self.scores['fitness'])
