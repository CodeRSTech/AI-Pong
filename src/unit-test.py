# src/unit-test.py
"""
Basic tests for core components.

Note: Running Game.start() opens an Arcade window for a timed epoch.
Consider lowering timeout or adding a headless mode for CI.
"""

import unittest
from copy import deepcopy
from src.ga.player import IndividualPlayer
from src.components import Ball
from src.game import Game
from src.ga import GeneticAlgorithm


class TestBall(unittest.TestCase):

    def test_initialization(self):
        ball = Ball(500, 500, 10, 10)
        self.assertIsInstance(ball, Ball)
        self.assertEqual(ball.speed.x, 0)
        self.assertEqual(ball.speed.y, 0)

    def test_flip_y(self):
        ball = Ball(10, 10, 5, 5)
        ball_y_speed = deepcopy(ball.speed.y) * -1
        ball.flip_y()
        self.assertEqual(ball_y_speed, ball.speed.y)


class TestGame(unittest.TestCase):

    def test_start(self):
        population = [IndividualPlayer()]
        # Keep epoch short to avoid a long window in tests
        game = Game(population, width=400, height=600, fps=60, timeout=0.5)
        result = game.start()
        self.assertEqual(population, result)
        self.assertIsInstance(result[0], IndividualPlayer)


class TestGeneticAlgorithm(unittest.TestCase):

    def test_start(self):
        pop = [IndividualPlayer() for _ in range(10)]
        ga = GeneticAlgorithm(pop)
        self.assertIsInstance(ga, GeneticAlgorithm)
        self.assertIsInstance(ga.population, list)
        self.assertEqual(ga.population, pop)
        self.assertEqual(len(ga.population), ga.population_size)
        # Keep runs small to shorten test duration (opens windows)
        ga.start(runs=1)


if __name__ == '__main__':
    unittest.main()
