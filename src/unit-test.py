import unittest
import numpy as np
from copy import deepcopy
from player import IndividualPlayer
from ball import Ball
from game import Game
from ga_core import GeneticAlgorithm
from layer import Layer


class TestBall(unittest.TestCase):

    def test_initialization(self):
        ball = Ball(500,500, 10,10)
        self.assertIsInstance(ball, Ball)
        self.assertEqual(ball.speed.x, 0)
        self.assertEqual(ball.speed.y, 0)
        pass

    def test_flip_y(self):
        ball = Ball(10, 10, 5, 5)
        ball_y_speed = deepcopy(ball.speed.y) * -1
        ball.flip_y()
        self.assertEqual(ball_y_speed, ball.speed.y)


class TestGame(unittest.TestCase):

    def test_start(self):
        population = [IndividualPlayer()]
        game = Game(population, width=400, height=600, fps=60, timeout=3)
        result = game.start()
        self.assertEqual(population, result)
        self.assertIsInstance(result[0], IndividualPlayer)
        pass


class TestLayer(unittest.TestCase):

    def test_initialization(self):
        layer1 = Layer(4,9, 'sigmoid')
        layer2 = Layer(4, 9, 'relu')
        layer3 = Layer(4,9, 'tanh')

        for layer in [layer1, layer2, layer3]:
            self.assertIsInstance(layer, Layer)
        pass

    def test_feed_forward(self):
        layer = Layer(3,5, 'tanh')
        weights = layer.weights
        biases = layer.biases

        activations = np.random.randn(1, 3)

        z = np.dot(activations, weights) + biases
        y = np.tanh(z)

        result = layer.feed_forward(activations)

        difference = result - y

        for d in difference[0]:
            self.assertEqual(d, 0.0, msg='feed forward method is not working properly')

    def test_mutate_layer(self):
        layer = Layer(3, 5, 'tanh')
        weights = deepcopy(layer.weights)
        biases = deepcopy(layer.biases)

        layer.mutate_layer()

        weights_difference = weights - layer.weights
        biases_difference = biases - layer.biases

        sum_w = np.sum(weights_difference)
        sum_b = np.sum(biases_difference)

        self.assertNotEqual(sum_w, 0.0, msg='Weights are not modified in layer mutation')
        self.assertNotEqual(sum_b, 0.0, msg='Biases are not modified in layer mutation')

    def test_string_representation(self):
        layer = Layer(5,7, 'relu')
        name = str(layer)
        self.assertEqual(name, 'Layer(5, 7, activation=relu)')


class TestGeneticAlgorithm(unittest.TestCase):

    def test_start(self):
        pop = [IndividualPlayer() for _ in range(10)]
        ga = GeneticAlgorithm(pop)
        self.assertIsInstance(ga, GeneticAlgorithm)
        self.assertIsInstance(ga.population, list)
        self.assertEqual(ga.population, pop)
        self.assertEqual(len(ga.population), ga.population_size)
        ga.start(runs=3)
        pass


if __name__ == '__main__':
    unittest.main()
