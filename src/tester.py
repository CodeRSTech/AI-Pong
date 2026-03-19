import random
from player import IndividualPlayer
from game import Game
import pickle

num_layers = 3
random.seed(119)


def load_player():
    """
    Load player data from saved weights and biases files.

    Returns:
    IndividualPlayer: Player object with loaded weights and biases.
    """
    player = IndividualPlayer()
    for i in range(num_layers):
        weights_file = open("layer{0}.weights".format(i), 'rb')
        player.neural_net.layers[i].weights = pickle.load(weights_file)
        weights_file.close()
        biases_file = open("layer{0}.biases".format(i), 'rb')
        player.neural_net.layers[i].biases = pickle.load(biases_file)
        biases_file.close()
    return player


players = [load_player()]
game = Game(players, timeout=-1)
game.display_score = True
game.start()
