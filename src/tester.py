
"""
Visualize a saved (elite) player in a single-zone infinite game.
"""

import random
import torch
from ga.player import IndividualPlayer
from game import Game
from utils.logger import logger

num_layers = 3
random.seed(119)


def load_player() -> IndividualPlayer:
    """
    Load player from a single PyTorch weights file (state_dict).
    """
    player = IndividualPlayer()
    checkpoint_path = "elite_model.pt"
    try:
        player.neural_net.load_weights(checkpoint_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. "
            f"Save a model with player.neural_net.save_weights('{checkpoint_path}') first."
        )
    return player


if __name__ == "__main__":
    players = [load_player()]
    game = Game(players, timeout=-1)
    game.display_score = True
    try:
        game.start()
    except KeyboardInterrupt:
        logger.info("Game interrupted by user. Exiting...")
        exit(0)