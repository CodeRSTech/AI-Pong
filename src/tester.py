
"""
Visualize a saved (elite) player in a single-zone infinite game.
"""

import random

from src.ga import IndividualPlayer
from game import Game
from src.utils import logger

random.seed(38345343)


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