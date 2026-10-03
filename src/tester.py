"""
Visualize the elite from a saved run or a specified PyTorch checkpoint.
"""

import argparse
from pathlib import Path

from src.ga.player import IndividualPlayer
from src.game import Game
from src.utils import logger


def find_latest_checkpoint(runs_dir: Path = Path("runs")) -> Path:
    """Find the newest run's elite model, retaining support for legacy output."""
    candidates = list(runs_dir.glob("*/elite_model.pt"))
    if candidates:
        return max(
            candidates,
            key=lambda path: (path.parent.name, path.stat().st_mtime_ns),
        )

    legacy_checkpoint = Path("elite_model.pt")
    if legacy_checkpoint.is_file():
        return legacy_checkpoint
    raise FileNotFoundError(
        f"No elite checkpoint found under '{runs_dir}' or in the current directory. "
        "Train a model with 'python -m src.main' first, or provide --checkpoint."
    )


def load_player(checkpoint_path: Path | None = None, runs_dir: Path = Path("runs")) -> IndividualPlayer:
    """Load an elite player from an explicit checkpoint or the latest run."""
    checkpoint = checkpoint_path if checkpoint_path is not None else find_latest_checkpoint(runs_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' does not exist.")

    player = IndividualPlayer()
    player.neural_net.load_weights(str(checkpoint))
    return player


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch an AI-Pong elite checkpoint.")
    parser.add_argument("--checkpoint", type=Path, help="path to a model weights file")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="training runs root to search (default: runs)")
    parser.add_argument("--no-sound", action="store_true",
                        help="disable generated gameplay sound effects")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    players = [load_player(args.checkpoint, args.runs_dir)]
    game = Game(players, timeout=-1, sound_enabled=not args.no_sound)
    game.display_score = True
    try:
        game.start()
    except KeyboardInterrupt:
        logger.info("Game interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
