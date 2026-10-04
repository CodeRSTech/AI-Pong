"""Shared CLI support for rendered and headless genetic-algorithm runs."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

from src.ga import GeneticAlgorithm
from src.ga.player import IndividualPlayer
from src.utils import logger
from src.variables import VARIABLES


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--population", type=int, default=200, help="individuals per generation (default: 200)")
    parser.add_argument("--generations", type=int, default=None,
                        help="maximum generations; run continuously if omitted")
    parser.add_argument("--seed", type=int, default=334, help="random seed (default: 334)")
    parser.add_argument("--timeout", type=float, default=VARIABLES["TIME_OUT"],
                        help="seconds per training generation (default: 12)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"),
                        help="root for timestamped run folders (default: runs)")
    parser.add_argument("--fps", type=float, default=VARIABLES["FPS"],
                        help=f"base update rate (default: {VARIABLES['FPS']})")
    parser.add_argument("--speed", type=float, default=VARIABLES["SPEED"],
                        help=f"update-rate and paddle speed multiplier (default: {VARIABLES['SPEED']})")
    parser.add_argument("--steps-per-frame", type=int, default=VARIABLES["STEPS_PER_FRAME"],
                        help=f"simulation steps per frame (default: {VARIABLES['STEPS_PER_FRAME']})")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="PyTorch intra-op CPU threads (default: 1 for small batched networks)")
    parser.add_argument("--validation-games", type=int, default=2,
                        help="fresh randomized elite-validation games per generation (default: 2)")


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.generations is not None and args.generations < 1:
        parser.error("--generations must be at least 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.speed)
        or args.fps <= 0
        or args.speed <= 0
        or not math.isfinite(args.fps * args.speed)
        or args.steps_per_frame < 1
        or not math.isfinite(args.fps * args.speed * args.steps_per_frame)
    ):
        parser.error("--fps and --speed must be positive and their product finite")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be at least 1")
    if args.validation_games < 1:
        parser.error("--validation-games must be at least 1")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch generators."""
    normalized_seed = seed % (2**32)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized_seed)


def run_training(args: argparse.Namespace, *, render: bool) -> int:
    seed_everything(args.seed)
    torch.set_num_threads(args.torch_threads)
    population = [IndividualPlayer() for _ in range(args.population)]
    ga = GeneticAlgorithm(
        population,
        output_dir=args.output_dir,
        render=render,
        seed=args.seed,
        timeout=args.timeout,
        fps=args.fps,
        speed=args.speed,
        steps_per_frame=args.steps_per_frame,
        validation_games=args.validation_games,
        sound_enabled=render and getattr(args, "sound_enabled", False),
    )
    try:
        logger.info(
            "Starting {} training; artifacts will be saved to {}",
            "rendered" if render else "headless",
            ga.run_dir,
        )
        ga.start(args.generations)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        return 130
    return 0
