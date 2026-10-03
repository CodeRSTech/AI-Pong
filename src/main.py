"""
Command-line entry point for training the genetic algorithm.
"""

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


def seed_everything(seed: int) -> None:
    """Seed every random-number generator used by training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AI-Pong with a genetic algorithm.")
    parser.add_argument("--population", type=int, default=200, help="number of individuals (default: 200)")
    parser.add_argument("--generations", type=int, default=1000, help="generation limit (default: 1000)")
    parser.add_argument("--seed", type=int, default=334, help="random seed (default: 334)")
    parser.add_argument("--timeout", type=float, default=VARIABLES["TIME_OUT"],
                        help="seconds per generation; headless runs require a finite positive value")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"),
                        help="root directory for timestamped run folders (default: runs)")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=False,
                        help="show the Arcade window (default: headless)")
    parser.add_argument("--fps", type=float, default=VARIABLES["FPS"],
                        help=f"base update rate (default: {VARIABLES['FPS']})")
    parser.add_argument("--speed", type=float, default=VARIABLES["SPEED"],
                        help=f"update-rate and paddle speed multiplier (default: {VARIABLES['SPEED']})")
    parser.add_argument("--steps-per-frame", type=int, default=VARIABLES["STEPS_PER_FRAME"],
                        help=f"simulation steps per Arcade update (default: {VARIABLES['STEPS_PER_FRAME']})")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.generations < 1:
        parser.error("--generations must be at least 1")
    if not math.isfinite(args.timeout) or (args.timeout <= 0 and args.timeout != -1):
        parser.error("--timeout must be positive, or -1 for an unbounded rendered game")
    if not args.render and args.timeout == -1:
        parser.error("headless training requires a finite positive --timeout")
    if (
        not math.isfinite(args.fps)
        or not math.isfinite(args.speed)
        or args.fps <= 0
        or args.speed <= 0
        or not math.isfinite(args.fps * args.speed)
        or not math.isfinite(args.fps * args.speed * args.steps_per_frame)
    ):
        parser.error("--fps and --speed must be positive")
    if args.steps_per_frame < 1:
        parser.error("--steps-per-frame must be at least 1")

    seed_everything(args.seed)
    population = [IndividualPlayer() for _ in range(args.population)]
    ga = GeneticAlgorithm(
        population,
        output_dir=args.output_dir,
        render=args.render,
        seed=args.seed,
        timeout=args.timeout,
        fps=args.fps,
        speed=args.speed,
        steps_per_frame=args.steps_per_frame,
    )
    try:
        logger.info("Starting training; run artifacts will be saved to {}", ga.run_dir)
        ga.start(args.generations)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
