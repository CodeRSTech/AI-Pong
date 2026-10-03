"""Rendered entry point for the AI-Pong genetic algorithm."""

import argparse

from src.training_cli import add_training_arguments, run_training, validate_arguments


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AI-Pong with the rendered Arcade game.")
    add_training_arguments(parser)
    parser.set_defaults(sound_enabled=True)
    parser.add_argument(
        "--no-sound",
        dest="sound_enabled",
        action="store_false",
        help="disable generated gameplay sound effects",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    return run_training(args, render=True)


if __name__ == "__main__":
    raise SystemExit(main())
