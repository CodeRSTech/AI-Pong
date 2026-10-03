"""Headless entry point for long-running AI-Pong training."""

import argparse

from src.training_cli import add_training_arguments, run_training, validate_arguments


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AI-Pong without opening a game window.")
    add_training_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    return run_training(args, render=False)


if __name__ == "__main__":
    raise SystemExit(main())
