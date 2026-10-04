"""Capture a deterministic trace from an explicitly selected checkpoint."""

import argparse
import json
from pathlib import Path

from scripts.gameplay_trace import capture_trace, write_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="scenario RNG seed")
    parser.add_argument("--steps", type=int, required=True, help="1-1000 simulation decisions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--generation", default="unknown")
    parser.add_argument(
        "--validation-json",
        type=Path,
        help="optional JSON file with validation provenance/results",
    )
    parser.add_argument(
        "--step-delta",
        type=float,
        help="positive simulation seconds per step (defaults to rendered-game timing)",
    )
    args = parser.parse_args(argv)

    validation = "unknown"
    if args.validation_json is not None:
        validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
        if not isinstance(validation, dict):
            parser.error("--validation-json must contain a JSON object.")
    trace = capture_trace(
        args.checkpoint,
        seed=args.seed,
        steps=args.steps,
        run_id=args.run_id,
        generation=args.generation,
        validation=validation,
        step_delta=args.step_delta,
    )
    destination = write_trace(trace, args.output)
    print(
        f"Wrote {len(trace['decisions'])} decisions to {destination} "
        f"(checkpoint SHA-256 {trace['metadata']['checkpoint']['sha256']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
