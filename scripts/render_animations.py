"""Render Phase 1 media without integrating it into the documentation."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.animation_theme import font_notice
from scripts.gameplay_trace import load_trace


ROOT = Path(__file__).resolve().parents[1]

SCENES = {
    "neural-network": ("NeuralNetworkAnimation", "animate_diagrams.py", "neural-network.mp4"),
    "two-point-crossover": ("CrossoverAnimation", "animate_diagrams.py", "two-point-crossover.mp4"),
    "observations": ("GameplayObservationsScene", "gameplay_scenes.py", "gameplay-observations.mp4"),
    "network-actions": ("ObservationNetworkActionsScene", "gameplay_scenes.py", "network-actions.mp4"),
    "actions-execution": ("ActionsExecutionScene", "gameplay_scenes.py", "actions-execution.mp4"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality", choices=("l", "m", "h"), default="l",
        help="Manim quality: l=480p/15fps, m=720p/30fps, h=1080p/60fps",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENES),
        default=("neural-network", "two-point-crossover"),
        help="one or more scene modules to render (gameplay scenes require --trace)",
    )
    parser.add_argument("--trace", type=Path, help="validated gameplay trace JSON")
    parser.add_argument("--decision-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "images" / "animations",
        help="destination for videos (use a session directory for previews)",
    )
    arguments = parser.parse_args()
    gameplay_scenes = {
        "observations", "network-actions", "actions-execution"
    }
    if gameplay_scenes.intersection(arguments.scenes):
        if arguments.trace is None:
            parser.error("gameplay scenes require an explicit --trace.")
        trace = load_trace(arguments.trace)
        if not 0 <= arguments.decision_index < len(trace["decisions"]):
            parser.error(
                f"--decision-index must be in [0, {len(trace['decisions']) - 1}]."
            )
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(font_notice())
    with tempfile.TemporaryDirectory(prefix="ai-pong-manim-") as media_dir:
        for scene_key in arguments.scenes:
            scene, module, filename = SCENES[scene_key]
            env = os.environ.copy()
            if arguments.trace is not None:
                env["AI_PONG_TRACE_PATH"] = str(arguments.trace.resolve())
                env["AI_PONG_DECISION_INDEX"] = str(arguments.decision_index)
            subprocess.run(
                [
                    sys.executable, "-m", "manim",
                    "-q" + arguments.quality, "--disable_caching",
                    "--media_dir", media_dir,
                    str(ROOT / "scripts" / module), scene,
                ],
                cwd=ROOT, check=True, env=env,
            )
            candidates = list(Path(media_dir).glob(f"videos/**/{scene}.mp4"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one rendered video for {scene}, found {len(candidates)}.")
            destination = output_dir / filename
            shutil.copyfile(candidates[0], destination)
            print(f"Rendered {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
