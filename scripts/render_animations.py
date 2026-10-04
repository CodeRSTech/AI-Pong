"""Render Phase 1 media without integrating it into the documentation."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality", choices=("l", "m", "h"), default="l",
        help="Manim quality: l=480p/15fps, m=720p/30fps, h=1080p/60fps",
    )
    arguments = parser.parse_args()
    output_dir = ROOT / "docs" / "images" / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai-pong-manim-") as media_dir:
        for scene, filename in (
            ("NeuralNetworkAnimation", "neural-network.mp4"),
            ("CrossoverAnimation", "two-point-crossover.mp4"),
        ):
            subprocess.run(
                [
                    sys.executable, "-m", "manim",
                    "-q" + arguments.quality, "--disable_caching",
                    "--media_dir", media_dir,
                    str(ROOT / "scripts" / "animate_diagrams.py"), scene,
                ],
                cwd=ROOT, check=True,
            )
            candidates = list(Path(media_dir).glob(f"videos/**/{scene}.mp4"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one rendered video for {scene}, found {len(candidates)}.")
            destination = output_dir / filename
            shutil.copyfile(candidates[0], destination)
            print(f"Rendered {destination.relative_to(ROOT)} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
