"""Export the current Pong state dict for dependency-free browser inference."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


LAYER_SPECS = ((7, 8, "tanh"), (8, 6, "relu"), (6, 2, "sigmoid"))


def export_model(checkpoint: Path, output: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {
        f"layers.{index}.0.{parameter}"
        for index in range(len(LAYER_SPECS))
        for parameter in ("weight", "bias")
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("Checkpoint must contain exactly the current 7 -> 8 -> 6 -> 2 state dict.")

    layers = []
    for index, (inputs, outputs, activation) in enumerate(LAYER_SPECS):
        tensors = {}
        for parameter, shape in (("weight", (outputs, inputs)), ("bias", (outputs,))):
            key = f"layers.{index}.0.{parameter}"
            tensor = state[key]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != shape
                or tensor.dtype != torch.float32
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"{key} must be a finite float32 tensor of shape {shape}.")
            tensors[parameter] = tensor.tolist()
        layers.append({"activation": activation, **tensors})

    model = {
        "version": 1,
        "architecture": [7, 8, 6, 2],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "layers": layers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("assets/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("assets/model.json"))
    args = parser.parse_args(argv)
    model = export_model(args.checkpoint, args.output)
    print(f"Exported {args.output} (checkpoint SHA-256 {model['checkpoint_sha256']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
