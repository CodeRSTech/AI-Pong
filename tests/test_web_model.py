import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.export_web_model import LAYER_SPECS, export_model, main


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "assets/model.pt"


def test_export_is_deterministic_and_matches_committed_weights(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    model = export_model(CHECKPOINT, first)
    export_model(CHECKPOINT, second)

    assert first.read_bytes() == second.read_bytes()
    assert model == json.loads((ROOT / "assets/model.json").read_text())
    assert model["checkpoint_sha256"] == hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    for index, (_, _, activation) in enumerate(LAYER_SPECS):
        assert model["layers"][index]["activation"] == activation
        for parameter in ("weight", "bias"):
            assert torch.equal(
                torch.tensor(model["layers"][index][parameter], dtype=torch.float32),
                state[f"layers.{index}.0.{parameter}"],
            )


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nan", "inf", "not_tensor"])
def test_export_rejects_invalid_checkpoint_without_writing_output(tmp_path, corruption):
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    key = "layers.0.0.weight"
    if corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state["unexpected"] = torch.zeros(1)
    elif corruption == "shape":
        state[key] = torch.zeros(1)
    elif corruption == "dtype":
        state[key] = state[key].double()
    elif corruption == "not_tensor":
        state[key] = state[key].tolist()
    else:
        state[key][0, 0] = float(corruption)
    checkpoint = tmp_path / "invalid.pt"
    output = tmp_path / "model.json"
    torch.save(state, checkpoint)

    with pytest.raises(ValueError):
        export_model(checkpoint, output)
    assert not output.exists()


def test_cli_creates_output_directory(tmp_path):
    output = tmp_path / "nested/model.json"
    assert main(["--checkpoint", str(CHECKPOINT), "--output", str(output)]) == 0
    assert output.is_file()
