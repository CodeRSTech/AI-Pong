import json
import random

import numpy as np
import pytest
import torch

from scripts.gameplay_trace import (
    MAX_TRACE_STEPS,
    capture_trace,
    load_trace,
    validate_trace,
    write_trace,
)
from src.ga.player import IndividualPlayer


def _checkpoint(path):
    torch.manual_seed(701)
    IndividualPlayer().neural_net.save_weights(str(path))
    return path


def _rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_snapshot_equal(before, after):
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_capture_is_reproducible_and_preserves_process_rngs(tmp_path):
    checkpoint = _checkpoint(tmp_path / "fixed.pt")
    before = _rng_snapshot()

    first = capture_trace(checkpoint, seed=42, steps=6)
    second = capture_trace(checkpoint, seed=42, steps=6)

    _assert_rng_snapshot_equal(before, _rng_snapshot())
    assert first == second
    assert first["metadata"]["scenario_seed"] == 42
    assert first["metadata"]["checkpoint"]["sha256"]
    assert first["metadata"]["provenance"] == {
        "run_id": "unknown",
        "generation": "unknown",
        "validation": "unknown",
    }


def test_trace_records_live_activations_actions_and_game_step_states(tmp_path):
    trace = capture_trace(_checkpoint(tmp_path / "fixed.pt"), seed=9, steps=8)

    assert trace["model"]["layer_sizes"] == [7, 8, 6, 2]
    assert len(trace["model"]["parameters"]) == 3
    for index, decision in enumerate(trace["decisions"]):
        assert decision["index"] == index
        np.testing.assert_allclose(
            decision["activations"][0], decision["observations"], atol=1e-6
        )
        assert decision["actions"] == (
            np.asarray(decision["activations"][-1]) > 0.5
        ).tolist()
        assert decision["events"]["ai_paddle_displacement"] == pytest.approx(
            decision["post_state"]["ai_paddle"]["x"]
            - decision["pre_state"]["ai_paddle"]["x"]
        )
        assert decision["events"]["cpu_paddle_displacement"] == pytest.approx(
            decision["post_state"]["cpu_paddle"]["x"]
            - decision["pre_state"]["cpu_paddle"]["x"]
        )
        half = decision["post_state"]["ai_paddle"]["width"] / 2
        assert half <= decision["post_state"]["ai_paddle"]["x"] <= (
            trace["metadata"]["court"]["width"] - half
        )


def test_trace_file_round_trip_and_corruption_are_checked(tmp_path):
    trace = capture_trace(_checkpoint(tmp_path / "fixed.pt"), seed=3, steps=2)
    path = write_trace(trace, tmp_path / "trace.json")
    assert load_trace(path) == trace

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["model"]["parameters"][0]["weights"][0].pop()
    with pytest.raises(ValueError, match="malformed|dimensions"):
        validate_trace(malformed)

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["decisions"][0]["observations"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_trace(malformed)

    malformed = json.loads(path.read_text(encoding="utf-8"))
    malformed["decisions"][0]["activations"][1][0] += 0.1
    with pytest.raises(ValueError, match="disagree with checkpoint"):
        validate_trace(malformed)

    malformed = json.loads(path.read_text(encoding="utf-8"))
    del malformed["decisions"][0]["events"]["ai_paddle_displacement"]
    with pytest.raises(ValueError, match="events are incomplete"):
        validate_trace(malformed)

    malformed = json.loads(path.read_text(encoding="utf-8"))
    del malformed["decisions"][0]["post_state"]["ball"]["vx"]
    with pytest.raises(ValueError, match="ball is incomplete"):
        validate_trace(malformed)


@pytest.mark.parametrize(
    "seed,steps",
    [(-1, 1), (2**32, 1), (1.5, 1), (1, 0), (1, MAX_TRACE_STEPS + 1)],
)
def test_capture_rejects_invalid_seed_or_step_count(tmp_path, seed, steps):
    checkpoint = _checkpoint(tmp_path / "fixed.pt")
    with pytest.raises(ValueError):
        capture_trace(checkpoint, seed=seed, steps=steps)


def test_capture_requires_an_explicit_existing_compatible_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        capture_trace(tmp_path / "missing.pt", seed=0, steps=1)

    incompatible = tmp_path / "incompatible.pt"
    torch.save({"unexpected": torch.tensor([1.0])}, incompatible)
    with pytest.raises(RuntimeError):
        capture_trace(incompatible, seed=0, steps=1)
