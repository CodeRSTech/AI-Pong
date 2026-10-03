import random

import numpy as np
import pytest
import torch

from src import main
from src.ga import GeneticAlgorithm
from src.ga.player import IndividualPlayer
from src.tester import find_latest_checkpoint, load_player


def test_cli_accepts_training_and_timing_options():
    args = main.create_parser().parse_args([
        "--population", "40",
        "--generations", "12",
        "--seed", "19",
        "--timeout", "3.5",
        "--output-dir", "training-runs",
        "--render",
        "--fps", "60",
        "--speed", "1.5",
        "--steps-per-frame", "4",
    ])
    assert (args.population, args.generations, args.seed) == (40, 12, 19)
    assert args.timeout == 3.5
    assert args.output_dir.name == "training-runs"
    assert args.render
    assert (args.fps, args.speed, args.steps_per_frame) == (60, 1.5, 4)


def test_cli_defaults_to_headless_and_can_disable_rendering():
    args = main.create_parser().parse_args(["--no-render"])
    assert not args.render


def test_headless_cli_rejects_unbounded_timeout():
    with pytest.raises(SystemExit):
        main.main(["--population", "2", "--timeout", "-1", "--generations", "1"])


@pytest.mark.parametrize("timeout", ["nan", "inf", "0"])
def test_cli_rejects_non_finite_or_zero_timeout(timeout):
    with pytest.raises(SystemExit):
        main.main(["--population", "2", "--timeout", timeout, "--generations", "1"])


def test_seed_everything_repeats_all_random_sources():
    main.seed_everything(72)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    first_weights = IndividualPlayer().neural_net.layers[0][0].weight.clone()

    main.seed_everything(72)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    second_weights = IndividualPlayer().neural_net.layers[0][0].weight.clone()

    assert first == second
    assert torch.equal(first_weights, second_weights)


def test_headless_epoch_uses_configured_time_step(monkeypatch):
    captured = {}

    class FakeGame:
        def __init__(self, players, **kwargs):
            captured["players"] = players
            captured["settings"] = kwargs

        def run_headless(self, step_delta):
            captured["step_delta"] = step_delta
            return captured["players"]

    monkeypatch.setattr("src.game.Game", FakeGame)
    fps, speed, steps = 40, 2, 5
    ga = GeneticAlgorithm(
        [IndividualPlayer(), IndividualPlayer()],
        render=False,
        timeout=0.1,
        fps=fps,
        speed=speed,
        steps_per_frame=steps,
    )

    ga.epoch()

    assert captured["settings"]["steps_per_frame"] == steps
    assert captured["step_delta"] == pytest.approx(1 / (fps * speed * steps))


def test_tester_finds_latest_run_checkpoint_and_loads_it(tmp_path):
    older = tmp_path / "20260101"
    newer = tmp_path / "20260102"
    older.mkdir()
    newer.mkdir()
    player = IndividualPlayer()
    player.neural_net.save_weights(older / "elite_model.pt")
    player.neural_net.save_weights(newer / "elite_model.pt")

    checkpoint = find_latest_checkpoint(tmp_path)
    loaded = load_player(checkpoint)

    assert checkpoint.parent == newer
    assert len(loaded.neural_net.layers) == 3
