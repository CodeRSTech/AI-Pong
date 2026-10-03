import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import main, train
from src.ga import GeneticAlgorithm
from src.ga.player import IndividualPlayer
from src.game import Game
from src.tester import find_latest_checkpoint, load_player
from src.training_cli import run_training, seed_everything


def test_headless_cli_accepts_training_and_timing_options():
    args = train.create_parser().parse_args([
        "--population", "40",
        "--generations", "12",
        "--seed", "19",
        "--timeout", "3.5",
        "--output-dir", "training-runs",
        "--fps", "60",
        "--speed", "1.5",
        "--steps-per-frame", "4",
        "--torch-threads", "2",
    ])
    assert (args.population, args.generations, args.seed) == (40, 12, 19)
    assert args.timeout == 3.5
    assert args.output_dir.name == "training-runs"
    assert (args.fps, args.speed, args.steps_per_frame) == (60, 1.5, 4)
    assert args.torch_threads == 2


def test_rendered_runner_is_the_main_default():
    assert main.create_parser().parse_args([]).generations is None
    assert train.create_parser().parse_args([]).generations is None
    assert main.create_parser().parse_args([]).sound_enabled
    assert not main.create_parser().parse_args(["--no-sound"]).sound_enabled
    assert not hasattr(train.create_parser().parse_args([]), "sound_enabled")


@pytest.mark.parametrize(("module", "render_expected"), [(main, True), (train, False)])
def test_entry_points_select_their_expected_render_mode(monkeypatch, module, render_expected):
    rendered = []

    def capture_render_mode(args, render):
        rendered.append(render)
        return 0

    monkeypatch.setattr(
        module,
        "run_training",
        capture_render_mode,
    )
    assert module.main(["--population", "2", "--generations", "1"]) == 0
    assert rendered == [render_expected]


@pytest.mark.parametrize(
    ("render", "sound_option", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_training_only_enables_sound_for_unmuted_rendered_runs(
    monkeypatch, render, sound_option, expected
):
    captured = {}

    class FakeGA:
        def __init__(self, population, **kwargs):
            captured.update(kwargs)
            self.run_dir = "test-runs"

        def start(self, generations):
            pass

    monkeypatch.setattr("src.training_cli.seed_everything", lambda seed: None)
    monkeypatch.setattr("src.training_cli.torch.set_num_threads", lambda threads: None)
    monkeypatch.setattr("src.training_cli.IndividualPlayer", object)
    monkeypatch.setattr("src.training_cli.GeneticAlgorithm", FakeGA)
    args = main.create_parser().parse_args(
        ["--population", "2"] + ([] if sound_option else ["--no-sound"])
    )

    assert run_training(args, render=render) == 0
    assert captured["sound_enabled"] is expected


def test_headless_cli_rejects_unbounded_timeout():
    with pytest.raises(SystemExit):
        train.main(["--population", "2", "--timeout", "-1", "--generations", "1"])


@pytest.mark.parametrize("timeout", ["nan", "inf", "0"])
def test_cli_rejects_non_finite_or_zero_timeout(timeout):
    with pytest.raises(SystemExit):
        train.main(["--population", "2", "--timeout", timeout, "--generations", "1"])


def test_seed_everything_repeats_all_random_sources():
    seed_everything(72)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    first_weights = IndividualPlayer().neural_net.layers[0][0].weight.clone()

    seed_everything(72)
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

        def run_headless(self, step_delta, progress_callback=None):
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
    assert captured["settings"]["validation_status"] is None
    assert captured["settings"]["sound_enabled"] is False


def test_validator_generates_fresh_scenarios_in_configured_ranges(monkeypatch):
    from src.ga.validation import EliteValidator

    applied_scenarios = []

    class FakeGame:
        win_count = 20
        shutout_count = 20

        def __init__(self, players, **kwargs):
            self.player = players[0]
            self.zones = [SimpleNamespace(
                ball=SimpleNamespace(
                    pos_x=0,
                    pos_y=0,
                    speed=SimpleNamespace(y=2),
                ),
                ai_paddle=SimpleNamespace(pos_x=0),
                cpu_paddle=SimpleNamespace(pos_x=0),
            )]
            self.timeout = kwargs["timeout"]
            self.paddle_width = kwargs["paddle_width"]

        def run_headless(self, step_delta):
            zone = self.zones[0]
            applied_scenarios.append((
                self.timeout,
                self.paddle_width,
                zone.ball.pos_x,
                zone.ball.pos_y,
                1 if zone.ball.speed.y > 0 else -1,
                zone.ai_paddle.pos_x,
                zone.cpu_paddle.pos_x,
            ))
            scenario_index = len(applied_scenarios) - 1
            self.player.scores["Player"] = int(scenario_index < self.win_count)
            self.player.scores["CPU"] = int(scenario_index >= self.shutout_count)

    monkeypatch.setattr("src.game.Game", FakeGame)
    validator = EliteValidator(seed=23)
    scenarios = validator.scenarios_for(0)
    elite = IndividualPlayer()

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    result = validator.evaluate(elite, 0)

    assert len(scenarios) == 20
    assert {scenario.ball_vertical_direction for scenario in scenarios} == {-1, 1}
    assert all(6 <= scenario.duration <= 18 for scenario in scenarios)
    assert all(50 <= scenario.paddle_width <= 120 for scenario in scenarios)
    assert scenarios != validator.scenarios_for(1)
    assert result["win_rate"] == 1
    assert result["shutout_rate"] == 1
    assert result["reached"]
    assert len(applied_scenarios) == 20
    assert all(6 <= applied[0] <= 18 and 50 <= applied[1] <= 120 for applied in applied_scenarios)
    assert random.getstate() == python_state
    assert all(np.array_equal(actual, expected) if isinstance(actual, np.ndarray) else actual == expected
               for actual, expected in zip(np.random.get_state(), numpy_state))
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    applied_scenarios.clear()
    FakeGame.win_count = 18
    below_target = validator.evaluate(elite, 2)
    assert below_target["win_rate"] == 0.9
    assert below_target["shutout_rate"] == 1
    assert not below_target["reached"]


def test_validation_success_latches_and_does_not_stop_training(tmp_path, monkeypatch):
    population = [IndividualPlayer() for _ in range(4)]
    ga = GeneticAlgorithm(population, output_dir=tmp_path, render=False, timeout=1, validation_games=1)
    ga.epoch = lambda: None

    def successful_validation(elite, generation):
        return {
            "generation": generation,
            "games": 20,
            "wins": 19,
            "losses": 1,
            "ties": 0,
            "win_rate": 0.95,
            "shutouts": 18,
            "shutout_rate": 0.9,
            "win_rate_target": 0.95,
            "shutout_rate_target": 0.9,
            "reached": True,
            "scenarios": [],
        }

    monkeypatch.setattr(ga.validator, "evaluate", successful_validation)
    ga.start(runs=2)

    assert ga.generation == 1
    assert ga.validation_reached
    assert ga.validation_status["reached"]


def test_headless_progress_callback_reports_completion():
    game = Game([IndividualPlayer()], width=400, height=500, timeout=0.03)
    progress = []
    step_delta = 0.01

    game.run_headless(step_delta, lambda steps, total, elapsed: progress.append((steps, total)))

    assert progress
    assert progress[-1][0] == 3
    assert progress[-1][1] == 3
    assert game.time_running >= game.timeout


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
