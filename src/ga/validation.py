"""Evaluate generation elites across deterministic, randomized Pong scenarios."""

import random
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.ga.player import IndividualPlayer
from src.utils import logger


@dataclass(frozen=True)
class ValidationScenario:
    seed: int
    duration: float
    paddle_width: int
    ball_x: float
    ball_y: float
    ball_vertical_direction: int
    ai_paddle_x: float
    cpu_paddle_x: float


class EliteValidator:
    """Run a fixed-size scenario suite and measure wins and CPU shutouts."""

    def __init__(
        self,
        *,
        seed: int,
        games: int = 20,
        duration_range: tuple[float, float] = (6.0, 18.0),
        paddle_width_range: tuple[int, int] = (50, 120),
        win_rate_target: float = 0.95,
        shutout_rate_target: float = 0.90,
        fps: float = 144,
        speed: float = 2.5,
        steps_per_frame: int = 15,
        width: int = 476,
        height: int = 500,
    ):
        if games < 1:
            raise ValueError("Validation requires at least one game.")
        if not 0 < win_rate_target <= 1 or not 0 < shutout_rate_target <= 1:
            raise ValueError("Validation rate targets must be in (0, 1].")
        if duration_range[0] <= 0 or duration_range[1] < duration_range[0]:
            raise ValueError("Validation durations must be positive and ordered.")
        if paddle_width_range[0] <= 0 or paddle_width_range[1] < paddle_width_range[0]:
            raise ValueError("Validation paddle widths must be positive and ordered.")

        self.seed = seed
        self.games = games
        self.duration_range = duration_range
        self.paddle_width_range = paddle_width_range
        self.win_rate_target = win_rate_target
        self.shutout_rate_target = shutout_rate_target
        self.fps = fps
        self.speed = speed
        self.steps_per_frame = steps_per_frame
        self.width = width
        self.height = height

    def scenarios_for(self, generation: int) -> list[ValidationScenario]:
        rng = random.Random(self.seed + generation * 1_000_003)
        scenarios = []
        for index in range(self.games):
            paddle_width = rng.randint(*self.paddle_width_range)
            half_paddle = paddle_width / 2
            scenarios.append(
                ValidationScenario(
                    seed=self.seed + generation * self.games + index + 1,
                    duration=rng.uniform(*self.duration_range),
                    paddle_width=paddle_width,
                    ball_x=rng.uniform(10, self.width - 10),
                    ball_y=rng.uniform(self.height / 2 - 100, self.height / 2 + 100),
                    ball_vertical_direction=rng.choice((-1, 1)),
                    ai_paddle_x=rng.uniform(half_paddle, self.width - half_paddle),
                    cpu_paddle_x=rng.uniform(half_paddle, self.width - half_paddle),
                )
            )
        return scenarios

    def start(self, elite: IndividualPlayer, generation: int, game=None) -> "EliteValidationRun":
        """Create a resumable suite; a supplied game is reused by the rendered runner."""
        return EliteValidationRun(self, elite, generation, game)

    def configure_scenario(self, game, scenario: ValidationScenario) -> None:
        """Apply the same randomized start state to both rendered and headless scenarios."""
        zone = game.zones[0]
        zone.ball.pos_x = scenario.ball_x
        zone.ball.pos_y = scenario.ball_y
        zone.ball.speed.y = abs(zone.ball.speed.y) * scenario.ball_vertical_direction
        zone.ai_paddle.pos_x = scenario.ai_paddle_x
        zone.cpu_paddle.pos_x = scenario.cpu_paddle_x

    def aggregate(self, generation: int, results: list[dict]) -> dict:
        """Summarize per-scenario outcomes identically for every validation mode."""
        wins = sum(result["win"] for result in results)
        shutouts = sum(result["shutout"] for result in results)
        win_rate = wins / self.games
        shutout_rate = shutouts / self.games
        return {
            "generation": generation,
            "games": self.games,
            "wins": wins,
            "losses": sum(result["player_score"] < result["cpu_score"] for result in results),
            "ties": sum(result["player_score"] == result["cpu_score"] for result in results),
            "win_rate": win_rate,
            "shutouts": shutouts,
            "shutout_rate": shutout_rate,
            "win_rate_target": self.win_rate_target,
            "shutout_rate_target": self.shutout_rate_target,
            "reached": win_rate >= self.win_rate_target and shutout_rate >= self.shutout_rate_target,
            "scenarios": results,
        }

    def evaluate(self, elite: IndividualPlayer, generation: int) -> dict:
        """Run the shared scenario suite to completion for headless callers."""
        run = self.start(elite, generation)
        step_delta = 1.0 / (self.fps * self.speed * self.steps_per_frame)
        try:
            while not run.is_complete:
                run.run_headless_scenario(step_delta)
            return run.result
        finally:
            run.close()


class EliteValidationRun:
    """Advance elite scenarios incrementally while isolating training RNG streams."""

    def __init__(self, validator: EliteValidator, elite: IndividualPlayer, generation: int, game=None):
        self.validator = validator
        self.generation = generation
        self.scenarios = validator.scenarios_for(generation)
        self.results = []
        self.scenario_index = -1
        self.player = None
        self.game = game
        self._reuse_game = game is not None
        self._closed = False
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        cuda_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        )
        # Validation seeds each scenario deterministically, then restores selection's RNG streams.
        self._rng_state = (python_state, numpy_state, torch_state, cuda_states)
        logger.info(
            "Validating generation {} across {} fresh scenarios.",
            generation,
            validator.games,
        )
        try:
            self._start_next_scenario(elite)
        except BaseException:
            self.close()
            raise

    @property
    def is_complete(self) -> bool:
        return self.scenario_index >= len(self.scenarios)

    @property
    def result(self) -> dict:
        if not self.is_complete:
            raise RuntimeError("Validation results are unavailable before the suite completes.")
        return self.validator.aggregate(self.generation, self.results)

    def _start_next_scenario(self, elite: IndividualPlayer) -> None:
        self.scenario_index += 1
        if self.is_complete:
            self.close()
            return

        scenario = self.scenarios[self.scenario_index]
        scenario_seed = scenario.seed % (2**32)
        random.seed(scenario_seed)
        np.random.seed(scenario_seed)
        torch.manual_seed(scenario_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(scenario_seed)

        self.player = deepcopy(elite)
        self.player.reset_scores()
        from src.game import Game

        if not self._reuse_game:
            self.game = Game(
                [self.player],
                width=self.validator.width,
                height=self.validator.height,
                fps=self.validator.fps,
                timeout=scenario.duration,
                speed=self.validator.speed,
                steps_per_frame=self.validator.steps_per_frame,
                paddle_width=scenario.paddle_width,
            )
        else:
            self.game.configure_epoch(
                [self.player],
                timeout=scenario.duration,
                paddle_width=scenario.paddle_width,
                phase="Elite validation",
                generation=self.generation,
                validation_scenario=self.scenario_index + 1,
                validation_count=self.validator.games,
            )
            # A single-player display lock keeps draw cadence out of the scenario RNG stream.
            self.game._display_player = self.player
            self.game._display_zone = self.game.zones[0]
        self.validator.configure_scenario(self.game, scenario)
        logger.debug(
            "Validation scenario {}/{} ({:.1f}s, {}px paddles)",
            self.scenario_index + 1,
            self.validator.games,
            scenario.duration,
            scenario.paddle_width,
        )

    def advance(self, steps: int, step_delta: float) -> None:
        """Run bounded fixed-delta steps; the rendered caller supplies the historical step size."""
        if self.is_complete:
            return
        self.game.run_steps(steps, step_delta)
        if self.game.is_finished:
            self._complete_scenario()

    def run_headless_scenario(self, step_delta: float) -> None:
        """Preserve the synchronous Game headless runner for existing evaluation callers."""
        if self.is_complete:
            return
        self.game.run_headless(step_delta)
        self._complete_scenario()

    def _complete_scenario(self) -> None:
        scenario = self.scenarios[self.scenario_index]
        scores = self.player.scores
        self.results.append({
            **asdict(scenario),
            "player_score": scores["Player"],
            "cpu_score": scores["CPU"],
            "win": scores["Player"] > scores["CPU"],
            "shutout": scores["CPU"] == 0,
        })
        logger.info(
            "Validation progress: {}/{} scenarios complete.",
            len(self.results),
            self.validator.games,
        )
        if len(self.results) == self.validator.games:
            self.scenario_index = len(self.scenarios)
            self.close()
        else:
            self._start_next_scenario(self.player)

    def close(self) -> None:
        """Restore every training RNG even if a rendered window is closed mid-suite."""
        if self._closed:
            return
        python_state, numpy_state, torch_state, cuda_states = self._rng_state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        self._closed = True
