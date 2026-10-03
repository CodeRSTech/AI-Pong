"""
Genetic algorithm core: selection, crossover, mutation, and run artifacts.
"""

import csv
import json
import math
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from src.ga.player import IndividualPlayer
from src.utils import logger
from src.utils.functions import two_point_crossover
from src.variables import VARIABLES


class GeneticAlgorithm:
    """Manage one population and persist its training run."""

    elite_factor = 0.1
    crossover_rate = 0.4
    metric_fields = (
        "generation", "best_fitness", "mean_fitness", "median_fitness",
        "player_score", "cpu_score", "player_hits", "cpu_hits",
    )

    def __init__(
        self,
        population: list[IndividualPlayer],
        *,
        output_dir: str | Path = "runs",
        render: bool = True,
        seed: int | None = None,
        timeout: float = VARIABLES["TIME_OUT"],
        fps: float = VARIABLES["FPS"],
        speed: float = VARIABLES["SPEED"],
        steps_per_frame: int = VARIABLES["STEPS_PER_FRAME"],
    ):
        if len(population) < 2:
            raise ValueError("The genetic algorithm requires at least two individuals.")
        if (
            not math.isfinite(fps)
            or not math.isfinite(speed)
            or fps <= 0
            or speed <= 0
            or not math.isfinite(fps * speed)
            or steps_per_frame < 1
            or not math.isfinite(fps * speed * steps_per_frame)
        ):
            raise ValueError("fps, speed, and steps_per_frame must be positive.")
        if not math.isfinite(timeout) or (timeout <= 0 and timeout != -1):
            raise ValueError("timeout must be positive, or -1 for an unbounded game.")
        if not render and timeout == -1:
            raise ValueError("Headless training requires a finite positive timeout.")

        self.population = population
        self.population_size = len(population)
        self.render = render
        self.seed = seed
        self.timeout = timeout
        self.fps = fps
        self.speed = speed
        self.steps_per_frame = steps_per_frame
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f")
        self.run_dir = Path(output_dir) / timestamp
        self.generation = 0

    def start(self, runs: int = 1000) -> None:
        """Run the requested number of generations."""
        if runs < 1:
            raise ValueError("runs must be at least 1.")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        configuration = {
            "created_at": datetime.now().astimezone().isoformat(),
            "seed": self.seed,
            "population_size": self.population_size,
            "generations": runs,
            "render": self.render,
            "timeout_seconds": self.timeout,
            "fps": self.fps,
            "speed": self.speed,
            "steps_per_frame": self.steps_per_frame,
            "output_directory": str(self.run_dir),
            "elite_factor": self.elite_factor,
            "crossover_rate": self.crossover_rate,
            "mutation_scale": 0.2,
            "mutation_probability": 0.1,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
            "variables": VARIABLES,
        }
        (self.run_dir / "settings.json").write_text(
            json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Starting GA for up to {} generations.", runs)
        for generation in range(runs):
            self.generation = generation
            logger.info("Generation: {}, Population: {}", generation, self.population_size)
            self.selection()
            offsprings = self.crossover()
            self.mutate_and_append_to_population(offsprings)

    def selection(self) -> None:
        """Run one epoch, calculate fitness, and prepare the next population."""
        self.epoch()
        logger.info("Performing selection...")
        self.calculate_fitness()
        self.cull_population()
        self.repopulate()

    def epoch(self) -> None:
        """Play one generation, with or without an Arcade window."""
        for player in self.population:
            player.reset_scores()

        random.shuffle(self.population)
        from src.game import Game

        game = Game(
            self.population,
            timeout=self.timeout,
            fps=self.fps,
            speed=self.speed,
            steps_per_frame=self.steps_per_frame,
        )
        self.population = game.start() if self.render else game.run_headless(
            step_delta=1.0 / (self.fps * self.speed * self.steps_per_frame)
        )

        for player in self.population:
            player.age += 1

    def calculate_fitness(self) -> None:
        """Calculate, rank, record, and checkpoint the current generation."""
        logger.info("Calculating fitness...")
        for player in self.population:
            player.calculate_fitness()

        self.population.sort(key=lambda player: player.scores["fitness"], reverse=True)
        if not self.population:
            raise RuntimeError("Cannot calculate fitness for an empty population.")

        self._append_metrics(self.generation)
        self.save_generation_samples()
        self.print_scores()

    def _append_metrics(self, generation: int) -> None:
        best = self.population[0]
        values = {
            "generation": generation,
            "best_fitness": best.scores["fitness"],
            "mean_fitness": mean(player.scores["fitness"] for player in self.population),
            "median_fitness": median(player.scores["fitness"] for player in self.population),
            "player_score": best.scores["Player"],
            "cpu_score": best.scores["CPU"],
            "player_hits": best.scores["Player Hits"],
            "cpu_hits": best.scores["CPU Hits"],
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.run_dir / "metrics.csv"
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=self.metric_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(values)

    def cull_population(self) -> None:
        """Keep elite copies and retain fitter members for reproduction."""
        population = self.population
        if not population:
            raise RuntimeError("Cannot select survivors from an empty population.")

        fittest = population[0]
        weakest = population[-1]
        num_elites = max(1, int(len(population) * self.elite_factor))
        elites = population[:num_elites]
        logger.info("Elite players chosen: {}", num_elites)

        max_score = fittest.get_fitness()
        min_score = weakest.get_fitness()
        score_range = max_score - min_score
        survivors = deepcopy(elites) + deepcopy(elites) + deepcopy(elites)

        for player in population:
            player_score = player.get_fitness()
            favourable_factor = (
                (player_score - min_score) / score_range if score_range > 0 else 0.1
            )
            if random.random() < favourable_factor:
                survivors.append(player)

        self.population = survivors

    def repopulate(self) -> None:
        """Restore the population to its configured size."""
        logger.info("Repopulating population...")
        current_size = len(self.population)
        if current_size < self.population_size:
            self.population.extend(
                IndividualPlayer() for _ in range(self.population_size - current_size)
            )
        elif current_size > self.population_size:
            del self.population[self.population_size:]

    def crossover(self) -> list[IndividualPlayer]:
        """Create paired offspring from the top-ranked mating pool."""
        logger.info("Performing crossover...")
        num_to_mate = min(self.population_size, max(2, int(self.population_size * self.crossover_rate)))
        if num_to_mate % 2:
            num_to_mate += 1 if num_to_mate < self.population_size else -1

        mating_pool = self.population[:num_to_mate]
        random.shuffle(mating_pool)
        offsprings = []
        for index in range(0, len(mating_pool), 2):
            parent_1, parent_2 = mating_pool[index:index + 2]
            offsprings.extend((
                two_point_crossover(parent_1, parent_2),
                two_point_crossover(parent_2, parent_1),
            ))

        logger.debug("Number of offsprings: {}", len(offsprings))
        return offsprings

    def mutate_and_append_to_population(self, offsprings: list[IndividualPlayer]) -> None:
        """Reset and mutate offspring, replacing the weakest population members."""
        for offspring in offsprings:
            offspring.reset_defaults()
            offspring.neural_net.mutate(mutation_scale=0.2)
        offspring = offsprings[:self.population_size]
        survivors = self.population[:self.population_size - len(offspring)]
        self.population = survivors + offspring
        self.repopulate()

    def save_generation_samples(self, top_n: int = 2) -> None:
        """Save the generation's top checkpoints and its current elite."""
        if not self.population:
            raise RuntimeError("Cannot save checkpoints for an empty population.")
        if top_n < 0:
            raise ValueError("top_n cannot be negative.")

        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for rank, player in enumerate(self.population[:top_n]):
            player.neural_net.save_weights(checkpoint_dir / f"p{rank}gen{self.generation}.pt")
        self.population[0].neural_net.save_weights(self.run_dir / "elite_model.pt")

    def print_scores(self) -> None:
        """Print the best individual's fitness breakdown."""
        if self.population:
            print(f"P0: {self.population[0].get_scores()}\n")
