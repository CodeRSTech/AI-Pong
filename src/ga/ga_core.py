# src/ga_core.py
"""
Genetic Algorithm core: selection, crossover, mutation, and population management.
"""

import pickle
import random

from game import Game
from ga.player import IndividualPlayer
from utils.logger import logger
from utils.functions import two_point_crossover


class GeneticAlgorithm:
    """
    A class representing core functionality of the Genetic Algorithm.
    """
    elite_factor = 0.1
    crossover_rate = 0.4

    def __init__(self, population: list[IndividualPlayer]):
        """
        Initialize a Genetic Algorithm with Initial Population `population`.
        """
        self.population = population
        self.population_size = len(population)

    def start(self, runs=1000) -> None:
        """
        Run the Genetic Algorithm for a given number of generations.
        """
        logger.info("Starting GA...")
        for i in range(runs):
            logger.info("Generation: {0}, Population: {1}".format(i, self.population_size))
            self.selection(i)  # <-- Pass the generation index 'i' here
            offsprings = self.crossover()
            self.mutate_and_append_to_population(offsprings)

    def selection(self, gen_idx: int) -> None:
        """
        Run one epoch/game, evaluate, and prepare the next generation.
        """
        self.epoch()
        logger.info("Performing selection...")
        try:
            self.calculate_fitness(gen_idx)  # <-- Pass it here
            self.cull_population()
            self.repopulate()
        except LookupError as e:
            logger.error("LookupError: {0}".format(e))
            exit(0)

    def epoch(self) -> None:
        """
        Play the game for one generation.
        """
        # Reset scores for ALL players (survivors + offspring + new pupils)
        for player in self.population:
            player.reset_scores()

        random.shuffle(self.population)
        game = Game(self.population)
        self.population = game.start()

        for player in self.population:
            player.age += 1

    def calculate_fitness(self, gen_idx: int) -> None:
        """
        Calculate fitness for every player and persist the elite.
        """
        logger.info("Calculating fitness...")
        for player in self.population:
            player.calculate_fitness()

        self.population = sorted(self.population,
                                 key=lambda player_i: player_i.scores['fitness'],
                                 reverse=True)

        self.save_generation_samples(gen_idx)  # <-- Call the new save method
        self.print_scores()

    def cull_population(self) -> None:
        fittest = self.population[0]
        num_elites = int(len(self.population) * self.elite_factor)
        elites = self.population[:num_elites]
        logger.info("Elite players chosen: {0}".format(num_elites))

        max_score = fittest.scores['fitness']
        min_score = self.population[-1].scores['fitness']

        score_range = max_score - min_score

        survivors = elites
        for player in self.population:
            player_score = player.scores['fitness']

            # Shift scores safely
            if score_range > 0:
                favourable_factor = (player_score - min_score) / score_range
            else:
                favourable_factor = 0.1  # 10% chance to survive if everyone is tied

            if random.random() < favourable_factor:
                survivors.append(player)

        self.population = survivors

    def repopulate(self) -> None:
        """
        Adjust population back to the configured size.
        Adds if short; trims if over (removes worst at the tail).
        """
        logger.info("Repopulating population...")
        target = self.population_size
        current = len(self.population)
        logger.debug("Target population size is {0}, current size is {1}".format(target, current))

        if current < target:
            logger.info("Adding new pupils...")
            need = target - current
            for _ in range(need):
                new_pupil = IndividualPlayer()
                self.population.append(new_pupil)
        elif current > target:
            logger.info("Removing the worst...")
            # Trim excess (assumes population is already sorted by fitness descending)
            del self.population[target:]

    def crossover(self) -> list:
        """
        Perform crossover and return new offsprings.
        """
        logger.info("Performing crossover...")
        offsprings = []
        num_to_mate = int(self.population_size * self.crossover_rate)

        if num_to_mate % 2 != 0:
            num_to_mate += 1
            logger.warning("Crossover rate is not even. Adding one more individual to mate.")
        mating_pool = self.population[:num_to_mate]

        logger.debug("Mating pool size is {0}, maintaining diversity...".format(num_to_mate))
        random.shuffle(mating_pool)

        for i in range(0, len(mating_pool), 2):
            parent_1, parent_2 = mating_pool[i], mating_pool[i+1]
            offspring1 = two_point_crossover(parent_1, parent_2)
            offspring2 = two_point_crossover(parent_2, parent_1)
            offsprings.extend([offspring1, offspring2])

        logger.debug("Number of  Offsprings {0}".format(len(offsprings)))
        return offsprings

    def mutate_and_append_to_population(self, offsprings: list[IndividualPlayer]) -> None:
        """
        Mutate each offspring, reset defaults, and append to population.
        """
        for offspring in offsprings:
            offspring.reset_defaults()
            # Stronger exploration for children
            offspring.neural_net.mutate(mutation_scale=0.2)
            self.population.append(offspring)

    def save_generation_samples(self, gen_idx: int, top_n: int = 2) -> None:
        """
        Save the top N players of the current generation to a checkpoints folder.
        """
        import os
        os.makedirs("checkpoints", exist_ok=True)

        # Save top N individuals (p0gen0, p1gen0, etc.)
        for rank in range(min(top_n, len(self.population))):
            player = self.population[rank]
            path = f"checkpoints/p{rank}gen{gen_idx}.pt"
            player.neural_net.save_weights(path)

        # Continue saving the overall best for tester.py
        self.population[0].neural_net.save_weights("elite_model.pt")


    def print_scores(self) -> None:
        """
        Print scores of the current generation.
        """
        #for i in range(len(self.population)):
        #    print("Player {0} : {1}".format(i, self.population[i].fit_scores))
        print("Scores : {0}, Elite player age = {1}".format(self.population[0].scores, self.population[0].age))
