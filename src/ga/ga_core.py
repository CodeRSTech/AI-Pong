# src/ga_core.py
"""
Genetic Algorithm core: selection, crossover, mutation, and population management.
"""

import random
from copy import deepcopy

from src.ga import IndividualPlayer
from src.game import Game
from src.utils import two_point_crossover, logger

class GeneticAlgorithm:
    """
    A class representing core functionality of the Genetic Algorithm.
    """
    # Elite factor decides what percentage of top individuals will be retained onto the next gen.
    elite_factor = 0.1
    # Crossover rate determines what percentage of individuals will join the mating pool.
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

        For each generation, 3 key steps are executed:

        - Selection,
        - Crossover and,
        - Mutation.
        """
        logger.info("Starting GA... "
                    "will run for upto {0} generations.".format(runs))
        # TODO: the `i` variable here, referred to as `gen_idx` in `selection` method,
        #  `calculate_fitness` method and, finally, `save_generation_samples`.
        #  This appears to be unnecessary and thus requires rectification.
        for i in range(runs):
            logger.info("Generation: {0}, Population: {1}".format(i, self.population_size))
            self.selection(i)
            offsprings = self.crossover()
            self.mutate_and_append_to_population(offsprings)

    def selection(self, gen_idx: int) -> None:
        """
        Run game for one generation, evaluate individual fitness, and prepare the next generation.
        """
        self.epoch()
        logger.info("Performing selection...")
        try:
            self.calculate_fitness(gen_idx)
            self.cull_population()
            self.repopulate()
        # FIXME: Too broad exception clause
        except LookupError:
            logger.opt(exception=True).error("LookupError")
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
            try:
                player.calculate_fitness()
            except KeyError:
                logger.opt(exception=True).error("Unable to read a certain key while calculating fitness,"
                                                 "setting fitness score to zero.")
                player.scores['fitness'] = 0

        self.population = sorted(self.population,
                                 key=lambda player_i: player_i.scores['fitness'],
                                 reverse=True)

        # TODO: Make this method get `gen_idx` from parent class (i.e. self),
        #  rather than having to pass it down through multiple parent methods.
        self.save_generation_samples(gen_idx)  # <-- Call the new save method
        self.print_scores()

    def cull_population(self) -> None:
        population = self.population
        # Get the fittest individual...
        # Since fitness is calculated AND,
        # population is sorted by fitness in decreasing order,
        # the fittest individual lies at index 0 while,
        # the weakest individual lies at the last index, i.e. -1 .
        fittest = population[0]
        weakest = population[-1]

        # Get elites (top players)
        num_elites = int(len(population) * self.elite_factor)
        elites = population[:num_elites]
        logger.info("Elite players chosen: {0}".format(num_elites))

        max_score = fittest.get_fitness()
        min_score = weakest.get_fitness()
        score_range = max_score - min_score

        # NEW!!! elites are now kept as a deep copy
        survivors = (deepcopy(elites) +
                     deepcopy(elites) +
                     deepcopy(elites))

        for player in population:
            player_score = player.get_fitness()

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
        population = self.population
        target_size = self.population_size
        current_size = len(population)
        logger.debug("Target population size is {0}, current size is {1}".format(target_size, current_size))

        if current_size < target_size:
            logger.info("Adding new pupils...")
            need = target_size - current_size
            for _ in range(need):
                new_pupil = IndividualPlayer()
                population.append(new_pupil)
        elif current_size > target_size:
            logger.info("Removing the worst...")
            # Trim excess (assumes population is already sorted by fitness descending)
            del population[target_size:]

    def crossover(self) -> list:
        """
        Perform crossover and return new offsprings.
        """
        logger.info("Performing crossover...")
        offsprings = []
        num_to_mate = int(self.population_size * self.crossover_rate)

        if num_to_mate % 2 != 0:
            num_to_mate += 1
            logger.warning("Mating pool size is not even. Adding one more individual to mate.")
        mating_pool = self.population[:num_to_mate]

        logger.debug("Mating pool size is {0}, maintaining diversity...".format(num_to_mate))
        random.shuffle(mating_pool)

        for i in range(0, len(mating_pool), 2):
            try:
                parent_1, parent_2 = mating_pool[i], mating_pool[i+1]
            except IndexError:
                logger.opt(exception=True).error("CRITICAL ERROR:\n"
                                                 "Index error occurred while attempting to crossover.\n"
                                                 "Are there sufficient individuals in the mating pool?"
                                                 " (current mating pool size = {0}.".format(len(mating_pool)))
                exit(-1)
            else:
                offspring1 = two_point_crossover(parent_1, parent_2)
                offspring2 = two_point_crossover(parent_2, parent_1)
                offsprings.extend([offspring1, offspring2])

        logger.debug("Number of Offsprings: {0}".format(len(offsprings)))
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
        Save the top N players of the current generation to a checkpoints' folder.
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
        for i in range(1):
            print(f"P{i}: {self.population[i].get_scores()}\n")
