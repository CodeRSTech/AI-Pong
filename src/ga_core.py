import pickle
import random

from game import Game
from player import IndividualPlayer
from functions import two_point_crossover


class GeneticAlgorithm:
    """
    A class representing core functionality of the Genetic Algorithm.

    """

    def __init__(self, population: list[IndividualPlayer]):
        """
        Initialize a Genetic Algorithm with Initial Population `population`.
        :type population: list
        :param population: Initial Population.
        :return: None
        """
        self.population = population
        self.population_size = len(population)

    def start(self, runs=100) -> None:
        """
        Run the Genetic Algorithm for a given number of generations.
        :param runs:Number of generations.
        :return: None
        """
        # For each generation
        for i in range(runs):
            print("Gen: {0}".format(i))

            # Perform selection
            self.selection()

            # Generate offspring by performing crossover
            offsprings = self.crossover()

            # Perform mutation and add offspring to the population
            self.mutate_and_append_to_population(offsprings)

    def selection(self) -> None:
        """
        Run the Genetic Algorithm and adjust the population.
        :return: None
        """

        # Play the game
        self.epoch()

        # Adjust the population
        try:
            self.calculate_fitness()
            self.cull_population()
            self.repopulate()
        except LookupError:
            exit(0)

    def epoch(self) -> None:
        """
        Play the game for one generation.
        :return: None
        """

        # Shuffle the population
        random.shuffle(self.population)

        # Start the game
        game = Game(self.population)
        self.population = game.start()

        # Update the population data
        for player in self.population:
            player.age += 1
        self.population_size -= 1

    def calculate_fitness(self) -> None:
        """
        Calculate the fitness values for each player.
        :return: None
        """

        # Calculate fitness
        for player in self.population:
            player.calculate_fitness()

        # Update population
        self.population = sorted(self.population,
                                 key=lambda player_i: player_i.scores['overall_fitness'],
                                 reverse=True)

        # Save the fittest individual to disk
        self.save_elite()
        self.print_scores()

    def cull_population(self) -> None:
        """
        Eliminate the least fit individuals.
        :return: None
        """

        # Get elite individual's data
        fittest = self.population[0]
        max_score = fittest.scores['overall_fitness']

        # Retain best-performers
        random_factor = random.random()
        survivors = self.population[:5]
        for player in self.population[5:]:                      # for each player in population after index 5
            player_score = player.scores['overall_fitness']     # get its score
            try:
                favourable_factor = (player_score / max_score)  # favour = score /max_score
            except ZeroDivisionError:
                favourable_factor = 1
            if random_factor < favourable_factor:               # if random factor is in favour
                survivors.append(player)                        # then add player to the survivors' list
            self.population = survivors

    def repopulate(self) -> None:
        """
        Add newly generated individuals to the population.
        :return: None
        """
        while len(self.population) != self.population_size:
            new_pupil = IndividualPlayer()
            self.population.append(new_pupil)

    def crossover(self) -> list:
        """
        Perform crossover.
        :return: list of individuals
        """
        offsprings = []

        # Cross P1 with P2
        p1, p2 = self.select_parents()
        offsprings += [two_point_crossover(p1, p2),
                       two_point_crossover(p2, p1)]

        # Cross P1 with a randomly chosen individual
        random_individual = self.population[random.randint(2, self.population_size)]
        random_offsprings = [two_point_crossover(p1, random_individual),
                             two_point_crossover(random_individual, p1)]

        # More preference to P1,P2 offspring
        return offsprings + offsprings + random_offsprings

    def mutate_and_append_to_population(self, offsprings: list[IndividualPlayer]) -> None:
        """
        mutate each offspring and append them to population, also reset uid/stats
        :param offsprings: list of newborn IndividualPlayer with new UID's
        :return: None
        """
        for offspring in offsprings:
            offspring.reset_defaults()
            offspring.neural_net.mutate()
            self.population.append(offspring)

    def select_parents(self) -> tuple:
        """
        Select the two fittest individuals.
        :return: parent_1, parent_2
        """
        parent_1 = self.population[0]
        parent_2 = self.population[1]
        return parent_1, parent_2

    def save_elite(self) -> None:
        """
        Write the weights and biases of the elite player to file.
        :return: None
        """
        elite = self.population[0]

        for i, layer in enumerate(elite.neural_net.layers):
            weights_file = open("layer{0}.weights".format(i), 'wb')
            biases_file = open("layer{0}.biases".format(i), 'wb')
            pickle.dump(layer.weights, weights_file)
            pickle.dump(layer.biases, biases_file)
            weights_file.close()
            biases_file.close()

    def print_scores(self) -> None:
        """
        Print scores of the current generation.
        :return: None
        """
        print("Scores : {0}, Elite player age = {1}".format(self.population[0].scores, self.population[0].age))
