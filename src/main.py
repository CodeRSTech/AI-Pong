# src/main.py
"""
Entry point for running the Genetic Algorithm end-to-end.
"""

from numpy import random
from ga.player import IndividualPlayer
from ga.ga_core import GeneticAlgorithm
from utils.logger import logger

random.seed(383)

if __name__ == "__main__":
    # Create Initial Population
    population_size = 50
    initial_population = [IndividualPlayer() for _ in range(population_size)]

    # Load and run the ga
    ga = GeneticAlgorithm(initial_population)
    try:
        ga.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
        exit(0)
