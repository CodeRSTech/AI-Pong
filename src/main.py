# src/main.py
"""
Entry point for running the Genetic Algorithm end-to-end.
"""

from numpy import random
from ga.player import IndividualPlayer
from ga.ga_core import GeneticAlgorithm
from utils.logger import logger

random.seed(38345343)

initial_population_size = 200

if __name__ == "__main__":
    # Create Initial Population
    population_size = initial_population_size
    initial_population = [IndividualPlayer() for _ in range(population_size)]

    # Load and run the ga
    ga = GeneticAlgorithm(initial_population)
    try:
        logger.info(f"Starting The Genetic Algorithm...")
        ga.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
        exit(0)
