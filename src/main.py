import pygame as pg
from numpy import random
from player import IndividualPlayer
from ga_core import GeneticAlgorithm

random.seed(3683)

# Create Initial Population
population_size = 100
initial_population = [IndividualPlayer() for _ in range(population_size)]

# Load the GA
ga = GeneticAlgorithm(initial_population)

# run
ga.start()
pg.quit()
