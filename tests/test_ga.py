import torch

from src.ga import GeneticAlgorithm
from src.ga.network import BatchedPopulationBrain
from src.ga.player import IndividualPlayer
from src.utils.functions import two_point_crossover


def test_crossover_keeps_architecture():
    child = two_point_crossover(IndividualPlayer(), IndividualPlayer())
    shapes = [tuple(layer[0].weight.shape) for layer in child.neural_net.layers]
    assert shapes == [(8, 7), (6, 8), (2, 6)]


def test_mutation_changes_weights():
    net = IndividualPlayer().neural_net
    before = net.layers[0][0].weight.clone()
    net.mutate(mutation_scale=0.5, mutation_probability=1.0)
    assert not torch.equal(before, net.layers[0][0].weight)


def test_batched_brain_matches_individual_nets():
    players = [IndividualPlayer() for _ in range(5)]
    inputs = torch.rand(5, 7).numpy()
    batched = BatchedPopulationBrain(players).predict_batch(inputs)
    single = [p.neural_net.predict(inputs[i])[0].astype(bool) for i, p in enumerate(players)]
    assert (batched == single).all()


def test_population_size_is_recorded():
    ga = GeneticAlgorithm([IndividualPlayer() for _ in range(10)])
    assert ga.population_size == 10
