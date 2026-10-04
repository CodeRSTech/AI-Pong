"""Source-derived examples shared by animation scenes and renderer-free tests."""

from dataclasses import dataclass

import numpy as np
import torch

from scripts.generate_diagrams import _diagram_cut_points, _network_spec
from src.ga.player import ACTION_LABELS, OBSERVATION_LABELS, IndividualPlayer
from src.utils.functions import (
    crossover_neuron_masks,
    crossover_parameter_masks,
    two_point_crossover,
)


@dataclass
class NetworkExample:
    sizes: list[int]
    activations: list[str]
    inputs: tuple[str, ...]
    actions: tuple[str, ...]
    values: list[np.ndarray]
    decisions: list[bool]


@dataclass
class CrossoverExample:
    sizes: list[int]
    cut_points: tuple[int, int]
    neurons: list[list[bool]]
    weights: list[list[list[bool]]]
    biases: list[list[bool]]


def network_example() -> NetworkExample:
    # Isolate the demonstration from any training RNG state in the calling process.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(334)
        player = IndividualPlayer()
        sizes, activations = _network_spec(player)
        observation = np.array(
            [0.25, -0.5, -0.2, 0.3, -0.1, 0.6, 0.8], dtype=np.float32
        )
        if len(observation) != sizes[0]:
            raise ValueError("Update the example observations for the current network.")
        decisions = player.think(observation)[0].astype(bool).tolist()
        values = [value[0].copy() for value in player.neural_net.last_activations]
    return NetworkExample(
        sizes, activations, OBSERVATION_LABELS, ACTION_LABELS, values, decisions
    )


def crossover_example() -> CrossoverExample:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(334)
        parent_a, parent_b = IndividualPlayer(), IndividualPlayer()
        sizes, _ = _network_spec(parent_a)
        cut_points = _diagram_cut_points(sum(sizes[1:]))
        neurons = crossover_neuron_masks(parent_a.neural_net, cut_points)
        weights, biases = crossover_parameter_masks(parent_a.neural_net, cut_points)

        # Sentinel parameters identify the donor without approximating crossover.
        with torch.no_grad():
            for parent, value in ((parent_a, 1.0), (parent_b, 2.0)):
                for parameter in parent.neural_net.parameters():
                    parameter.fill_(value)
        for first, second, selected_value, outside_value in (
            (parent_a, parent_b, 2.0, 1.0),
            (parent_b, parent_a, 1.0, 2.0),
        ):
            child = two_point_crossover(first, second, cut_points=cut_points)
            for index, layer in enumerate(child.neural_net.layers):
                weight_mask = torch.tensor(weights[index])
                bias_mask = torch.tensor(biases[index])
                if not torch.equal(
                    layer[0].weight,
                    torch.where(weight_mask, selected_value, outside_value),
                ) or not torch.equal(
                    layer[0].bias,
                    torch.where(bias_mask, selected_value, outside_value),
                ):
                    raise ValueError("Animation masks no longer match actual crossover.")
    return CrossoverExample(sizes, cut_points, neurons, weights, biases)
