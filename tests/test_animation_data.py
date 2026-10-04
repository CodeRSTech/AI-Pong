import numpy as np
import torch

from scripts.animation_data import crossover_example, network_example


def test_network_animation_uses_live_values_and_preserves_rng():
    state = torch.random.get_rng_state().clone()
    example = network_example()

    assert torch.equal(state, torch.random.get_rng_state())
    assert example.sizes == [7, 8, 6, 2]
    assert example.activations == ["tanh", "ReLU", "sigmoid"]
    assert [len(values) for values in example.values] == example.sizes
    assert example.decisions == (example.values[-1] > 0.5).tolist()
    other = network_example()
    for values, repeated in zip(example.values, other.values):
        np.testing.assert_array_equal(values, repeated)


def test_crossover_animation_verifies_real_reciprocal_children_and_preserves_rng():
    state = torch.random.get_rng_state().clone()
    example = crossover_example()

    assert torch.equal(state, torch.random.get_rng_state())
    assert example.cut_points == (3, 12)
    assert sum(sum(mask) for mask in example.neurons) == 9
    assert [len(mask) for mask in example.biases] == [8, 6, 2]
    assert [(len(mask), len(mask[0])) for mask in example.weights] == [(8, 7), (6, 8), (2, 6)]
    assert example == crossover_example()
