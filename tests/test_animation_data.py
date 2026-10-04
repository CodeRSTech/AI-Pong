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
    assert len(example.observation) == example.sizes[0]
    assert [len(values) for values in example.preactivations] == example.sizes[1:]
    assert [tuple(weights.shape) for weights in example.weights] == [
        (8, 7), (6, 8), (2, 6)
    ]
    assert [len(values) for values in example.biases] == example.sizes[1:]
    assert example.decisions == (example.values[-1] > 0.5).tolist()
    assert example.focused_neuron == 3
    assert len(example.focused_contributions) == example.sizes[0]
    np.testing.assert_array_equal(
        example.focused_contributions,
        example.observation * example.weights[0][example.focused_neuron],
    )
    assert example.focused_bias == example.biases[0][example.focused_neuron]
    assert example.focused_preactivation == example.preactivations[0][
        example.focused_neuron
    ]
    np.testing.assert_allclose(
        example.focused_contributions.sum() + example.focused_bias,
        example.focused_preactivation,
        rtol=1e-6,
        atol=1e-6,
    )
    assert example.focused_activation == example.values[1][example.focused_neuron]
    for layer_index, (matrix, sources) in enumerate(zip(
        example.weights, example.dominant_sources
    )):
        expected_sources = np.abs(
            example.values[layer_index][None, :] * matrix
        ).argmax(axis=1)
        np.testing.assert_array_equal(sources, expected_sources)
    other = network_example()
    for values, repeated in zip(example.values, other.values):
        np.testing.assert_array_equal(values, repeated)
    for values, repeated in zip(example.preactivations, other.preactivations):
        np.testing.assert_array_equal(values, repeated)
    for values, repeated in zip(example.weights, other.weights):
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
