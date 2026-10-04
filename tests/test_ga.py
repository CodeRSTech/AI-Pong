import csv
import json

import torch
import pytest

from src.ga import GeneticAlgorithm
from src.ga.network import BatchedPopulationBrain
from src.ga.player import IndividualPlayer
from src.utils.functions import crossover_neuron_masks, two_point_crossover


def test_crossover_keeps_architecture():
    child = two_point_crossover(IndividualPlayer(), IndividualPlayer())
    shapes = [tuple(layer[0].weight.shape) for layer in child.neural_net.layers]
    assert shapes == [(8, 7), (6, 8), (2, 6)]


def test_crossover_inherits_nonempty_neuron_range_and_outgoing_weights():
    parent_1, parent_2 = IndividualPlayer(), IndividualPlayer()
    with torch.no_grad():
        for layer in parent_1.neural_net.layers:
            layer[0].weight.fill_(1)
            layer[0].bias.fill_(1)
        for layer in parent_2.neural_net.layers:
            layer[0].weight.fill_(2)
            layer[0].bias.fill_(2)

    child = two_point_crossover(parent_1, parent_2).neural_net
    selected_masks = []
    for child_layer in child.layers:
        selected = child_layer[0].bias == 2
        assert selected.any()
        assert torch.all(child_layer[0].weight[selected] == 2)
        selected_masks.append(selected)

    for layer_index, source_mask in enumerate(selected_masks[:-1]):
        assert torch.all(child.layers[layer_index + 1][0].weight[:, source_mask] == 2)


def test_crossover_neuron_masks_follow_layer_output_order():
    player = IndividualPlayer()

    masks = crossover_neuron_masks(player.neural_net, (3, 12))

    assert masks == [
        [False, False, False, True, True, True, True, True],
        [True, True, True, True, False, False],
        [False, False],
    ]


@pytest.mark.parametrize("cut_points", [(0, 0), (12, 3), (-1, 3), (3, 17), (True, 3)])
def test_crossover_neuron_masks_reject_invalid_boundaries(cut_points):
    with pytest.raises(ValueError, match="cut_points"):
        crossover_neuron_masks(IndividualPlayer().neural_net, cut_points)


def test_crossover_fixed_interval_uses_shared_masks_for_reciprocal_children():
    parent_1, parent_2 = IndividualPlayer(), IndividualPlayer()
    with torch.no_grad():
        for layer in parent_1.neural_net.layers:
            layer[0].weight.fill_(1)
            layer[0].bias.fill_(1)
        for layer in parent_2.neural_net.layers:
            layer[0].weight.fill_(2)
            layer[0].bias.fill_(2)

    masks = crossover_neuron_masks(parent_1.neural_net, (3, 12))
    child_a = two_point_crossover(
        parent_1, parent_2, cut_points=(3, 12)
    ).neural_net
    child_b = two_point_crossover(
        parent_2, parent_1, cut_points=(3, 12)
    ).neural_net

    for child, selected_value, other_value in (
        (child_a, 2.0, 1.0),
        (child_b, 1.0, 2.0),
    ):
        previous_mask = None
        for layer, mask in zip(child.layers, masks):
            rows_from_selected_parent = torch.tensor(mask)
            expected_bias = torch.where(
                rows_from_selected_parent, selected_value, other_value
            )
            assert torch.equal(layer[0].bias.cpu(), expected_bias)

            rows_from_selected_parent = rows_from_selected_parent[:, None].expand_as(
                layer[0].weight
            ).clone()
            if previous_mask is not None:
                columns_from_selected_parent = torch.tensor(previous_mask)[None, :].expand_as(
                    layer[0].weight
                )
                rows_from_selected_parent |= columns_from_selected_parent
            expected_weight = torch.where(
                rows_from_selected_parent, selected_value, other_value
            )
            assert torch.equal(layer[0].weight.cpu(), expected_weight)
            previous_mask = mask


def test_crossover_rejects_incompatible_networks():
    parent_1, parent_2 = IndividualPlayer(), IndividualPlayer()
    parent_2.neural_net.layers[0][0] = torch.nn.Linear(7, 9)
    with pytest.raises(ValueError, match="matching neural-network architectures"):
        two_point_crossover(parent_1, parent_2)


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


def test_offspring_replace_survivors_without_growing_population():
    population = [IndividualPlayer() for _ in range(10)]
    ga = GeneticAlgorithm(population)
    offspring = [IndividualPlayer(), IndividualPlayer()]
    ga.mutate_and_append_to_population(offspring)

    assert len(ga.population) == 10
    assert ga.population[-2:] == offspring
    assert ga.population[:8] == population[:8]


def test_generation_metrics_and_checkpoints_are_saved(tmp_path):
    population = [IndividualPlayer(), IndividualPlayer()]
    population[0].scores.update({"Player": 2, "Player Hits": 3})
    population[1].scores.update({"CPU": 1, "CPU Hits": 2})
    ga = GeneticAlgorithm(population, output_dir=tmp_path, render=False, timeout=1)
    ga.validator.evaluate = lambda elite, generation: {
        "generation": generation,
        "games": 20,
        "wins": 19,
        "losses": 1,
        "ties": 0,
        "win_rate": 0.95,
        "shutouts": 18,
        "shutout_rate": 0.9,
        "win_rate_target": 0.95,
        "shutout_rate_target": 0.9,
        "reached": True,
        "scenarios": [],
    }

    ga.generation = 4
    ga.calculate_fitness()

    with (ga.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    assert len(rows) == 1
    assert rows[0]["generation"] == "4"
    assert (ga.run_dir / "elite_model.pt").is_file()
    assert (ga.run_dir / "checkpoints" / "p0gen4.pt").is_file()
    assert (ga.run_dir / "validation.csv").is_file()


def test_start_writes_reproducibility_settings(tmp_path):
    ga = GeneticAlgorithm(
        [IndividualPlayer(), IndividualPlayer()],
        output_dir=tmp_path,
        render=False,
        seed=42,
        timeout=3,
        fps=60,
        speed=1.5,
        steps_per_frame=4,
    )
    ga.selection = lambda: None
    ga.crossover = lambda: []
    ga.mutate_and_append_to_population = lambda offsprings: None

    ga.start(runs=2)

    settings = json.loads((ga.run_dir / "settings.json").read_text(encoding="utf-8"))
    assert settings["seed"] == 42
    assert settings["generations"] == 2
    assert settings["population_size"] == 2
    assert settings["steps_per_frame"] == 4


def test_fitness_errors_are_not_silently_ignored(tmp_path):
    player = IndividualPlayer()
    del player.scores["Player"]
    ga = GeneticAlgorithm([player, IndividualPlayer()], output_dir=tmp_path, render=False, timeout=1)

    with pytest.raises(KeyError, match="Player"):
        ga.calculate_fitness()
