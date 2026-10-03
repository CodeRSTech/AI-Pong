import pytest

from src.ga.player import IndividualPlayer


def test_new_player_has_architecture_and_zeroed_scores():
    p = IndividualPlayer()
    assert len(p.neural_net.layers) == 3
    assert p.scores['fitness'] == 0
    assert p.scores['Player'] == 0


def test_reset_hit_streak_tracks_max_hit_streak():
    p = IndividualPlayer()
    for _ in range(3):
        p.add_hit_to_streak()
    p.reset_hit_streak()
    assert p.max_hit_streak == 3
    assert p.current_hit_streak == 0


def test_reset_winning_streak_tracks_win_streak_only():
    p = IndividualPlayer()
    p.add_win_to_streak()
    p.add_win_to_streak()
    p.add_hit_to_streak()
    p.reset_winning_streak()
    assert p.max_win_streak == 2
    assert p.current_win_streak == 0
    assert p.current_hit_streak == 1


def test_fitness_rewards_winning():
    good, bad = IndividualPlayer(), IndividualPlayer()
    good.scores.update({'Player': 5, 'CPU': 0, 'Player Hits': 10})
    bad.scores.update({'Player': 0, 'CPU': 5, 'CPU Hits': 10})
    assert good.calculate_fitness() > bad.calculate_fitness()


def test_str_is_callable():
    assert 'fitness' in str(IndividualPlayer())
