from src.game import Game
from src.ga.player import IndividualPlayer


def test_headless_epoch_runs_and_returns_players():
    players = [IndividualPlayer() for _ in range(3)]
    game = Game(players, width=400, height=500, timeout=0.5)
    result = game.run_headless(step_delta=1 / 60)
    assert result == players
    assert game.is_finished


def test_step_moves_the_ball():
    game = Game([IndividualPlayer()], width=400, height=500, timeout=-1)
    ball = game.zones[0].ball
    before = (ball.pos_x, ball.pos_y)
    game.step(1 / 60)
    assert (ball.pos_x, ball.pos_y) != before
    assert not game.is_finished
