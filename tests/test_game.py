from types import SimpleNamespace

from src.game import Game
from src.components.geometry import Vec2
from src.components.colors import CPU_ACCENT, PLAYER_ACCENT
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


def test_paddle_collision_counts_edge_touch_as_a_hit():
    game = Game([IndividualPlayer()], width=400, height=500, timeout=-1)
    zone = game.zones[0]
    zone.ball.center = (zone.ai_paddle.pos_x, zone.ai_paddle.top - zone.ball.height / 2)
    zone.ball.speed = Vec2(0, 2)

    zone.check_collisions()

    assert zone.ai_player.scores["Player Hits"] == 1


def test_paddles_use_distinct_render_colors_without_changing_game_state():
    game = Game([IndividualPlayer()], width=400, height=500, timeout=-1)
    zone = game.zones[0]

    assert zone.ai_paddle.color == PLAYER_ACCENT
    assert zone.cpu_paddle.color == CPU_ACCENT


def test_only_the_visible_zone_routes_sound_events():
    game = Game([IndividualPlayer(), IndividualPlayer()], sound_enabled=True)
    events = []
    game._sound_effects = SimpleNamespace(play=events.append)
    game._display_zone = game.zones[0]

    game.zones[0]._emit_sound_event("paddle")
    game.zones[1]._emit_sound_event("wall")

    assert events == ["paddle"]


def test_headless_game_does_not_route_sound_events():
    game = Game([IndividualPlayer()], sound_enabled=False)
    events = []
    game._sound_effects = SimpleNamespace(play=events.append)
    game.zones[0]._emit_sound_event("score")

    assert events == []
