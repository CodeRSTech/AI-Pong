import pytest

from src.audio import SoundEffects
from src.components.playzone import PlayZone
from src.ga.player import IndividualPlayer


class FakePlayer:
    def __init__(self):
        self.playing = True
        self.queued = None
        self.volume = None

    def queue(self, source):
        self.queued = source

    def play(self):
        pass


def test_sound_effects_synthesize_tones_and_apply_cooldowns(monkeypatch):
    players = []
    now = [10.0]

    def create_player():
        player = FakePlayer()
        players.append(player)
        return player

    monkeypatch.setattr("src.audio.Player", create_player)
    monkeypatch.setattr("src.audio.monotonic", lambda: now[0])
    effects = SoundEffects()

    assert effects.play("paddle")
    assert not effects.play("paddle")
    now[0] += 0.06
    assert effects.play("paddle")
    assert len(players) == 2
    assert players[0].queued.duration == 0.065
    assert players[0].volume == 0.10


def test_sound_effects_reject_unknown_events():
    with pytest.raises(ValueError, match="Unknown sound event"):
        SoundEffects().play("unknown")


def test_playzone_reports_wall_paddle_and_score_events():
    events = []
    zone = PlayZone(400, 500, 2.5, IndividualPlayer(), sound_event_callback=events.append)

    zone.ball.center = (0, 100)
    zone.check_collisions()
    zone.handle_collision(zone.ball, zone.ai_paddle)
    zone.ball.center = (200, 510)
    zone.check_collisions()

    assert events == ["wall", "paddle", "score"]
