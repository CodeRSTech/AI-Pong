"""Generated sound effects for rendered Pong gameplay."""

from time import monotonic

from pyglet.media import Player
from pyglet.media.synthesis import LinearDecayEnvelope, Sine


class SoundEffects:
    """Play short, rate-limited synthesized tones without audio asset files."""

    _effects = {
        "wall": (520, 0.045, 0.07, 0.055),
        "paddle": (760, 0.065, 0.10, 0.055),
        "score": (360, 0.14, 0.16, 0.18),
    }

    def __init__(self):
        self._last_played = {}
        self._players = []

    def play(self, event: str) -> bool:
        """Play an event tone if its cooldown has elapsed; return whether it played."""
        if event not in self._effects:
            raise ValueError(f"Unknown sound event: {event}")

        frequency, duration, volume, cooldown = self._effects[event]
        now = monotonic()
        if now - self._last_played.get(event, float("-inf")) < cooldown:
            return False

        # Keep only active players so rapid rallies do not retain finished audio.
        self._players = [player for player in self._players if player.playing]
        player = Player()
        player.volume = volume
        player.queue(Sine(duration, frequency, envelope=LinearDecayEnvelope()))
        player.play()
        self._players.append(player)
        self._last_played[event] = now
        return True

    def close(self) -> None:
        """Release any playing pyglet players when the owning Arcade window closes."""
        for player in self._players:
            player.pause()
            player.delete()
        self._players.clear()
