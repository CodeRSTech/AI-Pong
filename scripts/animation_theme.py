"""Shared terminal-inspired visual helpers for the optional Manim scenes."""

from manim import (
    Circle,
    FadeIn,
    Line,
    Rectangle,
    Text,
    VGroup,
    config,
    RoundedRectangle,
)
from manimpango import list_fonts

from src.components.colors import WEIGHT_NEGATIVE, WEIGHT_POSITIVE


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


COLORS = {
    "background": "#09090b",
    "surface": "#121212",
    "border": "#37373a",
    "text": "#f4f4f5",
    "muted": "#a1a1aa",
    "subtle": "#71717a",
    "neutral": "#27272a",
    "cyan": "#00ffff",
    "green": "#00ff66",
    "magenta": "#ff00ff",
    "parent_a": "#00ffff",
    "parent_b": "#00ff66",
    "cut": "#ff00ff",
    "weight_positive": _rgb_hex(WEIGHT_POSITIVE),
    "weight_negative": _rgb_hex(WEIGHT_NEGATIVE),
}

_AVAILABLE_FONTS = set(list_fonts())
if "Fira Code" in _AVAILABLE_FONTS:
    FONT_NAME = "Fira Code"
elif "Consolas" in _AVAILABLE_FONTS:
    FONT_NAME = "Consolas"
else:
    raise RuntimeError(
        "Manim scenes require Fira Code or the documented monospace fallback Consolas."
    )


def font_notice() -> str:
    """Describe the selected font so fallback selection is visible to renderers."""
    if FONT_NAME == "Fira Code":
        return "Animation font: Fira Code."
    return "Fira Code is unavailable; using the installed monospace fallback Consolas."


def text(content: str, size: float = 20, color: str | None = None) -> Text:
    return Text(
        content,
        font=FONT_NAME,
        font_size=size,
        color=color or COLORS["text"],
    )


def background() -> VGroup:
    """Build the static grid and understated top glow used by every scene."""
    width, height = config.frame_width, config.frame_height
    layers = [Rectangle(
        width=width,
        height=height,
        stroke_width=0,
        fill_color=COLORS["background"],
        fill_opacity=1,
    ).set_z_index(-100)]
    grid = VGroup()
    spacing = 0.5
    x = -width / 2
    while x <= width / 2:
        grid.add(Line(
            [x, -height / 2, 0], [x, height / 2, 0],
            color=COLORS["text"], stroke_width=0.35, stroke_opacity=0.035,
        ))
        x += spacing
    y = -height / 2
    while y <= height / 2:
        grid.add(Line(
            [-width / 2, y, 0], [width / 2, y, 0],
            color=COLORS["text"], stroke_width=0.35, stroke_opacity=0.035,
        ))
        y += spacing
    grid.set_z_index(-99)
    layers.append(grid)
    for index in range(6, 0, -1):
        glow = Circle(radius=index * 0.8, stroke_width=0)
        glow.set_fill(COLORS["green"], opacity=0.008)
        glow.move_to([0, height / 2 + 0.9, 0]).set_z_index(-98)
        layers.append(glow)
    return VGroup(*layers)


def terminal_header(scene, title: str, subtitle: str, stage: str) -> VGroup:
    title_line = text(title, 29).to_edge([0, 1, 0], buff=0.48)
    if title_line.width > config.frame_width - 5.0:
        title_line.scale_to_fit_width(config.frame_width - 5.0)
    subtitle_line = text(subtitle, 13, COLORS["muted"]).next_to(
        title_line, [0, -1, 0], buff=0.12
    )
    stage_tag = text(f"[ {stage} ]", 12, COLORS["green"]).to_corner(
        [1, 1, 0], buff=0.5
    )
    group = VGroup(title_line, subtitle_line, stage_tag)
    scene.play(*[FadeIn(item) for item in group], run_time=0.45)
    return group


def card(width: float, height: float) -> RoundedRectangle:
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.08,
        stroke_color=COLORS["border"],
        stroke_width=1,
        fill_color=COLORS["surface"],
        fill_opacity=0.94,
    )


def callout(content: str, size: float = 17, color: str | None = None) -> Text:
    return text(content, size, color or COLORS["cyan"])


def signed_color(value: float) -> str:
    """Color a signed network value independently of parent/action ownership."""
    if value > 0:
        return COLORS["weight_positive"]
    if value < 0:
        return COLORS["weight_negative"]
    return COLORS["neutral"]


def signed_text_color(value: float) -> str:
    """Keep neutral numeric labels legible while their node fill stays neutral."""
    return COLORS["muted"] if value == 0 else signed_color(value)


def signed_opacity(value: float) -> float:
    """Keep exact zero neutral and scale nonzero signed values by magnitude."""
    if value == 0:
        return 1.0
    return 0.25 + 0.75 * min(abs(value), 1.0)
