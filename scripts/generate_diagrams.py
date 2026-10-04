"""Generate architecture and crossover diagrams from the current AI-Pong model."""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ga.player import ACTION_LABELS, OBSERVATION_LABELS, IndividualPlayer
from src.utils.functions import crossover_neuron_masks, crossover_parameter_masks


COLORS = {
    "background": "#09090b",
    "panel": "#121212",
    "panel_border": "#37373a",
    "text": "#f4f4f5",
    "muted": "#a1a1aa",
    "subtle": "#71717a",
    "neutral": "#27272a",
    "parent_a": "#00ffff",
    "parent_a_fill": "#123b3b",
    "parent_b": "#00ff66",
    "parent_b_fill": "#12351f",
    "cut": "#ff00ff",
}


def _network_spec(player: IndividualPlayer) -> tuple[list[int], list[str]]:
    layers = player.neural_net.layers
    if not layers:
        raise ValueError("Cannot draw a network with no layers.")

    sizes = [layers[0][0].in_features]
    activations = []
    for layer in layers:
        linear, activation = layer[0], layer[1]
        if sizes[-1] != linear.in_features:
            raise ValueError("Network layers must have matching input/output dimensions.")
        sizes.append(linear.out_features)
        activation_name = {
            "Tanh": "tanh",
            "ReLU": "ReLU",
            "Sigmoid": "sigmoid",
        }.get(type(activation).__name__)
        if activation_name is None:
            raise ValueError(
                f"Unsupported activation for diagram: {type(activation).__name__}"
            )
        activations.append(activation_name)

    if len(OBSERVATION_LABELS) != sizes[0]:
        raise ValueError("Observation labels do not match the network input size.")
    if len(ACTION_LABELS) != sizes[-1]:
        raise ValueError("Action labels do not match the network output size.")
    return sizes, activations


def _svg_document(title: str, description: str, width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(description)}</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="6.5" refY="4" orient="auto">
      <path d="M0 0 8 4 0 8z" fill="{COLORS['subtle']}"/>
    </marker>
    <style>
      text {{ font-family: Consolas, "Fira Code", monospace; fill: {COLORS['text']}; }}
      .title {{ font-size: 28px; font-weight: 700; }}
      .subtitle {{ font-size: 14px; fill: {COLORS['muted']}; }}
      .heading {{ font-size: 17px; font-weight: 700; }}
      .body {{ font-size: 13px; fill: {COLORS['muted']}; }}
      .caption {{ font-size: 11px; fill: {COLORS['subtle']}; }}
      .node {{ stroke-width: 2; }}
      .edge {{ fill: none; stroke-width: 1.15; opacity: .62; }}
    </style>
  </defs>
  <rect width="100%" height="100%" rx="18" fill="{COLORS['background']}"/>
{body}
</svg>
"""


def _vertical_positions(count: int, center: float, max_span: float) -> list[float]:
    if count == 1:
        return [center]
    span = min(max_span, (count - 1) * max_span / 7)
    return [
        center - span / 2 + index * span / (count - 1)
        for index in range(count)
    ]


def build_network_svg(player: IndividualPlayer) -> str:
    """Draw the live model's fully connected topology and its observation labels."""
    sizes, activations = _network_spec(player)
    width, height = 1480, 860
    layer_x = [300 + 300 * index for index in range(len(sizes))]
    center_y = 492
    positions = [
        _vertical_positions(size, center_y, 500)
        for size in sizes
    ]
    parts = [
        '<text x="38" y="48" class="title">AI-Pong neural network</text>',
        '<text x="38" y="76" class="subtitle">Layer sizes and activations are read from the current IndividualPlayer model.</text>',
    ]

    for layer_index in range(len(sizes) - 1):
        for source_y in positions[layer_index]:
            for target_y in positions[layer_index + 1]:
                parts.append(
                    f'<line x1="{layer_x[layer_index] + 15}" y1="{source_y:.1f}" '
                    f'x2="{layer_x[layer_index + 1] - 15}" y2="{target_y:.1f}" '
                    'class="edge" stroke="#64748b"/>'
                )

    headings = ["Observations"] + [
        f"Layer {index}" for index in range(1, len(sizes) - 1)
    ] + ["Actions"]
    for index, (size, x, ys, heading) in enumerate(
        zip(sizes, layer_x, positions, headings)
    ):
        parts.append(
            f'<text x="{x}" y="139" text-anchor="middle" class="heading">'
            f'{html.escape(heading)} ({size})</text>'
        )
        if index:
            parts.append(
                f'<text x="{x}" y="161" text-anchor="middle" class="caption">'
                f'{html.escape(activations[index - 1])}</text>'
            )
        else:
            parts.append(
                f'<text x="{x}" y="161" text-anchor="middle" class="caption">normalized game state</text>'
            )

        node_color = (
            COLORS["neutral"] if index == 0 else
            COLORS["parent_a_fill"] if index < len(sizes) - 1 else
            COLORS["parent_b_fill"]
        )
        node_stroke = (
            COLORS["subtle"] if index == 0 else
            COLORS["parent_a"] if index < len(sizes) - 1 else
            COLORS["parent_b"]
        )
        for neuron_index, y in enumerate(ys):
            parts.append(
                f'<circle cx="{x}" cy="{y:.1f}" r="13" fill="{node_color}" '
                f'stroke="{node_stroke}" class="node"/>'
            )
            parts.append(
                f'<text x="{x}" y="{y + 4:.1f}" text-anchor="middle" class="caption">'
                f'{neuron_index}</text>'
            )

    for index, label in enumerate(OBSERVATION_LABELS):
        parts.append(
            f'<text x="{layer_x[0] - 25}" y="{positions[0][index] + 4:.1f}" '
            f'text-anchor="end" class="body">{html.escape(label)}</text>'
        )
    for index, label in enumerate(ACTION_LABELS):
        parts.append(
            f'<text x="{layer_x[-1] + 25}" y="{positions[-1][index] + 4:.1f}" '
            f'text-anchor="start" class="body">{html.escape(label)}</text>'
        )

    parts.append(
        f'<text x="{width - 38}" y="{height - 28}" text-anchor="end" class="caption">'
        "Every neuron connects to every neuron in the next layer.</text>"
    )
    return _svg_document(
        "AI-Pong neural network",
        "Fully connected neural network with live layer sizes, activations, game observations, and paddle actions.",
        width,
        height,
        "\n".join(parts),
    )


def _diagram_cut_points(total_neurons: int) -> tuple[int, int]:
    start = max(0, total_neurons // 5)
    end = min(total_neurons, (total_neurons * 4) // 5)
    if end <= start:
        if total_neurons < 2:
            raise ValueError("Crossover diagram needs at least two output neurons.")
        start, end = 0, total_neurons
    return start, end


def _draw_genome_strip(
    parts: list[str],
    masks: list[list[bool]],
    cut_points: tuple[int, int],
    y: int,
    child: str,
) -> None:
    start_x = 190
    cell_width = 48
    selected_parent = "parent_b" if child == "Child A" else "parent_a"
    other_parent = "parent_a" if child == "Child A" else "parent_b"
    offset = 0

    for layer_index, mask in enumerate(masks):
        block_start = start_x + offset * cell_width
        block_end = block_start + len(mask) * cell_width
        layer_name = (
            f"Hidden {layer_index + 1}" if layer_index < len(masks) - 1 else "Output"
        )
        parts.append(
            f'<text x="{(block_start + block_end) / 2:.1f}" y="{y - 12}" '
            f'text-anchor="middle" class="caption">{layer_name} neurons</text>'
        )
        for local_index, is_selected in enumerate(mask):
            origin = selected_parent if is_selected else other_parent
            x = start_x + (offset + local_index) * cell_width
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="32" '
                f'fill="{COLORS[origin + "_fill"]}" stroke="{COLORS[origin]}" '
                'stroke-width="1.2"/>'
            )
            parts.append(
                f'<text x="{x + cell_width / 2:.1f}" y="{y + 21}" text-anchor="middle" '
                f'class="caption">{offset + local_index}</text>'
            )
        offset += len(mask)

    for cut_index, cut_point in enumerate(cut_points, start=1):
        x = start_x + cut_point * cell_width
        parts.extend([
            f'<line x1="{x}" y1="{y - 5}" x2="{x}" y2="{y + 39}" '
            f'stroke="{COLORS["cut"]}" stroke-width="2.5" stroke-dasharray="5 4"/>',
            f'<text x="{x}" y="{y + 53}" text-anchor="middle" class="caption" '
            f'fill="{COLORS["cut"]}">cut {cut_index}</text>',
        ])


def _draw_child_network(
    parts: list[str],
    sizes: list[int],
    masks: list[list[bool]],
    weight_masks: list[list[list[bool]]],
    bias_masks: list[list[bool]],
    child: str,
    row_top: int,
) -> None:
    child_a = child == "Child A"
    x_positions = [220 + index * 235 for index in range(len(sizes))]
    center_y = row_top + 282
    positions = [
        _vertical_positions(size, center_y, 205)
        for size in sizes
    ]

    for layer_index in range(len(sizes) - 1):
        for source_index, source_y in enumerate(positions[layer_index]):
            for target_index, target_y in enumerate(positions[layer_index + 1]):
                from_selected_parent = weight_masks[layer_index][target_index][source_index]
                if child_a:
                    origin = "parent_b" if from_selected_parent else "parent_a"
                else:
                    origin = "parent_a" if from_selected_parent else "parent_b"
                parts.append(
                    f'<line x1="{x_positions[layer_index] + 10}" y1="{source_y:.1f}" '
                    f'x2="{x_positions[layer_index + 1] - 10}" y2="{target_y:.1f}" '
                    f'class="edge" stroke="{COLORS[origin]}"/>'
                )

    for layer_index, (size, x, ys) in enumerate(
        zip(sizes, x_positions, positions)
    ):
        label = (
            "Inputs" if layer_index == 0 else
            "Output actions" if layer_index == len(sizes) - 1 else
            f"Hidden {layer_index}"
        )
        parts.append(
            f'<text x="{x}" y="{row_top + 167}" text-anchor="middle" class="caption">'
            f'{label} · {size}</text>'
        )
        for neuron_index, y in enumerate(ys):
            if layer_index == 0:
                fill, stroke = COLORS["neutral"], COLORS["subtle"]
            else:
                selected_parent = (
                    "parent_b" if child_a else "parent_a"
                )
                other_parent = (
                    "parent_a" if child_a else "parent_b"
                )
                origin = (
                    selected_parent
                    if bias_masks[layer_index - 1][neuron_index]
                    else other_parent
                )
                fill, stroke = COLORS[origin + "_fill"], COLORS[origin]
            parts.append(
                f'<circle cx="{x}" cy="{y:.1f}" r="10" fill="{fill}" '
                f'stroke="{stroke}" class="node"/>'
            )


def build_crossover_svg(
    player: IndividualPlayer, cut_points: tuple[int, int] | None = None
) -> str:
    """Draw reciprocal child inheritance using the GA's shared neuron masks."""
    sizes, _ = _network_spec(player)
    total_neurons = sum(sizes[1:])
    cut_points = cut_points or _diagram_cut_points(total_neurons)
    masks = crossover_neuron_masks(player.neural_net, cut_points)
    weight_masks, bias_masks = crossover_parameter_masks(
        player.neural_net, cut_points
    )
    width, height = 1500, 1010
    parts = [
        '<text x="38" y="48" class="title">Two-point crossover inheritance</text>',
        '<text x="38" y="76" class="subtitle">Neuron selection and parameter ownership use the same interval masks as the genetic algorithm.</text>',
        f'<rect x="35" y="100" width="{width - 70}" height="410" rx="14" '
        f'fill="{COLORS["panel"]}" stroke="{COLORS["panel_border"]}"/>',
        f'<rect x="35" y="525" width="{width - 70}" height="410" rx="14" '
        f'fill="{COLORS["panel"]}" stroke="{COLORS["panel_border"]}"/>',
    ]

    for child, row_top in (("Child A", 105), ("Child B", 530)):
        parts.append(
            f'<text x="55" y="{row_top + 32}" class="heading">{child}</text>'
        )
        parts.append(
            f'<text x="55" y="{row_top + 53}" class="caption">'
            f'{"selected interval" if child == "Child A" else "reciprocal interval"}</text>'
        )
        _draw_genome_strip(parts, masks, cut_points, row_top + 85, child)
        _draw_child_network(
            parts, sizes, masks, weight_masks, bias_masks, child, row_top
        )

    legend_x = 1212
    for y, parent, description in (
        (276, "parent_a", "Parent A parameters"),
        (310, "parent_b", "Parent B parameters"),
        (344, "neutral", "Input observations"),
    ):
        parts.append(
            f'<circle cx="{legend_x}" cy="{y}" r="9" fill="{COLORS[parent + "_fill"] if parent != "neutral" else COLORS["neutral"]}" '
            f'stroke="{COLORS[parent]}" class="node"/>'
        )
        parts.append(
            f'<text x="{legend_x + 18}" y="{y + 5}" class="caption">{description}</text>'
        )
    parts.extend([
        f'<text x="{legend_x}" y="386" class="caption">Node outline/fill: neuron bias source</text>',
        f'<text x="{legend_x}" y="405" class="caption">Edge color: weight source</text>',
        f'<text x="{legend_x}" y="424" class="caption">Cut points: [{cut_points[0]}, {cut_points[1]})</text>',
        '<text x="38" y="974" class="caption">Selected neurons inherit incoming weights and biases; connections from selected neurons in the previous layer inherit from that same parent.</text>',
        '<text x="38" y="993" class="caption">The strip shows the exact concatenated layer-output index order. Mutation is applied to offspring after crossover.</text>',
    ])
    return _svg_document(
        "AI-Pong two-point crossover",
        "The exact half-open neuron interval used to make reciprocal children, with neuron bias and connection weight provenance shown from the shared GA masks.",
        width,
        height,
        "\n".join(parts),
    )


def generate_diagrams(output_dir: Path = ROOT / "docs" / "images") -> tuple[Path, Path]:
    player = IndividualPlayer()
    output_dir.mkdir(parents=True, exist_ok=True)
    network_path = output_dir / "neural-network.svg"
    crossover_path = output_dir / "two-point-crossover.svg"
    network_path.write_text(build_network_svg(player), encoding="utf-8")
    crossover_path.write_text(build_crossover_svg(player), encoding="utf-8")
    return network_path, crossover_path


if __name__ == "__main__":
    for output_path in generate_diagrams():
        print(f"Wrote {output_path.relative_to(ROOT)}")
