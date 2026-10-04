"""Manim scenes for the network forward pass and two-point crossover videos.

Render both from the repository root with ``python -m scripts.render_animations``.
The scenes are organized as short story phases; visual text, frame coordinates,
and ``run_time`` / ``wait`` values are the main presentation edit points.

Scene outline:
1. NeuralNetworkAnimation: observations -> input layer -> weights -> action.
2. CrossoverAnimation: parents -> copied interval -> reciprocal children.
"""

from typing import NamedTuple

import numpy as np
from manim import (
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    MoveAlongPath,
    Rectangle,
    Scene,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    VMobject,
    config,
)

from scripts.animation_data import crossover_example, network_example
from scripts.animation_theme import (
    COLORS,
    background,
    card,
    signed_color,
    signed_opacity,
    signed_text_color,
    terminal_header,
    text,
)


config.background_color = COLORS["background"]


# Shared network drawing helpers

def caption(content, size=22, color=None):
    """Create scene text using the shared animation font and palette."""
    return text(content, size, color or COLORS["text"])


def fit_width(mobject, width):
    """Keep a text mobject within a known horizontal layout region."""
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def network_objects(sizes, center, width=7.0, height=3.6, radius=0.09, weights=None):
    """Build indexed node layers and edges for a fully connected network.

    ``edges[layer][target][source]`` preserves the same indexing as the
    layer's weight matrix, so the scenes can highlight individual connections.
    """
    nodes = []
    edges = []
    for layer_index, size in enumerate(sizes):
        x = center[0] - width / 2 + layer_index * width / (len(sizes) - 1)
        span = min(height, (size - 1) * height / max(sizes))
        ys = np.linspace(center[1] + span / 2, center[1] - span / 2, size)
        nodes.append(VGroup(*[
            Circle(
                radius=radius,
                stroke_color=COLORS["border"],
                fill_color=COLORS["neutral"],
                fill_opacity=1,
            ).move_to([x, y, 0])
            for y in ys
        ]))
    maximum_weight = max(
        (
            float(np.max(np.abs(matrix)))
            for matrix in (weights or [])
            if matrix.size
        ),
        default=1.0,
    ) or 1.0
    for layer_index in range(len(sizes) - 1):
        layer_edges = []
        for target_index, target in enumerate(nodes[layer_index + 1]):
            row = []
            for source_index, source in enumerate(nodes[layer_index]):
                weight = (
                    float(weights[layer_index][target_index, source_index])
                    if weights is not None else 0.0
                )
                magnitude = abs(weight) / maximum_weight
                row.append(
                    Line(
                        source.get_center(),
                        target.get_center(),
                        stroke_color=signed_color(weight),
                        stroke_width=0.45 + 1.15 * magnitude,
                    ).set_opacity(0.12 + 0.6 * magnitude)
                )
            layer_edges.append(row)
        edges.append(layer_edges)
    edge_group = VGroup(*[
        edge for layer in edges for row in layer for edge in row
    ])
    node_group = VGroup(*nodes)
    return nodes, edges, VGroup(edge_group, node_group)


def _signed_node_animation(node, value):
    """Color one neuron by its signed activation."""
    return node.animate.set_fill(
        signed_color(float(value)),
        opacity=signed_opacity(float(value)),
    )


def _network_value_labels(nodes, values, size=10):
    """Place a formatted numeric activation over each corresponding node."""
    return [
        VGroup(*[
            caption(
                f"{float(value):+.2f}",
                size,
                signed_text_color(float(value)),
            ).move_to(node)
            for node, value in zip(layer, layer_values)
        ])
        for layer, layer_values in zip(nodes, values)
    ]


# Neural-network animation

class NeuralNetworkAnimation(Scene):
    """Explain one source-derived input as it passes through the live model."""

    def construct(self):
        example = network_example()
        self.add(background())
        terminal_header(
            self,
            "AI-Pong: observations to paddle actions",
            "Actual PyTorch model | seed 334 | illustrative, not trained gameplay",
            "NETWORK DEMO",
        )
        nodes, edges, diagram = network_objects(
            example.sizes,
            np.array([0.0, -0.3, 0]),
            width=7.0,
            height=3.8,
            radius=0.15,
            weights=example.weights,
        )

        headings = self._layer_headings(example, nodes)
        input_labels, output_labels = self._node_labels(example, nodes)
        self._show_input_vector(example, nodes, headings, input_labels)
        value_labels = self._reveal_network(
            example, nodes, edges, headings, output_labels
        )
        status = caption(
            "INPUTS -> tanh -> ReLU -> sigmoid | edge color = weight sign",
            17,
            COLORS["cyan"],
        ).move_to([0, -2.55, 0])
        self.play(FadeIn(status))
        self.wait(0.8)

        self._explain_focused_neuron(
            example, nodes, diagram, headings, input_labels,
            output_labels, value_labels, status,
        )
        self._animate_forward_pass(example, nodes, value_labels, status)
        self._show_action_decision(example, nodes, status)

    def _layer_headings(self, example, nodes):
        """Label the input, hidden, and output layers above the network."""
        headings = VGroup()
        for index, layer in enumerate(nodes):
            label = (
                f"Inputs ({example.sizes[index]})" if index == 0 else
                f"{example.activations[index - 1]} ({example.sizes[index]})"
            )
            headings.add(caption(label, 14, COLORS["muted"]).move_to(
                [layer[0].get_x(), 2.05, 0]
            ))
        return headings

    def _node_labels(self, example, nodes):
        """Create the observation labels and the two action labels."""
        input_labels = VGroup(*[
            fit_width(caption(label, 13, COLORS["muted"]), 1.75).next_to(
                node, [-1, 0, 0], buff=0.12
            )
            for label, node in zip(example.inputs, nodes[0])
        ])
        output_labels = VGroup(*[
            caption(label, 14).next_to(node, [1, 0, 0], buff=0.13)
            for label, node in zip(example.actions, nodes[-1])
        ])
        return input_labels, output_labels

    def _show_input_vector(self, example, nodes, headings, input_labels):
        """Show the fixed observations, then hand them off to the input nodes."""
        # Card size and row positions are the main controls for this opening layout.
        observation_card = card(10.8, 4.15).move_to([0, -0.1, 0])
        observation_title = caption("ONE FIXED INPUT VECTOR", 18, COLORS["green"]).move_to(
            [0, 1.55, 0]
        )
        input_rows = VGroup()
        for index, (label, value) in enumerate(zip(example.inputs, example.observation)):
            y = 1.1 - index * 0.43
            input_rows.add(caption(f"{index}  {label}", 15, COLORS["muted"]).move_to(
                [-2.1, y, 0]
            ))
            input_rows.add(caption(
                f"{float(value):+.4f}", 16, signed_text_color(float(value))
            ).move_to([2.55, y, 0]))
        input_note = caption(
            "These same seven values enter the seeded model; positive / negative / zero use blue / pink / neutral.",
            13,
            COLORS["muted"],
        ).move_to([0, -1.78, 0])
        self.play(FadeIn(observation_card), FadeIn(observation_title))
        for row in input_rows:
            self.play(FadeIn(row), run_time=0.12)
        self.play(FadeIn(input_note))
        # Increase this pause if viewers need longer to read the input values.
        self.wait(1.2)
        input_signals = [
            Dot(
                input_rows[index * 2 + 1].get_center(),
                radius=0.055,
                color=signed_text_color(float(value)),
            )
            for index, value in enumerate(example.observation)
        ]
        self.add(*input_signals)
        # Establish the input layer on its own before revealing any weights.
        self.play(
            FadeOut(VGroup(
                observation_card, observation_title, input_rows, input_note
            )),
            FadeIn(headings[0]),
            FadeIn(input_labels),
            LaggedStart(*[FadeIn(node) for node in nodes[0]], lag_ratio=0.12),
            *[
                signal.animate.move_to(nodes[0][index].get_center())
                for index, signal in enumerate(input_signals)
            ],
            run_time=1.2,
        )
        self.remove(*input_signals)

    def _reveal_network(self, example, nodes, edges, headings, output_labels):
        """Reveal each connection matrix before the layer it feeds."""
        value_labels = _network_value_labels(nodes, example.values, size=9)
        self.play(*[
            _signed_node_animation(node, value)
            for node, value in zip(nodes[0], example.values[0])
        ], FadeIn(value_labels[0]), run_time=0.7)

        # These run times control the pace of the layer-by-layer reveal.
        for layer_index, layer_edges in enumerate(edges):
            edge_group = VGroup(*[
                edge for row in layer_edges for edge in row
            ])
            self.play(Create(edge_group), run_time=0.65)
            layer_animations = [
                LaggedStart(
                    *[FadeIn(node) for node in nodes[layer_index + 1]],
                    lag_ratio=0.12,
                ),
                FadeIn(headings[layer_index + 1]),
            ]
            if layer_index == len(edges) - 1:
                layer_animations.append(FadeIn(output_labels))
            self.play(*layer_animations, run_time=0.6)
        return value_labels

    def _explain_focused_neuron(
        self,
        example,
        nodes,
        diagram,
        headings,
        input_labels,
        output_labels,
        value_labels,
        status,
    ):
        """Zoom in on one hidden neuron and show its weighted-sum calculation."""
        focus_index = example.focused_neuron
        focus_node = nodes[1][focus_index]
        # These card and label coordinates control the focused-neuron layout.
        focus_heading = caption(
            f"ONE ACTUAL tanh NEURON | index {focus_index}", 15, COLORS["green"]
        ).move_to([-5.05, 2.0, 0])
        math_card = card(9.35, 5.25).move_to([1.45, -0.42, 0])
        math_title = caption(
            "WEIGHTED INPUTS + BIAS -> PRE-ACTIVATION", 13, COLORS["green"]
        ).move_to([1.45, 1.77, 0])
        math_rows = VGroup()
        for index, (label, input_value, weight, contribution) in enumerate(zip(
            example.inputs,
            example.observation,
            example.weights[0][focus_index],
            example.focused_contributions,
        )):
            y = 1.27 - index * 0.38
            math_rows.add(caption(
                f"{label:<13} {input_value:+.3f} x {weight:+.4f} = {contribution:+.4f}",
                13,
                signed_text_color(float(contribution)),
            ).move_to([1.45, y, 0]))
        bias_line = caption(
            f"bias = {example.focused_bias:+.6f}", 13, COLORS["muted"]
        ).move_to([1.45, -1.58, 0])
        sum_line = caption(
            f"linear sum z = {example.focused_preactivation:+.6f}",
            14,
            COLORS["cyan"],
        ).move_to([1.45, -1.99, 0])
        tanh_line = caption(
            f"tanh(z) = {example.focused_activation:+.6f}  (actual layer output)",
            14,
            signed_color(example.focused_activation),
        ).move_to([1.45, -2.42, 0])
        precision_note = caption(
            "Products rounded to 4 d.p.; z and activation are live float32 model values.",
            12,
            COLORS["muted"],
        ).move_to([1.45, -2.83, 0])
        other_hidden_nodes = VGroup(*[
            node for index, node in enumerate(nodes[1])
            if index != focus_index
        ])

        # Isolate one neuron and reserve the right side for its numerical explanation.
        self.play(
            FadeOut(status),
            FadeOut(diagram[0]),
            FadeOut(headings),
            FadeOut(input_labels),
            FadeOut(output_labels),
            FadeOut(value_labels[0]),
            FadeOut(nodes[0]),
            FadeOut(other_hidden_nodes),
            FadeOut(nodes[2]),
            FadeOut(nodes[3]),
            FadeIn(math_card),
            FadeIn(focus_heading),
            focus_node.animate.scale(1.65).move_to([-5.05, 0.25, 0]),
            run_time=0.8,
        )
        focus_edges = VGroup(*[
            Line(
                nodes[0][index].get_center(),
                focus_node.get_center(),
                color=signed_color(float(example.weights[0][focus_index, index])),
                stroke_width=1.5,
            ).set_opacity(0.55)
            for index in range(example.sizes[0])
        ])
        self.play(Create(focus_edges), run_time=0.45)
        self.play(focus_node.animate.set_stroke(
            signed_color(example.focused_activation), width=2
        ).set_fill(
            signed_color(example.focused_activation),
            opacity=signed_opacity(example.focused_activation),
        ))
        for index, (row, contribution) in enumerate(zip(
            math_rows, example.focused_contributions
        )):
            source = nodes[0][index].get_center()
            target = nodes[1][focus_index].get_center()
            particle = Dot(
                source,
                radius=0.055,
                color=signed_text_color(float(contribution)),
            )
            self.add(particle)
            self.play(
                particle.animate.move_to(target),
                FadeIn(row),
                run_time=0.24,
            )
            self.remove(particle)
        self.play(
            FadeIn(bias_line),
            FadeIn(sum_line),
            FadeIn(tanh_line),
            FadeIn(precision_note),
            run_time=0.55,
        )
        self.wait(1.6)

        # Restore the full network before continuing the forward-pass story.
        self.play(
            FadeOut(VGroup(
                math_card, math_title, math_rows, bias_line, sum_line,
                tanh_line, precision_note, focus_heading, focus_edges,
            )),
            focus_node.animate.scale(1 / 1.65).move_to(
                nodes[1][focus_index].get_center()
            ),
            run_time=0.7,
        )
        self.play(
            FadeIn(diagram[0]),
            FadeIn(headings),
            FadeIn(input_labels),
            FadeIn(output_labels),
            FadeIn(value_labels[0]),
            FadeIn(nodes[0]),
            FadeIn(other_hidden_nodes),
            FadeIn(nodes[2]),
            FadeIn(nodes[3]),
            run_time=0.6,
        )
        self.play(FadeIn(status))

    def _animate_forward_pass(self, example, nodes, value_labels, status):
        """Animate representative signals and actual activations through layers."""
        for layer_index in range(len(example.weights)):
            source_values = example.values[layer_index]
            target_values = example.values[layer_index + 1]
            particles = []
            for target_index, source_index in enumerate(
                example.dominant_sources[layer_index]
            ):
                contribution = (
                    source_values[source_index]
                    * example.weights[layer_index][target_index, source_index]
                )
                particle = Dot(
                    nodes[layer_index][source_index].get_center(),
                    radius=0.045,
                    color=signed_color(float(contribution)),
                )
                particles.append((particle, nodes[layer_index + 1][target_index].get_center()))
                self.add(particle)
            status_text = (
                f"Layer {layer_index + 1}: {example.activations[layer_index]} outputs"
                " | one largest incoming contribution travels to each neuron"
            )
            status_text = fit_width(
                caption(status_text, 14, COLORS["cyan"]).move_to([0, -2.55, 0]),
                12.4,
            )
            self.play(
                Transform(status, status_text),
                *[
                    particle.animate.move_to(target)
                    for particle, target in particles
                ],
                run_time=0.75,
            )
            self.remove(*[particle for particle, _ in particles])
            self.play(*[
                _signed_node_animation(node, value)
                for node, value in zip(nodes[layer_index + 1], target_values)
            ], FadeIn(value_labels[layer_index + 1]), run_time=0.6)
            self.wait(0.75)

    def _show_action_decision(self, example, nodes, status):
        """Highlight the thresholded output and summarize the resulting action."""
        for index, node in enumerate(nodes[-1]):
            node.set_stroke(
                COLORS["green"] if example.decisions[index] else COLORS["border"],
                width=2.2,
            )
        legend = VGroup(
            caption("WEIGHT / ACTIVATION  + BLUE", 11, COLORS["weight_positive"]),
            caption("WEIGHT / ACTIVATION  - PINK", 11, COLORS["weight_negative"]),
            caption("ZERO  NEUTRAL", 11, COLORS["muted"]),
            caption("ACTION > 0.5  GREEN OUTLINE", 11, COLORS["green"]),
        ).arrange([1, 0, 0], buff=0.25).move_to([0, -2.95, 0])
        left, right = example.decisions
        action = "no movement" if left == right else "move left" if left else "move right"
        result = caption(
            f"Independent sigmoid > 0.5: Left={left}, Right={right} -> {action}",
            19,
            COLORS["text"],
        ).move_to([0, -2.45, 0])
        probability_note = caption(
            "Outputs are separate thresholds, not a mutually exclusive probability pair; both/neither means hold.",
            12,
            COLORS["muted"],
        ).move_to([0, -3.43, 0])
        self.play(FadeOut(status), FadeIn(legend), FadeIn(result), FadeIn(probability_note))
        self.wait(2.6)


# Crossover strip drawing helpers

class _GenomeStripLayout(NamedTuple):
    """Keep the named Manim objects needed across the crossover strip phases."""

    parent_rows: list[tuple[VGroup, VGroup, Text, str]]
    copy_rows: list[tuple[VGroup, VGroup, Text, str]]
    cut_lines: VGroup
    interval_label: Text
    parents_note: Text


def _strip_row(total, cell_width, y, color, label, label_color=None, selected=None):
    """Build an indexed genome row, leaving a gap for an optional cut interval."""
    cells = VGroup()
    indices = VGroup()
    for index in range(total):
        x = (index - (total - 1) / 2) * cell_width
        if selected is not None and selected[0] <= index < selected[1]:
            continue
        cells.add(Rectangle(
            width=cell_width * 0.96,
            height=0.38,
            stroke_width=1,
            stroke_color=color,
            fill_color=color,
            fill_opacity=0.34,
        ).move_to([x, y, 0]))
        indices.add(caption(str(index), 12, COLORS["text"]).move_to([x, y, 0]))
    row_label = caption(label, 13, label_color or color).move_to([-5.15, y, 0])
    return cells, indices, row_label


def _segment_copy(cells, indices, start, end, y, label_color):
    """Copy a half-open genome interval to use as a moving donor segment."""
    segment = VGroup(
        *[cell.copy() for cell in cells[start:end]],
        *[index.copy() for index in indices[start:end]],
    )
    segment_label = caption(
        f"[{start},{end})", 14, label_color
    ).move_to([0, y - 0.34, 0])
    return VGroup(segment, segment_label)


# Crossover animation

class CrossoverAnimation(Scene):
    """Show reciprocal parent-segment exchange and resulting parameter owners."""

    def construct(self):
        example = crossover_example()
        start, end = example.cut_points
        self.add(background())
        terminal_header(
            self,
            "Two-point crossover: exact parameter inheritance",
            f"Selected layer-output neurons [{start}, {end}) | cyan: parent A | green: parent B",
            "CROSSOVER DEMO",
        )

        total = sum(example.sizes[1:])
        # This width controls the genome strip's total span on screen.
        cell_width = min(0.47, 10.5 / total)
        strips = self._show_genome_strips_and_cuts(
            total, cell_width, start, end
        )
        self._exchange_donor_segments(total, cell_width, start, end, strips)
        child_diagrams = self._show_child_networks(example)
        self._explain_weight_ownership(example, child_diagrams)

    def _show_genome_strips_and_cuts(self, total, cell_width, start, end):
        """Introduce immutable parents, working copies, and the selected range."""
        source_rows = []
        for y, name, color in (
            (1.65, "PARENT A / original", COLORS["parent_a"]),
            (0.95, "PARENT B / original", COLORS["parent_b"]),
        ):
            cells, indices, label = _strip_row(
                total, cell_width, y, color, name
            )
            source_rows.append((cells, indices, label, color))
        self.play(*[
            FadeIn(item)
            for row in source_rows
            for item in row[:3]
        ])
        self.play(Indicate(
            VGroup(source_rows[0][0], source_rows[1][0]), color=COLORS["cyan"]
        ))
        immutable_note = caption(
            "These original parents stay intact; crossover works from copied strips below.",
            13,
            COLORS["muted"],
        ).move_to([0, 2.25, 0])
        self.play(FadeIn(immutable_note))

        copy_rows = []
        for y, name, color in (
            (0.05, "COPY A", COLORS["parent_a"]),
            (-0.65, "COPY B", COLORS["parent_b"]),
        ):
            cells, indices, label = _strip_row(total, cell_width, y, color, name)
            copy_rows.append((cells, indices, label, color))
        self.play(*[
            FadeIn(item)
            for row in copy_rows
            for item in row[:3]
        ])
        # The parent and working-copy y positions define the strip layout.
        cuts = VGroup(*[
            DashedLine(
                [
                    (point - total / 2) * cell_width,
                    -0.88,
                    0,
                ],
                [
                    (point - total / 2) * cell_width,
                    0.28,
                    0,
                ],
                color=COLORS["magenta"],
                stroke_width=2,
            )
            for point in (start, end)
        ])
        cut_label = caption(
            f"Cut boundaries {start} and {end}  |  exact half-open interval [{start},{end})",
            13,
            COLORS["magenta"],
        ).move_to([0, -1.15, 0])
        self.play(Create(cuts), FadeIn(cut_label))
        self.wait(0.8)

        return _GenomeStripLayout(
            source_rows, copy_rows, cuts, cut_label, immutable_note
        )

    def _exchange_donor_segments(
        self, total, cell_width, start, end, strips: _GenomeStripLayout
    ):
        """Move selected copied segments into reciprocal child genome rows."""
        source_rows = strips.parent_rows
        copy_rows = strips.copy_rows
        # Copy the selected material out of the working strips; source parents above never move.
        donor_segments = []
        for copy_index in (1, 0):
            cells, indices, _, color = copy_rows[copy_index]
            source_y = cells[0].get_y()
            segment = _segment_copy(
                cells, indices, start, end, source_y, color
            )
            donor_segments.append((segment, color, copy_index, source_y))
            source = VGroup(cells[start:end], indices[start:end])
            self.play(TransformFromCopy(source, segment), run_time=0.8)
            self.play(
                *[FadeOut(cells[index]) for index in range(start, end)],
                *[FadeOut(indices[index]) for index in range(start, end)],
                run_time=0.45,
            )

        child_rows = []
        for y, name, base_color, donor_color in (
            (-1.95, "CHILD A", COLORS["parent_a"], COLORS["parent_b"]),
            (-2.75, "CHILD B", COLORS["parent_b"], COLORS["parent_a"]),
        ):
            cells, indices, label = _strip_row(
                total, cell_width, y, base_color, name, selected=(start, end)
            )
            gap = Rectangle(
                width=(end - start) * cell_width * 0.96,
                height=0.38,
                stroke_color=donor_color,
                stroke_width=1.5,
                fill_opacity=0,
            ).move_to([0, y, 0])
            child_rows.append((cells, indices, label, gap, y))
        self.play(*[
            FadeIn(item)
            for row in child_rows
            for item in row[:4]
        ])
        self.wait(0.6)

        # B's segment enters Child A directly; A's segment takes a separate outside route to Child B.
        segment_b = donor_segments[0][0]
        self.play(
            segment_b.animate.shift([0, child_rows[0][4] - donor_segments[0][3], 0]),
            FadeOut(child_rows[0][3]),
            run_time=1.25,
        )
        self.play(FadeOut(segment_b[1]))
        segment_a = donor_segments[1][0]
        start_point = segment_a.get_center()
        end_point = start_point + np.array(
            [0.0, child_rows[1][4] - donor_segments[1][3], 0.0]
        )
        transfer_path = VMobject().set_points_as_corners([
            start_point,
            np.array([4.65, start_point[1], 0]),
            np.array([4.65, end_point[1], 0]),
            end_point,
        ])
        transfer_path.set_stroke(COLORS["parent_a"], width=1.4, opacity=0.65)
        transfer_label = caption(
            "A donor -> Child B", 11, COLORS["parent_a"]
        ).move_to([4.65, -1.3, 0])
        self.play(Create(transfer_path), FadeIn(transfer_label), run_time=0.4)
        self.play(
            MoveAlongPath(segment_a, transfer_path),
            FadeOut(child_rows[1][3]),
            run_time=1.8,
        )
        self.play(
            FadeOut(transfer_path),
            FadeOut(transfer_label),
            FadeOut(segment_a[1]),
        )
        note = caption(
            "Illustration reuses [3,12) for both reciprocal children; production GA calls sample cuts independently.",
            12,
            COLORS["muted"],
        ).move_to([0, -3.42, 0])
        self.play(FadeIn(note))
        self.wait(1.7)
        self.play(FadeOut(VGroup(
            strips.parents_note, strips.interval_label, strips.cut_lines, note,
            *[VGroup(*row[:3]) for row in source_rows],
            *[VGroup(*row[:3]) for row in copy_rows],
            *[VGroup(*row[:3], row[3]) for row in child_rows],
            *[segment for segment, _, _, _ in donor_segments],
        )))

    def _show_child_networks(self, example):
        """Build the two child diagrams and color each neuron's donor."""
        child_diagrams = []
        for center_x, child, selected_color, base_color in (
            (
                -3.45, "Child A", COLORS["parent_b"], COLORS["parent_a"],
            ),
            (
                3.45, "Child B", COLORS["parent_a"], COLORS["parent_b"],
            ),
        ):
            nodes, edges, diagram = network_objects(
                example.sizes,
                np.array([center_x, -0.05, 0]),
                width=5.15,
                height=3.3,
                radius=0.075,
            )
            for layer_index, layer in enumerate(edges):
                for target_index, row in enumerate(layer):
                    for source_index, edge in enumerate(row):
                        if example.neurons[layer_index][target_index]:
                            edge.set_color(selected_color).set_opacity(0.58).set_stroke(
                                width=1.05
                            )
                        else:
                            edge.set_color(base_color).set_opacity(0.28)
            for layer_index, layer in enumerate(nodes[1:]):
                for neuron_index, node in enumerate(layer):
                    donor = (
                        selected_color
                        if example.neurons[layer_index][neuron_index]
                        else base_color
                    )
                    node.set_stroke(donor).set_fill(donor, opacity=0.38)
            label = caption(child, 20).move_to([center_x, 2.15, 0])
            child_diagrams.append((nodes, edges, selected_color, base_color))
            self.play(FadeIn(diagram), FadeIn(label), run_time=0.65)
        return child_diagrams

    def _explain_weight_ownership(self, example, child_diagrams):
        """Highlight outgoing weight columns inherited from selected neurons."""
        inherit_note = caption(
            "Selected neuron: donor bias + incoming row | cyan/green = parameter owner",
            14,
            COLORS["muted"],
        ).move_to([0, -2.38, 0])
        self.play(FadeIn(inherit_note))
        self.wait(1.1)

        outgoing = []
        for nodes, edges, donor_color, _ in child_diagrams:
            for layer_index, neuron_mask in enumerate(example.neurons[:-1]):
                for target_edges in edges[layer_index + 1]:
                    for source_index, selected in enumerate(neuron_mask):
                        if selected:
                            outgoing.append(
                                target_edges[source_index].animate.set_color(
                                    donor_color
                                ).set_opacity(0.98).set_stroke(width=2.0)
                            )
        self.play(*outgoing, run_time=1.0)
        outgoing_note = caption(
            "Next: outgoing weight columns follow selected neurons into the following layer.",
            14,
            COLORS["cyan"],
        ).move_to([0, -2.87, 0])
        self.play(Transform(inherit_note, outgoing_note))
        self.wait(1.5)
        mutation_note = caption(
            "Last-layer neurons have no next column. Children are assembled before mutation.",
            13,
            COLORS["muted"],
        ).move_to([0, -3.4, 0])
        self.play(FadeIn(mutation_note))
        self.wait(2.2)
