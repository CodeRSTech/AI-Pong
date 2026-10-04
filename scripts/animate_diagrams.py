"""Manim scenes. Render both with: python -m scripts.render_animations."""

import numpy as np
from manim import (
    AnimationGroup,
    Circle,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    Indicate,
    LaggedStart,
    Line,
    Rectangle,
    Scene,
    Text,
    Transform,
    VGroup,
    config,
)

from scripts.animation_data import crossover_example, network_example
from scripts.generate_diagrams import COLORS


config.background_color = COLORS["background"]


def caption(content, size=22, color=None):
    return Text(content, font_size=size, color=color or COLORS["text"])


def fit_width(mobject, width):
    if mobject.width > width:
        mobject.scale_to_fit_width(width)
    return mobject


def network_objects(sizes, center, width=7.0, height=3.6, radius=0.09):
    """Build fully connected nodes and indexed edges without opening an Arcade window."""
    nodes = []
    edges = []
    for layer_index, size in enumerate(sizes):
        x = center[0] - width / 2 + layer_index * width / (len(sizes) - 1)
        span = min(height, (size - 1) * height / max(sizes))
        ys = np.linspace(center[1] + span / 2, center[1] - span / 2, size)
        nodes.append(VGroup(*[
            Circle(
                radius=radius,
                stroke_color=COLORS["subtle"],
                fill_color=COLORS["neutral"],
                fill_opacity=1,
            ).move_to([x, y, 0])
            for y in ys
        ]))
    for layer_index in range(len(sizes) - 1):
        layer_edges = []
        for target in nodes[layer_index + 1]:
            layer_edges.append([
                Line(
                    source.get_center(), target.get_center(),
                    stroke_color=COLORS["subtle"], stroke_width=0.8,
                ).set_opacity(0.35)
                for source in nodes[layer_index]
            ])
        edges.append(layer_edges)
    edge_group = VGroup(*[
        edge for layer in edges for row in layer for edge in row
    ])
    node_group = VGroup(*nodes)
    return nodes, edges, VGroup(edge_group, node_group)


class NeuralNetworkAnimation(Scene):
    def construct(self):
        example = network_example()
        title = caption("AI-Pong: from observations to paddle actions", 32).to_edge([0, 1, 0])
        subtitle = caption(
            "Actual PyTorch model | fixed seed 334 | illustrative game observation", 19
        ).next_to(title, [0, -1, 0])
        self.play(FadeIn(title), FadeIn(subtitle))

        nodes, edges, diagram = network_objects(
            example.sizes, np.array([0.0, -0.15, 0]), width=7.5, height=4.4, radius=0.13
        )
        headings = VGroup()
        for index, layer in enumerate(nodes):
            label = (
                f"Inputs ({example.sizes[index]})" if index == 0 else
                f"{example.activations[index - 1]} ({example.sizes[index]})"
            )
            headings.add(caption(label, 21).move_to(
                [layer[0].get_x(), 2.1, 0]
            ))
        input_labels = VGroup(*[
            fit_width(caption(label, 16), 2.35).next_to(node, [-1, 0, 0], buff=0.18)
            for label, node in zip(example.inputs, nodes[0])
        ])
        output_labels = VGroup(*[
            caption(label, 22).next_to(node, [1, 0, 0], buff=0.18)
            for label, node in zip(example.actions, nodes[-1])
        ])
        self.play(FadeIn(headings), FadeIn(input_labels), FadeIn(output_labels))
        self.play(LaggedStart(*[FadeIn(layer) for layer in nodes], lag_ratio=0.25))
        self.play(Create(diagram[0]), run_time=2)

        status = caption("One feed-forward pass through the live model", 23).to_edge([0, -1, 0])
        self.play(FadeIn(status))
        for index, (layer, values) in enumerate(zip(nodes, example.values)):
            changes = [
                node.animate.set_fill(
                    COLORS["parent_a"] if value >= 0 else COLORS["parent_b"],
                    opacity=0.25 + 0.75 * min(abs(float(value)), 1),
                )
                for node, value in zip(layer, values)
            ]
            value_labels = VGroup(*[
                caption(f"{float(value):+.2f}", 13).next_to(node, [0, -1, 0], buff=0.04)
                for node, value in zip(layer, values)
            ])
            if index:
                flow = VGroup(*[edge for row in edges[index - 1] for edge in row])
                self.play(Indicate(flow, color=COLORS["parent_a"]), run_time=0.8)
            self.play(*changes, FadeIn(value_labels), run_time=0.8)
            self.wait(0.7)

        for index, node in enumerate(nodes[-1]):
            if example.decisions[index]:
                self.play(Indicate(node, color=COLORS["cut"]))
        left, right = example.decisions
        action = "no movement" if left == right else "move left" if left else "move right"
        result = caption(
            f"Sigmoid > 0.5: Left={left}, Right={right} -> {action}", 23
        ).move_to(status)
        self.play(Transform(status, fit_width(result, 12)))
        self.wait(3)


class CrossoverAnimation(Scene):
    def construct(self):
        example = crossover_example()
        start, end = example.cut_points
        title = caption("Two-point crossover: exact parameter inheritance", 31).to_edge([0, 1, 0])
        subtitle = caption(
            f"Selected layer-output neurons [{start}, {end}) | blue: A | pink: B", 21
        ).next_to(title, [0, -1, 0])
        self.play(FadeIn(title), FadeIn(subtitle))

        total = sum(example.sizes[1:])
        cell_width = min(0.47, 10.5 / total)
        strips = []
        strip_labels = VGroup()
        for y, name, color in (
            (1.8, "Parent A", COLORS["parent_a"]),
            (0.75, "Parent B", COLORS["parent_b"]),
        ):
            cells = VGroup(*[
                Rectangle(
                    width=cell_width, height=0.5, stroke_width=1,
                    stroke_color=color, fill_color=color, fill_opacity=0.4,
                ).move_to([(index - (total - 1) / 2) * cell_width, y, 0])
                for index in range(total)
            ])
            indices = VGroup(*[
                caption(str(index), 15).move_to(cell)
                for index, cell in enumerate(cells)
            ])
            strips.append((cells, indices))
            strip_labels.add(caption(name, 20).next_to(cells, [-1, 0, 0], buff=0.2))
        self.play(FadeIn(strip_labels), *[
            FadeIn(VGroup(cells, indices)) for cells, indices in strips
        ])
        offset = 0
        layer_labels = VGroup()
        for index, size in enumerate(example.sizes[1:]):
            x = ((offset + (size - 1) / 2) - (total - 1) / 2) * cell_width
            name = "Outputs" if index == len(example.sizes) - 2 else f"Hidden {index + 1}"
            layer_labels.add(caption(name, 15).move_to([x, 2.35, 0]))
            offset += size
        self.play(FadeIn(layer_labels))
        cuts = VGroup(*[
            DashedLine(
                [(point - total / 2) * cell_width, 2.15, 0],
                [(point - total / 2) * cell_width, 0.35, 0],
                color=COLORS["cut"],
            )
            for point in (start, end)
        ])
        self.play(Create(cuts))
        self.play(*[
            Indicate(VGroup(*cells[start:end]), color=COLORS["cut"])
            for cells, _ in strips
        ])
        self.wait(1)
        self.play(*[
            cells[index].animate.set_fill(other, opacity=0.55).set_stroke(other)
            for (cells, _), other in zip(
                strips, (COLORS["parent_b"], COLORS["parent_a"])
            )
            for index in range(start, end)
        ], run_time=2)
        children_labels = VGroup(*[
            caption(name, 20).move_to(label)
            for name, label in zip(("Child A", "Child B"), strip_labels)
        ])
        self.play(Transform(strip_labels, children_labels))
        note = caption(
            "For this illustration, both children use the same cuts; the GA samples each call independently.",
            18,
        ).move_to([0, -0.1, 0])
        self.play(FadeIn(note))
        self.wait(1.5)
        self.play(FadeOut(VGroup(
            strip_labels, layer_labels, cuts, note,
            *[VGroup(cells, indices) for cells, indices in strips],
        )))

        child_diagrams = []
        for center_x, child, selected_color, base_color in (
            (-3.4, "Child A", COLORS["parent_b"], COLORS["parent_a"]),
            (3.4, "Child B", COLORS["parent_a"], COLORS["parent_b"]),
        ):
            nodes, edges, diagram = network_objects(
                example.sizes, np.array([center_x, 0.0, 0]),
                width=5.3, height=3.6,
            )
            for layer in edges:
                for row in layer:
                    for edge in row:
                        edge.set_color(base_color)
            for layer in nodes[1:]:
                for node in layer:
                    node.set_stroke(base_color).set_fill(base_color, opacity=0.5)
            label = caption(child, 24).move_to([center_x, 2.25, 0])
            child_diagrams.append((nodes, edges, selected_color))
            self.play(FadeIn(diagram), FadeIn(label))
        status = caption(
            "Node color = bias source. Edge color = weight source. Inputs are not inherited parameters.", 18
        ).move_to([0, -2.35, 0])
        self.play(FadeIn(status))

        animations = []
        for nodes, edges, donor_color in child_diagrams:
            for layer_index, layer_mask in enumerate(example.biases):
                for index, selected in enumerate(layer_mask):
                    if selected:
                        animations.append(
                            nodes[layer_index + 1][index].animate.set_fill(
                                donor_color, opacity=0.7
                            ).set_stroke(donor_color)
                        )
            for layer_index, layer_mask in enumerate(example.weights):
                for target_index, row in enumerate(layer_mask):
                    for source_index, selected in enumerate(row):
                        if selected:
                            animations.append(
                                edges[layer_index][target_index][source_index]
                                .animate.set_color(donor_color).set_opacity(0.7)
                            )
        self.play(AnimationGroup(*animations), run_time=2)
        explanation = caption(
            "Selected neurons take incoming weights + biases; their outgoing weights follow the same donor.", 19
        ).move_to([0, -2.95, 0])
        self.play(FadeIn(explanation))
        self.wait(3)
        mutation_note = caption(
            "Crossover shown before mutation. The example is checked against real child parameters.", 18
        ).move_to([0, -3.5, 0])
        self.play(FadeIn(mutation_note))
        self.wait(2)
