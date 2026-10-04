"""Three modular Manim explanations driven by the same validated game trace."""

import os
from pathlib import Path

import numpy as np
from manim import (
    Arrow,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    Transform,
    VGroup,
)

from scripts.animation_theme import (
    COLORS,
    background,
    callout,
    card,
    signed_color,
    signed_opacity,
    signed_text_color,
    terminal_header,
    text,
)
from scripts.gameplay_trace import load_trace


FORMULAS = (
    "dx/W=(ball_x-paddle_x)/W",
    "dy/H=(ball_y-paddle_y)/H",
    "(2*paddle_x-W)/W",
    "(2*ball_x-W)/W",
    "(2*ball_y-H)/H",
    "vx/|v|",
    "vy/|v|",
)


def _trace_and_index() -> tuple[dict, int]:
    path = os.environ.get("AI_PONG_TRACE_PATH")
    if not path:
        raise ValueError("Set AI_PONG_TRACE_PATH to a validated gameplay trace.")
    trace = load_trace(Path(path))
    try:
        index = int(os.environ.get("AI_PONG_DECISION_INDEX", "0"))
    except ValueError as error:
        raise ValueError("AI_PONG_DECISION_INDEX must be an integer.") from error
    if not 0 <= index < len(trace["decisions"]):
        raise ValueError(
            f"Decision index {index} is outside this {len(trace['decisions'])}-decision trace."
        )
    return trace, index


def _court_position(entity: dict, trace: dict, center, court_width, court_height):
    width = trace["metadata"]["court"]["width"]
    height = trace["metadata"]["court"]["height"]
    x = center[0] - court_width / 2 + entity["x"] / width * court_width
    y = center[1] + court_height / 2 - entity["y"] / height * court_height
    return np.array([x, y, 0])


def _court(trace: dict, state: dict, center, court_width, court_height):
    board = Rectangle(
        width=court_width,
        height=court_height,
        stroke_color=COLORS["border"],
        stroke_width=1.2,
        fill_color=COLORS["surface"],
        fill_opacity=0.95,
    ).move_to(center)
    top_left = [
        center[0] - court_width / 2,
        center[1] + court_height / 2,
        0,
    ]
    middle = Line(
        [center[0], center[1] - court_height / 2, 0],
        [center[0], center[1] + court_height / 2, 0],
        color=COLORS["border"],
        stroke_width=1,
        stroke_opacity=0.75,
    )
    ai, cpu, ball = state["ai_paddle"], state["cpu_paddle"], state["ball"]

    def paddle(entity, color):
        position = _court_position(entity, trace, center, court_width, court_height)
        return RoundedRectangle(
            width=max(0.16, entity["width"] / trace["metadata"]["court"]["width"] * court_width),
            height=max(0.07, entity["height"] / trace["metadata"]["court"]["height"] * court_height),
            corner_radius=0.04,
            stroke_width=0,
            fill_color=color,
            fill_opacity=0.95,
        ).move_to(position)

    ball_position = _court_position(ball, trace, center, court_width, court_height)
    ball_dot = Dot(
        ball_position,
        radius=0.07,
        color=COLORS["text"],
    )
    objects = VGroup(
        board,
        middle,
        paddle(cpu, COLORS["green"]),
        paddle(ai, COLORS["cyan"]),
        ball_dot,
    )
    cpu_label = text("CPU", 10, COLORS["green"]).next_to(
        _court_position(cpu, trace, center, court_width, court_height), [1, 0, 0], buff=0.1
    )
    ai_label = text("AI", 10, COLORS["cyan"]).next_to(
        _court_position(ai, trace, center, court_width, court_height), [1, 0, 0], buff=0.1
    )
    origin = text("(0,0)", 9, COLORS["muted"]).move_to(
        [top_left[0] + 0.35, top_left[1] - 0.18, 0]
    )
    return objects, cpu_label, ai_label, origin


def _scene_background(scene):
    scene.add(background())


class GameplayObservationsScene(Scene):
    """Explain the real traced court state and its seven model observations."""

    def construct(self):
        trace, index = _trace_and_index()
        record = trace["decisions"][index]
        _scene_background(self)
        terminal_header(
            self,
            "Gameplay state -> seven observations",
            "Recorded pre-step state | simulation coordinates: origin top-left, y increases downward",
            "MODULE 01 / 03",
        )
        court_center = np.array([-3.45, -0.15, 0])
        court_width, court_height = 4.65, 4.05
        court, cpu_label, ai_label, origin = _court(
            trace, record["pre_state"], court_center, court_width, court_height
        )
        label = text(f"RECORDED STEP {index:04d}", 12, COLORS["cyan"]).move_to(
            [-3.45, 2.13, 0]
        )
        self.play(FadeIn(court), FadeIn(cpu_label), FadeIn(ai_label), FadeIn(origin), FadeIn(label))

        ball = record["pre_state"]["ball"]
        paddle = record["pre_state"]["ai_paddle"]
        ball_position = _court_position(
            ball, trace, court_center, court_width, court_height
        )
        ai_position = _court_position(
            paddle, trace, court_center, court_width, court_height
        )
        self.play(Create(DashedLine(
            ball_position, ai_position, color=COLORS["muted"], stroke_width=1
        )))
        velocity_scale = 0.28
        velocity_vector = Arrow(
            ball_position,
            ball_position + np.array([ball["vx"], -ball["vy"], 0]) * velocity_scale,
            buff=0,
            color=COLORS["cyan"],
            stroke_width=2,
            max_tip_length_to_length_ratio=0.24,
        )
        self.play(Create(velocity_vector))
        self.play(FadeIn(text(
            f"ball ({ball['x']:.1f}, {ball['y']:.1f}) | AI x={paddle['x']:.1f}",
            10,
            COLORS["text"],
        ).move_to([-3.45, -2.35, 0])))

        panel = card(7.35, 4.45).move_to([3.05, -0.12, 0])
        heading = text("OBSERVATION FORMULAS", 13, COLORS["green"]).move_to(
            [3.05, 1.78, 0]
        )
        self.play(FadeIn(panel), FadeIn(heading))
        values = record["observations"]
        rows = VGroup()
        for row_index, (formula, value, label_text) in enumerate(
            zip(FORMULAS, values, trace["model"]["observation_labels"])
        ):
            y = 1.32 - row_index * 0.43
            rows.add(text(
                f"{row_index} {label_text}: {formula} = {value:+.4f}",
                10,
            ).move_to(
                [3.05, y, 0]
            ))
        for row in rows:
            self.play(FadeIn(row), run_time=0.2)
            self.wait(0.25)
        vector = text(
            "INPUT ORDER  [ " + "  ".join(f"{value:+.3f}" for value in values) + " ]",
            13,
            COLORS["cyan"],
        ).move_to([0, -3.12, 0])
        vector_card = card(12.1, 0.58).move_to([0, -3.12, 0])
        self.play(FadeIn(vector_card), FadeIn(vector))
        self.wait(2)


class ObservationNetworkActionsScene(Scene):
    """Display the traced activations, signed checkpoint weights, and actual actions."""

    def construct(self):
        trace, index = _trace_and_index()
        record = trace["decisions"][index]
        model = trace["model"]
        _scene_background(self)
        terminal_header(
            self,
            "Observation -> network -> action",
            f"Same trace decision {index:04d} | values and signed weights are from this checkpoint",
            "MODULE 02 / 03",
        )

        sizes = model["layer_sizes"]
        weights = [np.asarray(layer["weights"], dtype=float) for layer in model["parameters"]]
        values = [np.asarray(layer, dtype=float) for layer in record["activations"]]
        center_x, center_y = 0.0, 0.15
        network_width, network_height = 9.4, 3.4
        nodes = []
        positions = []
        for layer_index, size in enumerate(sizes):
            x = center_x - network_width / 2 + layer_index * network_width / (len(sizes) - 1)
            span = min(network_height, (size - 1) * network_height / max(sizes))
            ys = np.linspace(center_y + span / 2, center_y - span / 2, size)
            positions.append([(x, y) for y in ys])
            nodes.append(VGroup(*[
                Circle(
                    radius=0.14,
                    stroke_color=COLORS["border"],
                    stroke_width=1,
                    fill_color=signed_color(float(value)),
                    fill_opacity=signed_opacity(float(value)),
                ).move_to([x, y, 0])
                for y, value in zip(ys, values[layer_index])
            ]))

        maximum_weight = max(
            (float(np.max(np.abs(layer))) for layer in weights),
            default=1.0,
        ) or 1.0
        edges = VGroup()
        for layer_index, matrix in enumerate(weights):
            for target_index, row in enumerate(matrix):
                for source_index, weight in enumerate(row):
                    magnitude = abs(float(weight)) / maximum_weight
                    edge = Line(
                        [*positions[layer_index][source_index], 0],
                        [*positions[layer_index + 1][target_index], 0],
                        color=signed_color(float(weight)),
                        stroke_width=0.35 + 1.3 * magnitude,
                    ).set_opacity(0.16 + 0.65 * magnitude)
                    edges.add(edge)
        self.play(Create(edges), run_time=1.4)
        self.play(*[FadeIn(layer) for layer in nodes], run_time=0.45)

        layer_titles = VGroup()
        numeric_values = VGroup()
        for layer_index, (size, positions_for_layer, layer_values) in enumerate(
            zip(sizes, positions, values)
        ):
            name = (
                "Inputs" if layer_index == 0 else
                "Actions" if layer_index == len(sizes) - 1 else
                f"Layer {layer_index}"
            )
            descriptor = (
                f"{name} ({size})" if layer_index == 0 or layer_index == len(sizes) - 1
                else f"{name} / {model['activations'][layer_index - 1]} ({size})"
            )
            layer_titles.add(text(descriptor, 11, COLORS["muted"]).move_to(
                [positions_for_layer[0][0], 2.17, 0]
            ))
            for node, (x, y), value in zip(nodes[layer_index], positions_for_layer, layer_values):
                numeric_values.add(text(
                    f"{float(value):+.2f}", 8, signed_text_color(float(value))
                ).move_to([x, y, 0]))
                if layer_index == len(sizes) - 1:
                    node.set_stroke(
                        COLORS["green"] if float(value) > 0.5 else COLORS["border"],
                        width=2,
                    )
        self.play(FadeIn(layer_titles), FadeIn(numeric_values))

        legend = VGroup(
            text("WEIGHT / ACTIVATION + BLUE", 9, COLORS["weight_positive"]),
            text("WEIGHT / ACTIVATION - PINK", 9, COLORS["weight_negative"]),
            text("ZERO NEUTRAL", 9, COLORS["muted"]),
            text("ACTION > 0.5 GREEN OUTLINE", 9, COLORS["green"]),
        ).arrange([1, 0, 0], buff=0.24).move_to([0, -1.95, 0])
        self.play(FadeIn(legend))
        outputs = record["activations"][-1]
        left, right = record["actions"]
        action = "no movement" if left == right else "move left" if left else "move right"
        output_line = callout(
            f"SIGMOID > 0.5  Left={outputs[0]:.3f} -> {left}   "
            f"Right={outputs[1]:.3f} -> {right}   |   {action}",
            15,
        ).move_to([0, -2.48, 0])
        self.play(FadeIn(output_line))
        rules = text(
            "RULE REFERENCE   Left/Right: 00 -> hold  |  10 -> left  |  "
            "01 -> right  |  11 -> hold",
            11,
            COLORS["muted"],
        ).move_to([0, -3.12, 0])
        self.play(FadeIn(rules))
        self.wait(2)


class ActionsExecutionScene(Scene):
    """Replay the selected actual step and nearby recorded transitions."""

    def construct(self):
        trace, index = _trace_and_index()
        decisions = trace["decisions"]
        record = decisions[index]
        _scene_background(self)
        terminal_header(
            self,
            "Action -> paddle movement -> next state",
            f"Actual Game.step replay | decision {index:04d} onward | visual pauses do not add physics steps",
            "MODULE 03 / 03",
        )
        court_center = np.array([-3.35, -0.2, 0])
        court_width, court_height = 5.6, 4.8
        court, cpu_label, ai_label, origin = _court(
            trace, record["pre_state"], court_center, court_width, court_height
        )
        header = text(f"PRE-STEP {index:04d}", 12, COLORS["cyan"]).move_to(
            [-3.35, 2.33, 0]
        )
        self.play(
            FadeIn(court), FadeIn(cpu_label), FadeIn(ai_label), FadeIn(origin), FadeIn(header)
        )

        detail_card = card(5.65, 4.7).move_to([3.55, -0.15, 0])
        self.play(FadeIn(detail_card))
        speed = trace["metadata"]["timing"]["speed"]
        paddle = record["pre_state"]["ai_paddle"]
        half = paddle["width"] / 2
        width = trace["metadata"]["court"]["width"]
        displacement = record["events"]["ai_paddle_displacement"]
        details = [
            ("MODEL OUTPUT", f"Left={record['actions'][0]}  Right={record['actions'][1]}"),
            ("OBSERVED MOVE", f"AI center x: {paddle['x']:.2f} -> {record['post_state']['ai_paddle']['x']:.2f}"),
            ("DISPLACEMENT", f"{displacement:+.2f} px (measured from states)"),
            ("MOVEMENT RULE", f"max(2, speed) = {max(2.0, speed):.2f} px / step"),
            ("LEGAL CENTER", f"[{half:.1f}, {width - half:.1f}] px"),
            ("CPU", f"center x {record['pre_state']['cpu_paddle']['x']:.2f} -> {record['post_state']['cpu_paddle']['x']:.2f}"),
        ]
        text_items = VGroup()
        for row_index, (heading, body) in enumerate(details):
            y = 1.54 - row_index * 0.56
            text_items.add(text(heading, 10, COLORS["green"]).move_to([3.55, y + 0.11, 0]))
            text_items.add(text(body, 11).move_to([3.55, y - 0.12, 0]))
        self.play(FadeIn(text_items))
        self.wait(1)

        objects = list(court)
        cpu_paddle, ai_paddle, ball = objects[2], objects[3], objects[4]
        timeline = []
        for current_index in range(index, min(index + 4, len(decisions))):
            current = decisions[current_index]
            prior_state = current["pre_state"]
            next_state = current["post_state"]
            if current_index > index:
                self.play(
                    ai_paddle.animate.move_to(_court_position(
                        prior_state["ai_paddle"], trace, court_center, court_width, court_height
                    )),
                    cpu_paddle.animate.move_to(_court_position(
                        prior_state["cpu_paddle"], trace, court_center, court_width, court_height
                    )),
                    ai_label.animate.move_to(_court_position(
                        prior_state["ai_paddle"], trace, court_center, court_width, court_height
                    ) + np.array([0.35, 0, 0])),
                    cpu_label.animate.move_to(_court_position(
                        prior_state["cpu_paddle"], trace, court_center, court_width, court_height
                    ) + np.array([0.35, 0, 0])),
                    ball.animate.move_to(_court_position(
                        prior_state["ball"], trace, court_center, court_width, court_height
                    )),
                    run_time=0.45,
                )
            displacement = current["events"]["ai_paddle_displacement"]
            left, right = current["actions"]
            action = "NO MOTION" if left == right or displacement == 0 else (
                "LEFT" if displacement < 0 else "RIGHT"
            )
            events = current["events"]
            event_names = [
                label for key, label in (
                    ("player_paddle_hit", "AI HIT"),
                    ("cpu_paddle_hit", "CPU HIT"),
                    ("wall_bounce", "WALL"),
                    ("score_changed", "SCORE"),
                ) if events[key]
            ]
            if current_index == index:
                status_line = text(
                    f"STEP {current_index:04d}   {action}   "
                    f"ball v=({prior_state['ball']['vx']:.2f}, {prior_state['ball']['vy']:.2f})",
                    11,
                    COLORS["cyan"],
                ).move_to([-3.35, -2.92, 0])
                timeline.append(status_line)
                self.play(FadeIn(status_line))
            else:
                status_line = text(
                    f"STEP {current_index:04d}   {action}   "
                    f"AI dx={displacement:+.1f}   {' / '.join(event_names) or 'no recorded event'}",
                    10,
                    COLORS["muted"],
                ).move_to([3.55, -2.03 - 0.27 * (current_index - index), 0])
                timeline.append(status_line)
                self.play(FadeIn(status_line), run_time=0.35)
            ai_after = _court_position(
                next_state["ai_paddle"], trace, court_center, court_width, court_height
            )
            cpu_after = _court_position(
                next_state["cpu_paddle"], trace, court_center, court_width, court_height
            )
            ball_after = _court_position(
                next_state["ball"], trace, court_center, court_width, court_height
            )
            self.play(
                ai_paddle.animate.move_to(ai_after),
                cpu_paddle.animate.move_to(cpu_after),
                ai_label.animate.move_to(ai_after + np.array([0.35, 0, 0])),
                cpu_label.animate.move_to(cpu_after + np.array([0.35, 0, 0])),
                ball.animate.move_to(ball_after),
                run_time=0.55,
            )
        self.wait(1.5)
