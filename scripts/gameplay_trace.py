"""Capture and validate offline, checkpoint-driven gameplay traces."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.ga.player import ACTION_LABELS, OBSERVATION_LABELS
from src.game import Game
from src.tester import load_player
from src.variables import VARIABLES


TRACE_SCHEMA = "ai-pong-gameplay-trace-v1"
MAX_TRACE_STEPS = 1000


def _finite(value, description: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must contain only finite numeric values.")


def _state(zone, scores: dict) -> dict:
    ball = zone.ball
    ai = zone.ai_paddle
    cpu = zone.cpu_paddle
    result = {
        "ball": {
            "x": ball.pos_x, "y": ball.pos_y,
            "vx": ball.speed.x, "vy": ball.speed.y,
            "width": ball.width, "height": ball.height,
        },
        "ai_paddle": {
            "x": ai.pos_x, "y": ai.pos_y,
            "width": ai.width, "height": ai.height,
        },
        "cpu_paddle": {
            "x": cpu.pos_x, "y": cpu.pos_y,
            "width": cpu.width, "height": cpu.height,
        },
        "scores": {key: int(value) for key, value in scores.items()},
    }
    for entity, values in result.items():
        if entity == "scores":
            continue
        for key, value in values.items():
            _finite(value, f"{entity}.{key}")
    return result


def _observations(state: dict, width: float, height: float) -> np.ndarray:
    ball = state["ball"]
    paddle = state["ai_paddle"]
    speed = math.hypot(ball["vx"], ball["vy"])
    if speed == 0:
        raise ValueError("A gameplay trace cannot record a zero-speed ball observation.")
    return np.asarray([
        (ball["x"] - paddle["x"]) / width,
        (ball["y"] - paddle["y"]) / height,
        (paddle["x"] * 2 - width) / width,
        (ball["x"] * 2 - width) / width,
        (ball["y"] * 2 - height) / height,
        ball["vx"] / speed,
        ball["vy"] / speed,
    ], dtype=np.float32)


def _activation_name(module) -> str:
    names = {"Tanh": "tanh", "ReLU": "ReLU", "Sigmoid": "sigmoid"}
    try:
        return names[type(module).__name__]
    except KeyError as error:
        raise ValueError(f"Unsupported checkpoint activation: {type(module).__name__}") from error


def _network_parameters(player) -> tuple[list[int], list[str], list[dict]]:
    layers = player.neural_net.layers
    if not layers:
        raise ValueError("Checkpoint contains no network layers.")
    sizes = [layers[0][0].in_features]
    activations = []
    parameters = []
    for layer in layers:
        linear = layer[0]
        if linear.in_features != sizes[-1]:
            raise ValueError("Checkpoint layers have incompatible input/output dimensions.")
        sizes.append(linear.out_features)
        activations.append(_activation_name(layer[1]))
        weights = linear.weight.detach().cpu().numpy()
        biases = linear.bias.detach().cpu().numpy()
        if weights.shape != (linear.out_features, linear.in_features):
            raise ValueError("Checkpoint weight matrix dimensions do not match its layer.")
        if biases.shape != (linear.out_features,):
            raise ValueError("Checkpoint bias vector dimensions do not match its layer.")
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError("Checkpoint parameters contain nonfinite values.")
        parameters.append({
            "weights": weights.tolist(),
            "biases": biases.tolist(),
        })
    if sizes[0] != len(OBSERVATION_LABELS) or sizes[-1] != len(ACTION_LABELS):
        raise ValueError(
            "Checkpoint input/output dimensions do not match the current observation/action labels."
        )
    return sizes, activations, parameters


def validate_trace(trace: dict) -> dict:
    """Reject malformed, mismatched, or nonfinite traces before scene rendering."""
    if not isinstance(trace, dict) or trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"Trace schema must be {TRACE_SCHEMA!r}.")
    metadata = trace.get("metadata")
    model = trace.get("model")
    decisions = trace.get("decisions")
    if not isinstance(metadata, dict) or not isinstance(model, dict):
        raise ValueError("Trace metadata and model must be objects.")
    if not isinstance(decisions, list) or not 1 <= len(decisions) <= MAX_TRACE_STEPS:
        raise ValueError(f"Trace must contain 1 to {MAX_TRACE_STEPS} decisions.")
    sizes = model.get("layer_sizes")
    parameters = model.get("parameters")
    activations = model.get("activations")
    if (
        model.get("observation_labels") != list(OBSERVATION_LABELS)
        or model.get("action_labels") != list(ACTION_LABELS)
    ):
        raise ValueError("Trace model labels do not match current observation/action labels.")
    if (
        not isinstance(sizes, list)
        or len(sizes) < 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in sizes
        )
        or not isinstance(parameters, list)
        or len(parameters) != len(sizes) - 1
        or not isinstance(activations, list)
        or len(activations) != len(parameters)
        or any(name not in ("tanh", "ReLU", "sigmoid") for name in activations)
        or activations[-1] != "sigmoid"
    ):
        raise ValueError("Trace model architecture or parameters are malformed.")
    if sizes[0] != len(OBSERVATION_LABELS) or sizes[-1] != len(ACTION_LABELS):
        raise ValueError("Trace model dimensions do not match current observation/action labels.")

    for index, (layer, source_size, target_size) in enumerate(
        zip(parameters, sizes, sizes[1:])
    ):
        try:
            weights = np.asarray(layer["weights"], dtype=float)
            biases = np.asarray(layer["biases"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Trace parameters for layer {index} are malformed.") from error
        if weights.shape != (target_size, source_size) or biases.shape != (target_size,):
            raise ValueError(f"Trace parameter dimensions are invalid for layer {index}.")
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError(f"Trace parameters for layer {index} are nonfinite.")

    court = metadata.get("court")
    if not isinstance(court, dict):
        raise ValueError("Trace court settings are missing.")
    width = court.get("width")
    height = court.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("Trace court dimensions are missing.")
    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ValueError("Trace court dimensions must be positive.")
    timing = metadata.get("timing")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(timing, dict) or not isinstance(checkpoint, dict):
        raise ValueError("Trace timing and checkpoint provenance are missing.")
    try:
        speed = timing["speed"]
        step_delta = metadata["step_delta_seconds"]
    except KeyError as error:
        raise ValueError("Trace timing metadata is incomplete.") from error
    _finite(speed, "Trace game speed")
    _finite(step_delta, "Trace step delta")
    if speed <= 0 or step_delta <= 0:
        raise ValueError("Trace speed and step delta must be positive.")
    if not isinstance(checkpoint.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint["sha256"]
    ):
        raise ValueError("Trace checkpoint SHA-256 is malformed.")

    previous_post_state = None
    previous_time = None
    for index, record in enumerate(decisions):
        if not isinstance(record, dict):
            raise ValueError(f"Trace decision {index} must be an object.")
        if record.get("index") != index:
            raise ValueError("Trace decisions must have consecutive zero-based indices.")
        try:
            observation = np.asarray(record["observations"], dtype=float)
            output_actions = record["actions"]
            layer_values = record["activations"]
            pre = record["pre_state"]
            post = record["post_state"]
            observed_from_state = _observations(pre, width, height)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError(f"Trace decision {index} is malformed.") from error
        if observation.shape != (sizes[0],) or not np.isfinite(observation).all():
            if observation.shape != (sizes[0],):
                raise ValueError(f"Trace observations have invalid dimensions at decision {index}.")
            raise ValueError(f"Trace observations are nonfinite at decision {index}.")
        if not np.allclose(observation, observed_from_state, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Trace observations disagree with state at decision {index}.")
        if not isinstance(layer_values, list) or len(layer_values) != len(sizes):
            raise ValueError(f"Trace activations are invalid at decision {index}.")
        try:
            activation_arrays = [
                np.asarray(values, dtype=float) for values in layer_values
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Trace activations are invalid at decision {index}.") from error
        if any(
            values.shape != (size,)
            for values, size in zip(activation_arrays, sizes)
        ):
            raise ValueError(f"Trace activation dimensions are invalid at decision {index}.")
        if any(not np.isfinite(values).all() for values in activation_arrays):
            raise ValueError(f"Trace activations are nonfinite at decision {index}.")
        if not np.allclose(activation_arrays[0], observation, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Trace input activation disagrees at decision {index}.")
        computed = observation
        for layer_index, (layer, activation_name) in enumerate(
            zip(parameters, activations)
        ):
            weights = np.asarray(layer["weights"], dtype=float)
            biases = np.asarray(layer["biases"], dtype=float)
            computed = computed @ weights.T + biases
            if activation_name == "tanh":
                computed = np.tanh(computed)
            elif activation_name == "ReLU":
                computed = np.maximum(computed, 0)
            else:
                computed = 1 / (1 + np.exp(-np.clip(computed, -709, 709)))
            if not np.allclose(
                computed, activation_arrays[layer_index + 1], rtol=1e-5, atol=1e-6
            ):
                raise ValueError(
                    f"Trace activations disagree with checkpoint parameters at "
                    f"decision {index}, layer {layer_index}."
                )
        if (
            not isinstance(output_actions, list)
            or len(output_actions) != len(ACTION_LABELS)
            or any(type(action) is not bool for action in output_actions)
        ):
            raise ValueError(f"Trace actions are invalid at decision {index}.")
        if output_actions != (activation_arrays[-1] > 0.5).tolist():
            raise ValueError(f"Trace action does not match sigmoid outputs at decision {index}.")
        required_fields = {
            "ball": {"x", "y", "vx", "vy", "width", "height"},
            "ai_paddle": {"x", "y", "width", "height"},
            "cpu_paddle": {"x", "y", "width", "height"},
        }
        for state_name, state in (("pre_state", pre), ("post_state", post)):
            if not isinstance(state, dict):
                raise ValueError(f"{state_name} must be an object at decision {index}.")
            for entity_name in ("ball", "ai_paddle", "cpu_paddle"):
                entity = state.get(entity_name)
                if not isinstance(entity, dict):
                    raise ValueError(f"{state_name}.{entity_name} is missing at decision {index}.")
                if not required_fields[entity_name].issubset(entity):
                    raise ValueError(
                        f"{state_name}.{entity_name} is incomplete at decision {index}."
                    )
                for field in required_fields[entity_name]:
                    _finite(
                        entity[field],
                        f"decision {index} {state_name}.{entity_name}.{field}",
                    )
            scores = state.get("scores")
            if not isinstance(scores, dict) or not {
                "Player", "CPU", "Player Hits", "CPU Hits"
            }.issubset(scores):
                raise ValueError(f"{state_name}.scores is incomplete at decision {index}.")
            for score_name, value in scores.items():
                _finite(value, f"decision {index} {state_name}.scores.{score_name}")
        try:
            time_before = float(record["simulation_time_before"])
            time_after = float(record["simulation_time_after"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Trace simulation times are invalid at decision {index}.") from error
        if (
            not math.isfinite(time_before)
            or not math.isfinite(time_after)
            or time_after <= time_before
            or (previous_time is not None and time_before != previous_time)
        ):
            raise ValueError(f"Trace simulation times are invalid at decision {index}.")
        if previous_post_state is not None and pre != previous_post_state:
            raise ValueError(f"Trace state continuity breaks before decision {index}.")
        events = record.get("events")
        required_events = {
            "wall_bounce",
            "player_paddle_hit",
            "cpu_paddle_hit",
            "score_changed",
            "ai_paddle_displacement",
            "cpu_paddle_displacement",
        }
        if not isinstance(events, dict) or not required_events.issubset(events):
            raise ValueError(f"Trace events are incomplete at decision {index}.")
        for event_name in (
            "wall_bounce", "player_paddle_hit", "cpu_paddle_hit", "score_changed"
        ):
            if type(events[event_name]) is not bool:
                raise ValueError(f"Trace event {event_name} is invalid at decision {index}.")
        ai_displacement = events["ai_paddle_displacement"]
        cpu_displacement = events["cpu_paddle_displacement"]
        _finite(ai_displacement, f"decision {index} AI displacement")
        _finite(cpu_displacement, f"decision {index} CPU displacement")
        if not math.isclose(
            ai_displacement, post["ai_paddle"]["x"] - pre["ai_paddle"]["x"], abs_tol=1e-6
        ) or not math.isclose(
            cpu_displacement,
            post["cpu_paddle"]["x"] - pre["cpu_paddle"]["x"],
            abs_tol=1e-6,
        ):
            raise ValueError(f"Trace paddle displacement disagrees with states at decision {index}.")
        if events["player_paddle_hit"] != (
            post["scores"]["Player Hits"] > pre["scores"]["Player Hits"]
        ) or events["cpu_paddle_hit"] != (
            post["scores"]["CPU Hits"] > pre["scores"]["CPU Hits"]
        ) or events["score_changed"] != (
            post["scores"]["Player"] != pre["scores"]["Player"]
            or post["scores"]["CPU"] != pre["scores"]["CPU"]
        ):
            raise ValueError(f"Trace events disagree with scores at decision {index}.")
        previous_post_state = post
        previous_time = time_after
    return trace


def load_trace(path: Path) -> dict:
    """Load a trace file and run its structural and numerical checks."""
    try:
        trace = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read gameplay trace '{path}': {error}") from error
    return validate_trace(trace)


@contextmanager
def _isolated_rngs(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _project_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def capture_trace(
    checkpoint: Path,
    *,
    seed: int,
    steps: int,
    run_id: str = "unknown",
    generation: int | str = "unknown",
    validation: dict | str = "unknown",
    step_delta: float | None = None,
) -> dict:
    """Capture real Game.step transitions from one explicitly selected checkpoint."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Scenario seed must be an integer between 0 and 2**32 - 1.")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= MAX_TRACE_STEPS:
        raise ValueError(f"Step count must be between 1 and {MAX_TRACE_STEPS}.")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint '{checkpoint}' does not exist.")

    fps = float(VARIABLES["FPS"])
    speed = float(VARIABLES["SPEED"])
    steps_per_frame = int(VARIABLES["STEPS_PER_FRAME"])
    delta = (
        1.0 / (fps * speed * steps_per_frame)
        if step_delta is None else float(step_delta)
    )
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Step delta must be a finite positive value.")

    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    with tempfile.TemporaryDirectory(prefix="ai-pong-trace-checkpoint-") as temp_dir:
        snapshot = Path(temp_dir) / "checkpoint.pt"
        snapshot.write_bytes(checkpoint_bytes)
        with _isolated_rngs(seed):
            player = load_player(snapshot)
            sizes, activations, parameters = _network_parameters(player)
            game = Game(
                [player],
                width=VARIABLES["WIDTH"],
                height=VARIABLES["HEIGHT"],
                fps=fps,
                timeout=-1,
                speed=speed,
                steps_per_frame=steps_per_frame,
                paddle_width=80,
            )
            game._ensure_brain()
            zone = game.zones[0]
            records = []
            for index in range(steps):
                pre_state = _state(zone, player.scores)
                observations = player.look(zone).astype(np.float32)
                single_action = player.think(observations)[0].astype(bool)
                cached_activations = [
                    np.asarray(values[0], dtype=np.float64).copy()
                    for values in player.neural_net.last_activations
                ]
                batched_inputs = observations.reshape(1, -1)
                batched_action = game.batched_brain.predict_batch(batched_inputs)[0].astype(bool)
                if not np.array_equal(single_action, batched_action):
                    raise RuntimeError(
                        f"Single and batched model actions differ at decision {index}."
                    )
                if not np.all(np.isfinite(observations)) or any(
                    not np.isfinite(values).all() for values in cached_activations
                ):
                    raise ValueError(f"Nonfinite inference data at decision {index}.")
                ball_before = pre_state["ball"]
                score_before = dict(player.scores)
                game.step(delta)
                post_state = _state(zone, player.scores)
                ball_after = post_state["ball"]
                records.append({
                    "index": index,
                    "simulation_time_before": index * delta,
                    "simulation_time_after": (index + 1) * delta,
                    "pre_state": pre_state,
                    "observations": observations.astype(float).tolist(),
                    "activations": [values.astype(float).tolist() for values in cached_activations],
                    "actions": batched_action.tolist(),
                    "post_state": post_state,
                    "events": {
                        "wall_bounce": (
                            (
                                ball_before["x"] + ball_before["width"] / 2 > game.play_width
                                or ball_before["x"] - ball_before["width"] / 2 < 0
                            )
                            and ball_before["vx"] * ball_after["vx"] < 0
                        ),
                        "player_paddle_hit": (
                            post_state["scores"]["Player Hits"]
                            > score_before["Player Hits"]
                        ),
                        "cpu_paddle_hit": (
                            post_state["scores"]["CPU Hits"]
                            > score_before["CPU Hits"]
                        ),
                        "score_changed": (
                            post_state["scores"]["Player"] != score_before["Player"]
                            or post_state["scores"]["CPU"] != score_before["CPU"]
                        ),
                        "ai_paddle_displacement": (
                            post_state["ai_paddle"]["x"] - pre_state["ai_paddle"]["x"]
                        ),
                        "cpu_paddle_displacement": (
                            post_state["cpu_paddle"]["x"] - pre_state["cpu_paddle"]["x"]
                        ),
                    },
                })

    trace = {
        "schema": TRACE_SCHEMA,
        "metadata": {
            "project": "AI-Pong",
            "project_version": _project_version(),
            "checkpoint": {
                "sha256": checkpoint_sha256,
                "source_name": checkpoint.name,
            },
            "scenario_seed": seed,
            "requested_steps": steps,
            "step_delta_seconds": delta,
            "court": {
                "width": VARIABLES["WIDTH"],
                "height": VARIABLES["HEIGHT"],
                "coordinate_origin": "top-left",
                "positive_y_direction": "down",
            },
            "timing": {
                "fps": fps,
                "speed": speed,
                "steps_per_frame": steps_per_frame,
                "timeout": -1,
                "paddle_width": 80,
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "provenance": {
                "run_id": run_id,
                "generation": generation,
                "validation": validation,
            },
        },
        "model": {
            "layer_sizes": sizes,
            "activations": activations,
            "observation_labels": list(OBSERVATION_LABELS),
            "action_labels": list(ACTION_LABELS),
            "parameters": parameters,
        },
        "decisions": records,
    }
    return validate_trace(trace)


def write_trace(trace: dict, destination: Path) -> Path:
    validate_trace(trace)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(trace, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
