# src/game.py
"""
Game orchestration using Arcade.

Creates a window, runs one timed epoch, and returns control to the ga.
"""

import math
import random
from math import ceil
from time import perf_counter
from typing import Callable

import arcade
import numpy as np

from src.audio import SoundEffects
from src.components.colors import (
    ARENA_BACKGROUND,
    ARENA_LINE,
    GRAY,
    PANEL_BACKGROUND,
    PANEL_BORDER,
    PLAYER_ACCENT,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WEIGHT_NEGATIVE,
    WEIGHT_POSITIVE,
)
from src.components.playzone import PlayZone
from src.ga.network import BatchedPopulationBrain
from src.utils import logger
from src.variables import VARIABLES


class _PongWindow(arcade.Window):
    """
    Arcade window that can either run a standalone epoch or host a training lifecycle.
    """

    network_node_radius = 16
    network_panel_margin_x = 140
    network_panel_margin_y = 80
    network_fallback_architecture = [7, 8, 6, 2]
    network_input_labels = ["Ball dist x", "Ball dist y",
                            "Paddle pos x",
                            "Ball pos x", "Ball pos y",
                            "Ball speed x", "Ball speed y"]
    network_output_labels = ["Left", "Right"]

    def __init__(self, game_ref: "Game"):
        self.game = game_ref
        update_rate = 1.0 / (self.game.fps * self.game.speed)
        super().__init__(
            width=self.game.play_width + self.game.panel_width,
            height=self.game.height,
            title="Pong 2D (Arcade)",
            update_rate=update_rate
        )
        if self.game.sound_enabled:
            self.game._sound_effects = SoundEffects()
            # Lock the visible zone before stepping so event routing never consumes RNG.
            self.game.get_display_zone()
        self.set_update_rate(update_rate)
        arcade.set_background_color(ARENA_BACKGROUND)
        self.time_running = 0.0  # seconds
        self._exiting = False
        self._nn_cache_key = None
        self._nn_layers_positions = None
        # Cache for Text objects to avoid per-frame draw_text calls
        self._text_cache = {}

    def on_close(self):
        try:
            if self.game.training_close_callback is not None:
                self.game.training_close_callback()
        finally:
            super().on_close()

    def on_draw(self):
        # Clear the back buffer for this frame
        self.clear()

        # To keep the UI Neural Net panel working, do one manual un-batched
        # pass just to populate the cached state of the displayed player.
        display_player = self.game.get_display_player()
        display_zone = self.game.get_display_zone()
        if display_player and display_zone:
            display_player.think(display_player.look(display_zone))

        window_h = self.height

        self._draw_arena()

        # Render only the elite zone; the rest of the population remains headless.
        if display_zone is not None:
            display_zone.render_to(window_h)

        if self.game.training_update_callback is not None:
            self._draw_training_status()

        status = self.game.validation_status
        if status and status.get("reached"):
            success_color = (40, 220, 110)
            left, right = 3, self.game.play_width - 3
            bottom, top = 3, self.height - 3
            arcade.draw_line(left, bottom, right, bottom, success_color, 4)
            arcade.draw_line(left, top, right, top, success_color, 4)
            arcade.draw_line(left, bottom, left, top, success_color, 4)
            arcade.draw_line(right, bottom, right, top, success_color, 4)
            arcade.draw_text(
                f"VALIDATION REACHED (gen {status['generation']}) - "
                f"wins {status['win_rate']:.0%}, shutouts {status['shutout_rate']:.0%}",
                self.game.play_width // 2,
                self.height - 56,
                success_color,
                13,
                anchor_x="center",
                anchor_y="center",
                bold=True,
            )

        # Optional score overlay for tester (single zone)
        if self.game.num_zones == 1 and self.game.display_score:
            scores = self.game.players[0].scores
            score_text = f"Player:{scores['Player']}  CPU:{scores['CPU']}"
            arcade.draw_text(
                score_text,
                self.game.play_width // 2,
                self.height // 2,
                TEXT_PRIMARY,
                18,
                anchor_x="center",
                anchor_y="center",
            )

        self._draw_network_panel()

    def _draw_training_status(self) -> None:
        """Keep lifecycle transitions legible while the same window stays active."""
        game = self.game
        phase = f"Generation {game.generation + 1} - {game.training_phase}"
        arcade.draw_text(
            phase,
            game.play_width // 2,
            self.height - 20,
            TEXT_PRIMARY,
            14,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        if game.validation_count:
            wins, losses, ties = game.validation_record
            detail = (
                f"Scenario {game.validation_scenario}/{game.validation_count}  "
                f"Score {game.training_scores.get('Player', 0)}-"
                f"{game.training_scores.get('CPU', 0)}  "
                f"W/L/T {wins}/{losses}/{ties}"
            )
        else:
            detail = (
                f"Player {game.training_scores.get('Player', 0)}  "
                f"CPU {game.training_scores.get('CPU', 0)}"
            )
        arcade.draw_text(
            detail,
            game.play_width // 2,
            self.height - 42,
            TEXT_PRIMARY,
            12,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_arena(self) -> None:
        """Draw the render-only court markings behind the active play zone."""
        width = self.game.play_width
        height = self.height
        arcade.draw_lrbt_rectangle_filled(
            0, width, 0, height, ARENA_BACKGROUND
        )
        arcade.draw_lrbt_rectangle_outline(
            4, width - 4, 4, height - 4, ARENA_LINE, 2
        )
        center_x = width / 2
        arcade.draw_circle_outline(
            center_x, height / 2, min(width, height) * 0.12, ARENA_LINE, 2
        )
        # Use short dashes instead of a solid divider to match the court's center mark.
        for y in range(12, height, 24):
            arcade.draw_line(
                center_x, y, center_x, min(y + 12, height), ARENA_LINE, 2
            )

    def on_update(self, delta_time: float):
        if self._exiting:
            return

        game = self.game
        if game.training_update_callback is not None:
            game.training_update_callback(delta_time)
            return

        steps_per_frame = game.steps_per_frame
        step_delta = delta_time / steps_per_frame
        game.run_steps(steps_per_frame, step_delta)
        self.time_running = game.time_running

        if game.is_finished:
            self._exiting = True
            arcade.exit()

    def _draw_network_panel(self) -> None:
        panel_left = self.game.play_width
        panel_right = self.game.play_width + self.game.panel_width
        label_margin = 40  # more spacing so labels don't collide with nodes

        # 1. Background
        # --------------------------------------------------------------------------------------------------------------
        arcade.draw_lrbt_rectangle_filled(
            panel_left, panel_right, 0, self.height, PANEL_BACKGROUND
        )
        arcade.draw_line(panel_left, 0, panel_left, self.height, PANEL_BORDER, 2)

        # 2. Title & Legend (create Text once and reuse to avoid warning)
        #    Also draw a color swatch for the displayed player's paddle.
        # --------------------------------------------------------------------------------------------------------------
        title_y = self.height - 30
        dz = self.game.get_display_zone()
        swatch_color = getattr(dz.ai_paddle, "color", PLAYER_ACCENT) if dz else PLAYER_ACCENT
        swatch_w, swatch_h = 28, 16
        swatch_left = panel_left + 16
        swatch_right = swatch_left + swatch_w
        swatch_bottom = title_y - swatch_h / 2
        swatch_top = title_y + swatch_h / 2
        arcade.draw_lrbt_rectangle_filled(swatch_left, swatch_right, swatch_bottom, swatch_top, swatch_color)

        title_x = swatch_right + 8

        if "nn_title" not in self._text_cache:
            self._text_cache["nn_title"] = arcade.Text(
                "Elite Neural Network Architecture", title_x, title_y,
                TEXT_PRIMARY, 16, bold=True
            )
            self._text_cache["legend_1"] = arcade.Text(
                "Blue: positive | Coral: negative | Thickness: magnitude",
                panel_left + 16, self.height - 60, TEXT_MUTED, 10
            )
        else:
            self._text_cache["nn_title"].x = title_x
            self._text_cache["nn_title"].y = title_y

        self._text_cache["nn_title"].draw()
        self._text_cache["legend_1"].draw()

        architecture = self.game.get_display_architecture()
        if not architecture:
            # Robust fallback so the panel still renders even if the net isn't ready yet
            architecture = self.network_fallback_architecture

        # 3. Dynamic Spacing Logic (wider margins so text doesn't overlap diagram)
        # --------------------------------------------------------------------------------------------------------------
        cache_key = (tuple(architecture), self.height, self.game.panel_width)
        needs_rebuild = (
                self._nn_layers_positions is None
                or len(self._nn_layers_positions) != len(architecture)
                or self._nn_cache_key != cache_key
        )
        if needs_rebuild:
            self._nn_cache_key = cache_key
            margin_x = self.network_panel_margin_x
            margin_y = self.network_panel_margin_y
            usable_w = self.game.panel_width - (2 * margin_x)
            usable_h = self.height - (2 * margin_y)
            layer_gap = usable_w / max(1, len(architecture) - 1)

            self._nn_layers_positions = []
            for idx, nodes in enumerate(architecture):
                x = panel_left + margin_x + idx * layer_gap
                step = usable_h / (nodes - 1) if nodes > 1 else 0
                ys = [margin_y + j * step for j in range(nodes)] if nodes > 1 else [margin_y + usable_h / 2]
                self._nn_layers_positions.append([(x, y) for y in ys])

            # Rebuild static label Text objects for inputs/outputs (one-time per rebuild)
            input_labels = self.network_input_labels
            output_labels = self.network_output_labels
            self._text_cache["in_labels"] = []
            self._text_cache["out_labels"] = []
            for i, (x, y) in enumerate(self._nn_layers_positions[0]):
                if i < len(input_labels):
                    self._text_cache["in_labels"].append(
                        arcade.Text(input_labels[i], x - label_margin, y, TEXT_MUTED, 11,
                                    anchor_x="right", anchor_y="center")
                    )
            for i, (x, y) in enumerate(self._nn_layers_positions[-1]):
                if i < len(output_labels):
                    self._text_cache["out_labels"].append(
                        arcade.Text(output_labels[i], x + label_margin, y, TEXT_MUTED, 11,
                                    anchor_x="left", anchor_y="center", bold=True)
                    )

            arch_text = " → ".join(str(n) for n in architecture)
            self._text_cache["arch"] = arcade.Text(
                arch_text, panel_left + 16, 20, TEXT_MUTED, 12
            )

        layers_positions = self._nn_layers_positions
        player = self.game.get_display_player()
        net = getattr(player, "neural_net", None)

        # 4. Draw Weights (Connections) — skip if net is missing or has no layers
        # --------------------------------------------------------------------------------------------------------------
        if net and getattr(net, "layers", None) and len(net.layers) > 0:
            weights = []
            max_abs_w = 1e-8
            for seq in net.layers:
                linear = seq[0]
                linear_weights = linear.weight.detach().cpu().numpy()
                weights.append(linear_weights)
                max_abs_w = max(max_abs_w, float(abs(linear_weights).max()))

            positive_color = WEIGHT_POSITIVE
            negative_color = WEIGHT_NEGATIVE
            min_weight_thickness, max_weight_thickness = 0.5, 4.0

            for layer_i, linear_weights in enumerate(weights):
                source_nodes = layers_positions[layer_i]
                destination_nodes = layers_positions[layer_i + 1]
                for dst_j, (x2, y2) in enumerate(destination_nodes):
                    for src_i, (x1, y1) in enumerate(source_nodes):
                        w = float(linear_weights[dst_j, src_i])
                        t = abs(w) / max_abs_w
                        thickness = min_weight_thickness + t * (max_weight_thickness - min_weight_thickness)
                        color = positive_color if w >= 0 else negative_color
                        arcade.draw_line(x1, y1, x2, y2, color, thickness)

        # 5. Draw Neurons (with Activations) — always render nodes based on positions
        # --------------------------------------------------------------------------------------------------------------
        acts = getattr(net, "last_activations", None) if net else None
        network_node_radius = self.network_node_radius
        active_out_idx = None
        if net and getattr(net, "last_output_binary", None) is not None:
            try:
                lob = net.last_output_binary
                if lob.ndim == 2 and lob.shape[0] >= 1:
                    active_out_idx = int(lob[0].argmax())
            except Exception:
                logger.opt(exception=True).error("ERROR:\n"
                                                 "Unknown error while reading `last_output_binary` of Neural Network.\n"
                                                 "setting `active_out_idx` to `None`")
                active_out_idx = None

        active_color = PLAYER_ACCENT
        inactive_color = (74, 91, 112)

        for layer_idx, nodes in enumerate(layers_positions):
            layer_act = acts[layer_idx][0] if (acts is not None and layer_idx < len(acts)) else None
            for j, (x, y) in enumerate(nodes):
                if layer_act is not None and j < len(layer_act):
                    val = float(layer_act[j])
                    intensity = int(100 + 155 * min(1.0, abs(val)))
                    node_color = (
                        (*WEIGHT_POSITIVE, intensity)
                        if val >= 0
                        else (*WEIGHT_NEGATIVE, intensity)
                    )
                else:
                    node_color = TEXT_MUTED

                if layer_idx == len(layers_positions) - 1:
                    if active_out_idx is not None and j == active_out_idx:
                        # Draw a larger circle with alpha=80, draw the smaller circle on top of that
                        arcade.draw_circle_filled(x, y, network_node_radius + 4, (*active_color[:3], 80))
                        arcade.draw_circle_filled(x, y, network_node_radius, active_color)
                        continue
                    else:
                        node_color = inactive_color

                arcade.draw_circle_filled(x, y, network_node_radius, node_color)

        # 6. Labels
        for t in self._text_cache.get("in_labels", []):
            t.draw()

        flash_on = (int(self.time_running * 5) % 2) == 0
        for i, (x, y) in enumerate(layers_positions[-1]):
            if i < len(self._text_cache.get("out_labels", [])):
                is_active = (active_out_idx == i)
                if is_active:
                    bg_col = (255, 235, 100) if flash_on else (255, 180, 60)
                    txt_col = (20, 20, 20) if flash_on else (0, 0, 0)
                    pad_w, pad_h = 60, 24
                    left = x + label_margin
                    right = left + pad_w
                    bottom = y - pad_h / 2
                    top = y + pad_h / 2
                    arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, bg_col)
                    self._text_cache["out_labels"][i].color = txt_col
                else:
                    self._text_cache["out_labels"][i].color = TEXT_MUTED
                self._text_cache["out_labels"][i].draw()

        # 7. Architecture summary
        self._text_cache["arch"].draw()


class Game:
    """
    Manages zones and runs one Arcade epoch, returning the updated players list.
    """
    

    def __init__(self, players, width=VARIABLES['WIDTH'], height=VARIABLES['HEIGHT'],
                 fps=VARIABLES['FPS'], timeout=VARIABLES['TIME_OUT'],
                 speed=VARIABLES['SPEED'], steps_per_frame=VARIABLES['STEPS_PER_FRAME'],
                 paddle_width=80, validation_status=None, sound_enabled=False,
                 generation=0, training_phase="Gameplay"):

        if not players:
            raise ValueError("Game requires at least one player.")
        if (
            not math.isfinite(fps)
            or not math.isfinite(speed)
            or fps <= 0
            or speed <= 0
            or not math.isfinite(fps * speed)
            or steps_per_frame < 1
            or not math.isfinite(fps * speed * steps_per_frame)
        ):
            raise ValueError("fps, speed, and steps_per_frame must be positive.")
        if not math.isfinite(timeout) or (timeout <= 0 and timeout != -1):
            raise ValueError("timeout must be positive, or -1 for an unbounded game.")
        
        self.batched_brain = None
        self.time_running = 0.0
        self.display_score = False
        self.fps = fps
        self.speed = speed
        self.steps_per_frame = steps_per_frame
        self.paddle_width = paddle_width
        self.timeout = timeout
        self.validation_status = validation_status
        self.sound_enabled = sound_enabled
        self._sound_effects = None
        self.players = players
        self.play_width = width
        self.panel_width = VARIABLES.get('PANEL_WIDTH', 260)
        self.height = height
        self.num_zones = len(players)
        self.zones = []
        self._display_player = None
        self._display_zone = None
        self._window = None
        self._batch_inputs = np.empty((self.num_zones, 7), dtype=np.float32)
        self.training_phase = training_phase
        self.generation = generation
        self.validation_scenario = 0
        self.validation_count = 0
        self.validation_record = (0, 0, 0)
        self.training_scores = {}
        self.training_update_callback = None
        self.training_close_callback = None
        self.configure_epoch(players, timeout=timeout, paddle_width=paddle_width)

    def configure_epoch(
        self,
        players,
        *,
        timeout=None,
        paddle_width=None,
        phase=None,
        generation=None,
        validation_scenario=0,
        validation_count=0,
    ) -> None:
        """Replace a simulation epoch without transferring or recreating its Arcade window."""
        if not players:
            raise ValueError("Game requires at least one player.")
        next_timeout = self.timeout if timeout is None else timeout
        if not math.isfinite(next_timeout) or (next_timeout <= 0 and next_timeout != -1):
            raise ValueError("timeout must be positive, or -1 for an unbounded game.")

        self.players = players
        self.timeout = next_timeout
        self.paddle_width = self.paddle_width if paddle_width is None else paddle_width
        self.num_zones = len(players)
        self.batched_brain = None
        self.time_running = 0.0
        self._display_player = None
        self._display_zone = None
        self.zones = []
        if phase is not None:
            self.training_phase = phase
        if generation is not None:
            self.generation = generation
        self.validation_scenario = validation_scenario
        self.validation_count = validation_count

        best_score = players[0].scores['fitness']
        for i, player in enumerate(players):
            self.zones.append(
                PlayZone(
                    self.play_width,
                    self.height,
                    self.speed,
                    player,
                    best_score,
                    paddle_width=self.paddle_width,
                    sound_event_callback=(
                        lambda event, zone_index=i: self._dispatch_sound_event(
                            self.zones[zone_index], event
                        )
                        if self.sound_enabled
                        else None
                    ),
                )
            )
        self._batch_inputs = np.empty((self.num_zones, 7), dtype=np.float32)
        self.training_scores = {
            "Player": players[0].scores.get("Player", 0),
            "CPU": players[0].scores.get("CPU", 0),
        }

    def _dispatch_sound_event(self, zone: PlayZone, event: str) -> None:
        """Only the visible zone may play audio; sound never affects headless runs."""
        if (
            self.sound_enabled
            and self._sound_effects is not None
            and zone is self._display_zone
        ):
            self._sound_effects.play(event)

    def step(self, step_delta: float) -> None:
        """
        Advance every zone by one simulation step. Independent of any window.
        """
        if not math.isfinite(step_delta) or step_delta <= 0:
            raise ValueError("step_delta must be a finite positive value.")
        self._ensure_brain()
        self.time_running += step_delta

        for index, zone in enumerate(self.zones):
            zone.ai_player.look_into(zone, self._batch_inputs[index])
        outputs = self.batched_brain.predict_batch(self._batch_inputs)

        for i, zone in enumerate(self.zones):
            zone.ai_player.apply_move(zone, bool(outputs[i][0]), bool(outputs[i][1]))
            zone.update()

    def run_steps(self, count: int, step_delta: float) -> None:
        """
        Run up to `count` steps, stopping early once the timeout is reached.
        """
        for _ in range(count):
            self.step(step_delta)
            if self.is_finished:
                return

    @property
    def is_finished(self) -> bool:
        return self.timeout != -1 and self.time_running >= self.timeout

    def run_headless(
        self,
        step_delta: float = 1.0 / 60,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> list:
        """
        Run the whole epoch without a window; optionally report step progress.
        """
        if self.timeout == -1:
            raise ValueError("run_headless requires a finite timeout")
        if not math.isfinite(step_delta) or step_delta <= 0:
            raise ValueError("step_delta must be a finite positive value.")
        total_steps = max(1, ceil(self.timeout / step_delta))
        progress_interval = max(1, total_steps // 10)
        next_progress_step = progress_interval
        started_at = perf_counter()
        completed_steps = 0
        while not self.is_finished:
            self.step(step_delta)
            completed_steps += 1
            if progress_callback and (
                completed_steps >= next_progress_step or self.is_finished
            ):
                progress_callback(completed_steps, total_steps, perf_counter() - started_at)
                while next_progress_step <= completed_steps:
                    next_progress_step += progress_interval
        return self.players

    def _ensure_brain(self) -> None:
        if self.batched_brain is None:
            self.batched_brain = BatchedPopulationBrain(self.players)

    def get_display_player(self):
        if self._display_player is not None:
            return self._display_player
        if not self.players:
            return None

        fitness_values = [p.scores.get('fitness', 0) for p in self.players]
        if len(set(fitness_values)) == 1:
            self._display_player = random.choice(self.players)
        else:
            self._display_player = max(self.players, key=lambda p: p.scores.get('fitness', 0))
        return self._display_player

    def get_display_zone(self):
        if self._display_zone is not None:
            return self._display_zone

        player = self.get_display_player()
        if player is None:
            return None
        for zone in self.zones:
            if zone.ai_player is player:
                self._display_zone = zone
                return zone

        self._display_zone = self.zones[0] if self.zones else None
        return self._display_zone

    def get_display_architecture(self):
        player = self.get_display_player()
        if player is None:
            # Safe fallback so UI can still render something
            return [5, 6, 4, 2]

        net = getattr(player, "neural_net", None)
        if net is None or not hasattr(net, "layers"):
            return [5, 6, 4, 2]

        if len(net.layers) == 0:
            # Net not built yet (lazy or failed init) — draw default scaffold
            return [5, 6, 4, 2]

        first_linear = net.layers[0][0]
        architecture = [first_linear.in_features]
        for layer in net.layers:
            linear = layer[0]
            architecture.append(linear.out_features)
        return architecture

    def start(self) -> list:
        """
        Run the standalone Arcade epoch or the attached continuous training lifecycle.
        """
        # The Game owns the single Arcade window; training reconfigures its simulation
        # between epochs instead of closing the window and starting another event loop.
        self.batched_brain = BatchedPopulationBrain(self.players)
        self.time_running = 0.0

        self._window = _PongWindow(self)
        try:
            arcade.run()
        finally:
            try:
                if self._sound_effects is not None:
                    try:
                        self._sound_effects.close()
                    finally:
                        self._sound_effects = None
            finally:
                try:
                    if self._window is not None:
                        self._window.close()
                finally:
                    self._window = None

        return self.players