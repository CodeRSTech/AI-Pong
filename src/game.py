# src/game.py
"""
Game orchestration using Arcade.

Creates a window, runs one timed epoch, and returns control to the ga.
"""

import arcade
import random
from components.playzone import PlayZone
from utils.functions import timeit
from variables import VARIABLES
from components.colors import OFF_WHITE, GRAY, LIGHT_GRAY, BLUE


class _PongWindow(arcade.Window):
    """
    Arcade window that runs a single epoch of the game.
    """
    

    def __init__(self, game_ref: "Game"):
        self.game = game_ref
        update_rate = 1.0 / max(1, int(self.game.fps * self.game.speed))
        super().__init__(
            width=self.game.play_width + self.game.panel_width,
            height=self.game.height,
            title="Pong 2D (Arcade)",
            update_rate=update_rate
        )
        self.set_update_rate(update_rate)
        arcade.set_background_color(OFF_WHITE)
        self.time_running = 0.0  # seconds
        self._exiting = False
        self._nn_cache_key = None
        self._nn_layers_positions = None
        # Cache for Text objects to avoid per-frame draw_text calls
        self._text_cache = {}

    def on_draw(self):
        # Clear the backbuffer for this frame
        self.clear()

        # Render a single zone at full size (avoid overlapping zones)
        window_h = self.height
        # COMMENTED OUT CODE RENDERS ONLY ONE PLAYZONE
        '''zone = self.game.get_display_zone()
        if zone is not None:
            zone.render_to(window_h)'''
        # Draw ALL zones overlapping in the same play area
        for zone in self.game.zones:
            zone.render_to(window_h)

        # Optional score overlay for tester (single zone)
        if self.game.num_zones == 1 and self.game.display_score:
            scores = self.game.players[0].scores
            score_text = f"Player:{scores['Player']}  CPU:{scores['CPU']}"
            arcade.draw_text(
                score_text,
                self.game.play_width // 2,
                self.height // 2,
                GRAY,
                18,
                anchor_x="center",
                anchor_y="center",
            )

        self._draw_network_panel()

    def _draw_network_panel(self) -> None:
        panel_left = self.game.play_width
        panel_right = self.game.play_width + self.game.panel_width
        label_margin = 40  # more spacing so labels don't collide with nodes

        # 1. Background
        arcade.draw_lrbt_rectangle_filled(panel_left, panel_right, 0, self.height, LIGHT_GRAY)
        arcade.draw_line(panel_left, 0, panel_left, self.height, GRAY, 2)

        # 2. Title & Legend (create Text once and reuse to avoid warning)
        #    Also draw a color swatch for the displayed player's paddle.
        title_y = self.height - 30
        dz = self.game.get_display_zone()
        swatch_color = getattr(dz.ai_paddle, "color", BLUE) if dz else BLUE
        swatch_w, swatch_h = 28, 16
        swatch_left = panel_left + 16
        swatch_right = swatch_left + swatch_w
        swatch_bottom = title_y - swatch_h / 2
        swatch_top = title_y + swatch_h / 2
        arcade.draw_lrbt_rectangle_filled(swatch_left, swatch_right, swatch_bottom, swatch_top, swatch_color)

        title_x = swatch_right + 8

        if "nn_title" not in self._text_cache:
            self._text_cache["nn_title"] = arcade.Text(
                "Elite NN Analysis", title_x, title_y, GRAY, 16, bold=True
            )
            self._text_cache["legend_1"] = arcade.Text(
                "Blue: Positive Weight | Red: Negative Weight", panel_left + 16, self.height - 60, GRAY, 10
            )
            self._text_cache["legend_2"] = arcade.Text(
                "Line Thickness: Magnitude", panel_left + 16, self.height - 75, GRAY, 10
            )
            self._text_cache["legend_3"] = arcade.Text(
                "Inputs: Δx = ball_x - paddle_x (normalized), |v_ball| = speed, pad_x/ball_x/ball_y ∈ [0,1]",
                panel_left + 16, self.height - 95, GRAY, 10
            )
        else:
            self._text_cache["nn_title"].x = title_x
            self._text_cache["nn_title"].y = title_y

        self._text_cache["nn_title"].draw()
        self._text_cache["legend_1"].draw()
        self._text_cache["legend_2"].draw()
        self._text_cache["legend_3"].draw()

        architecture = self.game.get_display_architecture()
        if not architecture:
            # Robust fallback so the panel still renders even if the net isn't ready yet
            architecture = [5, 6, 4, 2]

        # 3. Dynamic Spacing Logic (wider margins so text doesn't overlap diagram)
        cache_key = (tuple(architecture), self.height, self.game.panel_width)
        needs_rebuild = (
                self._nn_layers_positions is None
                or len(self._nn_layers_positions) != len(architecture)
                or self._nn_cache_key != cache_key
        )
        if needs_rebuild:
            self._nn_cache_key = cache_key
            margin_x = 140
            margin_y = 120
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
            INPUT_LABELS = ["Ball dist x", "Ball dist y", "Paddle pos x", "Ball pos x", "Ball pos y"]
            OUTPUT_LABELS = ["Left", "Right"]
            self._text_cache["in_labels"] = []
            self._text_cache["out_labels"] = []
            for i, (x, y) in enumerate(self._nn_layers_positions[0]):
                if i < len(INPUT_LABELS):
                    self._text_cache["in_labels"].append(
                        arcade.Text(INPUT_LABELS[i], x - label_margin, y, GRAY, 11,
                                    anchor_x="right", anchor_y="center")
                    )
            for i, (x, y) in enumerate(self._nn_layers_positions[-1]):
                if i < len(OUTPUT_LABELS):
                    self._text_cache["out_labels"].append(
                        arcade.Text(OUTPUT_LABELS[i], x + label_margin, y, GRAY, 11,
                                    anchor_x="left", anchor_y="center", bold=True)
                    )

            arch_text = " → ".join(str(n) for n in architecture)
            self._text_cache["arch"] = arcade.Text(arch_text, panel_left + 16, 20, GRAY, 12)

        layers_positions = self._nn_layers_positions
        player = self.game.get_display_player()
        net = getattr(player, "neural_net", None)

        # 4. Draw Weights (Connections) — skip if net is missing or has no layers
        if net and getattr(net, "layers", None) and len(net.layers) > 0:
            weights = []
            max_abs_w = 1e-8
            for seq in net.layers:
                linear = seq[0]
                W = linear.weight.detach().cpu().numpy()
                weights.append(W)
                max_abs_w = max(max_abs_w, float(abs(W).max()))

            POS_COLOR = BLUE
            NEG_COLOR = (200, 50, 50)  # Red
            MIN_THICK, MAX_THICK = 0.5, 4.0

            for layer_i, W in enumerate(weights):
                src_nodes = layers_positions[layer_i]
                dst_nodes = layers_positions[layer_i + 1]
                for dst_j, (x2, y2) in enumerate(dst_nodes):
                    for src_i, (x1, y1) in enumerate(src_nodes):
                        w = float(W[dst_j, src_i])
                        t = abs(w) / max_abs_w
                        thickness = MIN_THICK + t * (MAX_THICK - MIN_THICK)
                        color = POS_COLOR if w >= 0 else NEG_COLOR
                        arcade.draw_line(x1, y1, x2, y2, color, thickness)

        # 5. Draw Neurons (with Activations) — always render nodes based on positions
        acts = getattr(net, "last_activations", None) if net else None
        RADIUS = 20
        active_out_idx = None
        if net and getattr(net, "last_output_binary", None) is not None:
            try:
                lob = net.last_output_binary
                if lob.ndim == 2 and lob.shape[0] >= 1:
                    active_out_idx = int(lob[0].argmax())
            except Exception:
                active_out_idx = None

        ACTIVE_COLOR = (50, 200, 50)
        INACTIVE_COLOR = (120, 120, 160)

        for layer_idx, nodes in enumerate(layers_positions):
            layer_act = acts[layer_idx][0] if (acts is not None and layer_idx < len(acts)) else None
            for j, (x, y) in enumerate(nodes):
                if layer_act is not None and j < len(layer_act):
                    val = float(layer_act[j])
                    intensity = int(100 + 155 * min(1.0, abs(val)))
                    node_color = (*BLUE[:3], intensity) if val >= 0 else (200, 50, 50, intensity)
                else:
                    node_color = BLUE

                if layer_idx == len(layers_positions) - 1:
                    if active_out_idx is not None and j == active_out_idx:
                        arcade.draw_circle_filled(x, y, RADIUS + 4, (*ACTIVE_COLOR[:3], 80))
                        arcade.draw_circle_filled(x, y, RADIUS, ACTIVE_COLOR)
                        continue
                    else:
                        node_color = INACTIVE_COLOR

                arcade.draw_circle_filled(x, y, RADIUS, node_color)

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
                    self._text_cache["out_labels"][i].color = GRAY
                self._text_cache["out_labels"][i].draw()

        # 7. Architecture summary
        self._text_cache["arch"].draw()

    def on_update(self, delta_time: float):
        if self._exiting:
            return

        self.time_running += delta_time

        for zone in self.game.zones:
            zone.update()

        if self.game.timeout != -1 and self.time_running >= self.game.timeout:
            self._exiting = True
            arcade.exit()   # <-- important: end the event loop cleanly
            # DO NOT call self.close() here


class Game:
    """
    Manages zones and runs one Arcade epoch, returning the updated players list.
    """
    

    def __init__(self, players, width=VARIABLES['WIDTH'], height=VARIABLES['HEIGHT'],
                 fps=VARIABLES['FPS'], timeout=VARIABLES['TIME_OUT'],
                 speed=VARIABLES['SPEED']):
        
        self.display_score = False
        self.fps = fps
        self.speed = speed
        self.timeout = timeout
        self.players = players
        self.play_width = width
        self.panel_width = VARIABLES.get('PANEL_WIDTH', 260)
        self.height = height
        self.num_zones = len(players)
        self.zones = []
        self._display_player = None
        self._display_zone = None

        # Build zones (one per player for now; drawn in same space like original)
        best_score = players[0].scores['fitness']
        for i in range(self.num_zones):
            zone = PlayZone(width, height, speed, players[i], best_score)
            self.zones.append(zone)

        self._window = None

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
        Run the Arcade window/epoch and return updated players.
        """
        self._window = _PongWindow(self)
        arcade.run()  # blocks until window exits

        # After run loop ends, ensure window resources are released.
        try:
            if self._window:
                self._window.close()
        finally:
            self._window = None

        return self.players