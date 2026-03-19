# src/playzone.py
"""
PlayZone: a self-contained arena that owns a ball, two paddles and an AI player.

This module keeps simulation in y-down space (origin top-left). Rendering
converts to y-up in draw calls of each entity.
"""

from random import randint

from utils.functions import skew_ball_direction, create_paddle, create_ball, timeit


class PlayZone:
    """
    The Play-zone container.

    Contains unique instances of ball, paddles and player.

    Updates by letting the ball, cpu paddle and player paddle make their moves
    and handling collisions. Updates the scores.
    """

    def __init__(self, width, height, speed, ai_player, best_ai_fitness=0):
        """
        Args:
            width (int): Width of the play zone.
            height (int): Height of the play zone.
            speed (int): Speed scalar for movements.
            ai_player (IndividualPlayer): AI player object.
            best_ai_fitness (int): Best fitness of the AI player (for display/tuning).
        """
        self.time_running = 0
        self.WIDTH = width
        self.HEIGHT = height
        self.speed = speed
        self.ai_player = ai_player
        self.best_ai_fitness = best_ai_fitness
        self.ball = create_ball()
        self.ai_paddle = create_paddle(width, height, self.ball.color)
        self.cpu_paddle = create_paddle(width, height, self.ball.color, is_cpu=True)

        self.respawn_ball(self.ball)

    def update(self) -> None:
        """
        Advance simulation one step: AI, CPU, collisions, and state sync.
        """
        self.player_move()
        self.cpu_move()
        self.check_collisions()
        self.update_variables()

    def cpu_move(self) -> None:
        """
        Move the CPU paddle to track the ball when ball is moving upwards.
        """
        cpu_paddle = self.cpu_paddle
        ball = self.ball

        # Only react if ball is heading towards the CPU (top in y-down space)
        if ball.speed.y < 0:
            distance = ball.pos_x - cpu_paddle.pos_x
            # Choose a capped step based on distance
            max_step = 6.0  # CPU skill
            step = max(-max_step, min(max_step, distance * 0.1))
            cpu_paddle.pos_x += step

            # Clamp to visible bounds using center coordinates
            half = cpu_paddle.width / 2.0
            cpu_paddle.pos_x = max(half, min(self.WIDTH - half, cpu_paddle.pos_x))

    def player_move(self) -> None:
        """
        Move the AI player using its neural network.
        """
        self.ai_player.execute_move(zone=self)

    def check_collisions(self) -> None:
        """
        Handle wall and paddle collisions, score updates, and ball respawns.
        """
        zone = self
        ball = zone.ball

        # Wall collisions (left/right): bounce horizontally
        if ball.right > zone.WIDTH or ball.left < 0:
            ball.flip_x()
            # nudging to avoid sticking to the wall
            ball.update_variables()

        # Bottom border crossed, player paddle missed: CPU +1
        if ball.bottom > zone.HEIGHT:
            zone.ai_player.scores['CPU'] += 1
            # If ball was last hit by CPU, CPU ought to score DOUBLE?.
            if ball.last_hit_by_cpu:
                zone.ai_player.scores['CPU'] += 1
            zone.respawn_ball(ball)
            zone.ai_player.reset_streak()

        # Top border crossed, CPU paddle missed: Player +1
        elif ball.top < 0:
            zone.ai_player.scores['Player'] += 1
            # If ball was last hit by player, player ought to score DOUBLE?.
            if ball.last_hit_by_player:
                zone.ai_player.scores['Player'] += 1
                zone.ai_player.add_streak()  # streak only if player actually touched the ball
            zone.respawn_ball(ball)

        # Paddle collisions
        if self.ball.colliderect(self.ai_paddle):
            self.handle_collision(self.ball, self.ai_paddle)
        if self.ball.colliderect(self.cpu_paddle):
            self.handle_collision(self.ball, self.cpu_paddle, player_is_cpu=True)

    def handle_collision(self, ball, paddle, player_is_cpu=False) -> None:
        """
        Respond to ball hitting a paddle: bounce and skew, update hit counter.
        """
        ball.flip_y()
        skew_ball_direction(ball, paddle, is_cpu=player_is_cpu)
        # TODO: Have ball store hit count, last hit by CPU/player etc.
        #  If ball was last hit by CPU, player ought to score MORE.
        if not player_is_cpu:
            self.ai_player.scores['hits'] += 1
            ball.last_hit_by_player = True
            ball.last_hit_by_cpu = False
        else:
            ball.last_hit_by_cpu = True
            ball.last_hit_by_player = False
        ball.hits += 1

    def update_variables(self) -> None:
        """
        Update the variables of game elements after each update.
        """
        self.ball.update_variables()
        self.ai_paddle.update_variables()
        self.cpu_paddle.update_variables()

    def render_to(self, window_height: float) -> None:
        """
        Render zone entities (delegates to their Arcade draw routines).

        Args:
            window_height: Current window height, used to flip y-axis for Arcade.
        """
        self.ball.render_to(window_height)
        self.ai_paddle.render_to(window_height)
        self.cpu_paddle.render_to(window_height)

    def respawn_ball(self, ball) -> None:
        """
        Respawn the ball near the vertical center with a random X.
        """
        from components.geometry import Vec2
        from random import random, choice, randint

        # Re-seed position
        ball.pos_x, ball.pos_y = randint(120, self.WIDTH - 120), self.HEIGHT / 2
        ball.reset_status()

        # --- NEW: Randomize AI paddle position so it can't safely camp ---
        # half_pad = self.ai_paddle.width / 2
        # self.ai_paddle.pos_x = randint(int(half_pad), int(self.WIDTH - half_pad))

        # Re-seed speed with a safe angle (avoid near-horizontal)
        speed_mag = 12.0
        # Ensure the ball heads generally towards a paddle (pick vertical sign randomly)
        vy_sign = choice([-1.0, 1.0])
        # Pick a small horizontal component so path is learnable, avoid 0
        vx = (0.3 + 0.5 * random()) * choice([-1.0, 1.0]) * speed_mag * 0.2
        vy = vy_sign * (speed_mag * (1.0 - abs(vx) / speed_mag))
        ball.speed = Vec2(vx, vy)