# src/playzone.py
"""
PlayZone: a self-contained arena that owns a ball, two paddles and an AI player.

This module keeps simulation in y-down space (origin top-left). Rendering
converts to y-up in draw calls of each entity.
"""

from utils.functions import skew_ball_direction, create_paddle, create_ball


class PlayZone:
    """
    The Play-zone container.

    Contains unique instances of ball, paddles and player.

    Updates by letting the ball, cpu paddle and player paddle make their moves
    and handling collisions. Updates the scores.
    """
    # TODO: correlate with pre-set magnitude of ball speed
    ball_speed_magnitude = 5.0

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

        self.ball = create_ball()
        self.cpu_paddle = create_paddle(width, height, self.ball.color, is_cpu=True)
        self.ai_paddle = create_paddle(width, height, self.ball.color)
        self.ai_player = ai_player
        self.best_ai_fitness = best_ai_fitness

        self.respawn_ball(self.ball)

    def update(self) -> None:
        """
        Advance simulation one step: CPU, collisions, and state sync.
        """
        self.cpu_move()
        self.check_collisions()
        self.update_variables()

    def cpu_move(self) -> None:
        """
        Move the CPU paddle to track the ball when ball is moving upwards.
        """
        # FIXME: This CPU Move logic is rather flawed...
        #  ideally, the paddle moves faster when ball is in close proximity.
        #  otherwise the paddle just procrastinates and moves slowly.

        # TODO: revisit original, base version and see how the CPU Move implementation differs.
        #  After that, re-integrate that into this version.
        cpu_paddle = self.cpu_paddle
        ball = self.ball

        # Only react if ball is heading towards the CPU (top in y-down space)

        if ball.speed.y < 0:
            distance = ball.pos_x - cpu_paddle.pos_x
            # Choose a capped step based on distance
            max_step = self.speed  # CPU skill
            step = max(-max_step, min(max_step, distance * 0.1))
            cpu_paddle.pos_x += step

            # Clamp to visible bounds using center coordinates
            half = cpu_paddle.width / 2.0
            cpu_paddle.pos_x = max(half, min(self.WIDTH - half, cpu_paddle.pos_x))

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
            zone.respawn_ball(ball)
            zone.ai_player.reset_winning_streak()

        # Top border crossed, CPU paddle missed: Player +1
        elif ball.top < 0:
            zone.ai_player.scores['Player'] += 1
            zone.respawn_ball(ball)
            zone.ai_player.add_win_to_streak()


        # Paddle collisions
        if self.ball.colliderect(self.ai_paddle):
            self.handle_collision(self.ball, self.ai_paddle)

        if self.ball.colliderect(self.cpu_paddle):
            self.handle_collision(self.ball, self.cpu_paddle, player_is_cpu=True)

    def handle_collision(self, ball, paddle, player_is_cpu=False) -> None:
        """
        Respond to ball hitting a paddle: bounce and skew, update hit counter.
        """

        ai_player = self.ai_player
        ball.flip_y()
        skew_ball_direction(ball, paddle, is_cpu=player_is_cpu)
        if player_is_cpu:
            ai_player.scores['CPU Hits'] += 1
        else:
            ai_player.scores['Player Hits'] += 1
            ai_player.add_hit_to_streak()
        # TODO: Update variables here to avoid ball sticking to the paddle.
        self.update_variables()

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

        self.ai_player.reset_hit_streak()
        # Re-seed position
        # TODO: New coordinates should be random within a 'safe-space' that is close to the center,
        #  both horizontally and vertically
        ball.pos_x = randint(10, self.WIDTH - 10)
        ball.pos_y = randint(self.HEIGHT // 2 - 100, self.HEIGHT // 2 + 100)

        # --- NEW: Randomize paddle positions so it AI Paddle can't safely camp ---
        # FIXME: This was recently enabled,
        #  try commenting out 3 lines below incase code struggles with convergence
        #half_pad = self.ai_paddle.width / 2
        #self.ai_paddle.pos_x = randint(int(half_pad), int(self.WIDTH - half_pad))
        #self.cpu_paddle.pos_x = randint(int(half_pad), int(self.WIDTH - half_pad))

        # Re-seed speed with a safe angle (avoid near-horizontal)
        # Ensure the ball heads generally towards a paddle (pick vertical sign randomly)
        vy_sign = choice([-1.0, 1.0])
        # Pick a small horizontal component so path is learnable, avoid 0
        vx = (0.3 + 0.5 * random()) * choice([-1.0, 1.0]) * self.ball_speed_magnitude * 0.4
        vy = vy_sign * (self.ball_speed_magnitude * (1.0 - abs(vx) / self.ball_speed_magnitude))
        ball.speed = Vec2(vx, vy)

