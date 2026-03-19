from random import randint

from functions import create_paddle, create_ball, skew_ball_direction


class PlayZone:
    """
    The Play-zone container.

    Contains unique instances of ball, paddles and player.

    Updates by letting the ball, cpu paddle and player paddle make their moves and handling collisions.

    Updates the scores.

    Args:
        width (int): Width of the play zones.
        height (int): Height of the play zone.
        speed (int): Speed of the game.
        ai_player (IndividualPlayer): AI player object.
        best_ai_fitness (int): Best fitness of the AI player.
    """

    def __init__(self, width, height, speed, ai_player, best_ai_fitness=0):
        """
        Initialize the PlayZone with the specified parameters.

        Args:
        width (int): Width of the play zone.
        height (int): Height of the play zone.
        speed (int): Speed of the game.
        ai_player (IndividualPlayer): AI player object.
        best_ai_fitness (int): Best fitness of the AI player.
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

    def update(self) -> None:
        """
        Update the Play Zone by calculating each player's move, handling collisions and updating variables.
        """
        self.player_move()
        self.cpu_move()
        self.check_collisions()
        self.update_variables()

    def player_move(self) -> None:
        """
        Move the AI player in the play zone by using its Neural Network and related Play-zone attributes.
        """
        self.ai_player.execute_move(zone=self)

    def cpu_move(self) -> None:
        """
        Move the CPU paddle solely based on ball position.
        """
        cpu_paddle = self.cpu_paddle
        ball = self.ball
        cpu_x = cpu_paddle.pos_x
        ball_x = ball.pos_x

        # If ball is heading towards CPU
        if ball.speed.y < 0:
            # Calculate horizontal distance between Ball and CPU-paddle
            distance = cpu_x - ball_x

            # If distance is large : move fast
            if abs(distance) >= cpu_x - self.WIDTH / 2:
                if distance > 0 and cpu_x > 0:
                    cpu_paddle.pos_x -= 5
                elif distance <= 0 and cpu_x < self.WIDTH:
                    cpu_paddle.pos_x += 5

            # Else : move slow
            else:
                cpu_paddle.pos_x += 1

    def check_collisions(self) -> None:
        """
        Check for collisions between the ball and paddles/wall boundaries and handle them.

        Update scores if ball crosses the top or the bottom boundary.

        Paddle collision are handled by separate handling methods.
        """
        zone = self
        ball = zone.ball
        # Ball crosses rightmost area or leftmost area : Bounce off
        if ball.right > zone.WIDTH or ball.left < 0:
            ball.flip_x()
            # update in advance so that the ball doesn't stick to the wall
            ball.update_variables()

        # Ball crossed the bottom border : CPU Score +1
        if ball.bottom > zone.HEIGHT:
            zone.ai_player.scores['cpu'] += 1
            zone.respawn_ball(ball)

        # Ball crossed the top border : Player Score +1
        elif ball.top < 0:
            zone.ai_player.scores['player'] += 1
            zone.respawn_ball(ball)

        # Check and handle "Paddle Collisions"
        if self.ball.colliderect(self.ai_paddle):
            self.handle_collision(self.ball, self.ai_paddle)
        if self.ball.colliderect(self.cpu_paddle):
            self.handle_collision(self.ball, self.cpu_paddle, player_is_cpu=True)

    def handle_collision(self, ball, paddle, player_is_cpu=False) -> None:
        """
        Handle the collision between the ball and a paddle and update score(hits).

        Args:
        ball (Ball): The ball object.
        paddle (Paddle): The paddle object.
        player_is_cpu (bool): Flag to indicate whether the player is the CPU or not.
        """

        ball.flip_y()

        # Tilt the ball's direction
        skew_ball_direction(ball, paddle)

        # Update the score
        if not player_is_cpu:
            self.ai_player.scores['hits'] += 1

    def update_variables(self) -> None:
        """
        Update the variables of game elements after each update.
        """
        self.ball.update_variables()
        self.ai_paddle.update_variables()
        self.cpu_paddle.update_variables()

    def render_to(self, surface) -> None:
        """
        Render the game elements on the screen.

        Args:
        surface: The surface to render the elements on.
        """
        self.ball.render_to(surface)
        self.ai_paddle.render_to(surface)
        self.cpu_paddle.render_to(surface)

    def respawn_ball(self, ball) -> None:
        """
        Respawn the ball at a random position.

        Args:
        ball (Ball): The ball object.
        """
        ball.flip_y()
        ball.pos_x, ball.pos_y = randint(100, self.WIDTH - 100), self.HEIGHT / 2
