# src/player.py
"""
AI Player driven by a tiny neural network.
"""
import uuid
import numpy as np
from src.ga import NeuralNet
from src.utils import squash


class IndividualPlayer:
    """
    Class for an Individual (AI) in the Population.
    """

    def __init__(self):
        self.uid = None
        self.age = 0
        self.current_hit_streak = 0
        self.max_hit_streak = 0
        self.current_win_streak = 0
        self.max_win_streak = 0
        self.fit_scores = None
        self.scores = None
        self.neural_net = None
        
        self.create_neural_net()
        self.reset_defaults()

    def create_neural_net(self):
        self.neural_net = NeuralNet()
        self.neural_net.add_layer(7, 8, activation='tanh')  # coarse intercept features
        self.neural_net.add_layer(8, 6, activation='relu')  # fine adjustment / gating
        self.neural_net.add_layer(6, 2, activation='binary')  # Left / Right decisions

    @staticmethod
    def look(zone) -> np.ndarray:
        ball = zone.ball
        ai_paddle = zone.ai_paddle

        ball_distance_x = ball.pos_x - ai_paddle.pos_x
        ball_distance_y = ball.pos_y - ai_paddle.pos_y

        ball_speed = ball.speed.magnitude()

        zone_width = zone.WIDTH
        zone_height = zone.HEIGHT

        # Normalize inputs relative to zone dimensions
        inputs_list = np.array([
            ball_distance_x / zone_width,
            ball_distance_y / zone_height,
            (ai_paddle.pos_x * 2.0 - zone_width) / zone_width,  # Paddle position normalized
            (ball.pos_x * 2.0 - zone_width) / zone_width,  # Ball X normalized
            (ball.pos_y * 2.0 - zone_height) / zone_height,  # Ball Y normalized
            ball.speed.x / ball_speed,
            ball.speed.y / ball_speed
        ])
        return inputs_list

    def think(self, inputs_list) -> np.ndarray:
        """
        Feed-forward decision using the neural network.
        """
        outputs = self.neural_net.predict(inputs_list)
        return outputs

    def apply_move(self, zone, move_left: bool, move_right: bool) -> bool:
        """
        Perform a move based on directly supplied boolean actions from the Batch Brain.
        """
        player_paddle = zone.ai_paddle
        speed = zone.speed

        if move_left and move_right:
            return False

        move_distance = max(2.0, float(speed))
        half = player_paddle.width / 2.0
        min_x = half
        max_x = zone.WIDTH - half

        intended_dir = 0
        if move_left:
            intended_dir = -1
        elif move_right:
            intended_dir = 1

        if intended_dir == -1:
            new_pos = max(min_x, player_paddle.pos_x - move_distance)
            if new_pos == player_paddle.pos_x:
                # No movement occurred
                pass
            else:
                player_paddle.pos_x = new_pos
                self.scores['Left Moves'] += 1

        elif intended_dir == 1:
            new_pos = min(max_x, player_paddle.pos_x + move_distance)
            if new_pos == player_paddle.pos_x:
                # No movement occurred
                pass
            else:
                player_paddle.pos_x = new_pos
                self.scores['Right Moves'] += 1
        return True

    def add_hit_to_streak(self) -> None:
        """Update the streak counter."""
        self.current_hit_streak += 1
        self.scores['Hit Streak'] = max(self.max_hit_streak, self.current_hit_streak)

    # NEW! Winning streak. Counts the number of consecutive wins.
    def add_win_to_streak(self) -> None:
        """Update the streak counter."""
        self.current_win_streak += 1
        self.scores['Winning Streak'] = max(self.max_win_streak, self.current_win_streak)

    def calculate_fitness(self) -> float:
        score = self.scores

        # Score points (wins scored by either side)
        # -----------------------------------------
        player_score = score['Player']
        cpu_score = score['CPU']

        # A positive score gap generally implies that the player is better than the CPU.
        score_gap = player_score - cpu_score
        score_ratio = player_score / (cpu_score + 1)
        # ^^^^^^^ Score ratio is directly proportional to player score and,
        #           Inversely proportional to CPU score

        # Movements (left, right and total)
        # ------------------------------------------
        left_moves = score['Left Moves']
        right_moves = score['Right Moves']
        total_moves = left_moves + right_moves
        # A moves ratio of 1 implies that the player is moving equally in both directions.
        # Regardless of how many times those moves were performed or switched.
        moves_ratio = 0.0
        if (left_moves > 0) and (right_moves > 0):
            #moves_ratio = math.sqrt(min(left_moves, right_moves) / max(left_moves, right_moves))
            moves_ratio = min(left_moves, right_moves) / max(left_moves, right_moves)

        # Hits (number of times the ball struck the pedal)
        # ------------------------------------------------
        player_hits = score['Player Hits']
        cpu_hits = score['CPU Hits']
        # total_hits = player_hits + cpu_hits
        hits_ratio = player_hits / (cpu_hits + 1)

        # Streaks (number of consecutive hits/wins by player)
        # ---------------------------------------------------
        hits_streak = score['Hit Streak']
        winning_streak = score['Winning Streak']

        # CONSTANTS
        # ===========================================
        gap_weight = 17
        score_ratio_multiplier = (score_ratio+1) ** 2
        victory_bonus_multiplier = 1.25 if (player_score > 1 and cpu_score == 0) else 1.0
        move_bonus_multiplier = 2 ** (2 * moves_ratio)
        move_bonus_squash_factor = 1000
        hits_weight = 2
        hits_streak_multiplier = 2.5
        winning_streak_squash_factor = 5

        fit_scores = {
            'Gap': score_gap * gap_weight,
            'Move-bonus': squash(total_moves, move_bonus_squash_factor) * move_bonus_multiplier,
            #'Hits': squash(player_hits, 50),
            'Hits': player_hits * hits_ratio * hits_weight,
            #'Hits-streak': squash(hits_streak, 20),
            'Hits-streak': hits_streak * hits_streak_multiplier,
            #'Winning-streak': squash(winning_streak * 10, 25),
            'Winning-streak': squash(winning_streak, winning_streak_squash_factor),
        }

        fitness = sum(fit_scores.values())

        # A higher score_ratio implies player dominates the CPU
        fitness *= score_ratio_multiplier
        fitness *= victory_bonus_multiplier
        fitness = fitness

        fit_scores['Score Ratio'] = score_ratio
        fit_scores['Total Moves'] = total_moves
        fit_scores['Move Ratio'] = moves_ratio
        fit_scores['Fitness'] = int(fitness)

        self.scores['fitness'] = fitness
        self.fit_scores = fit_scores
        return fitness

    def increase_age(self) -> int:
        """Increase age by one."""
        self.age += 1
        return self.age

    # NEW! Separate method for retrieving fitness value.
    def get_fitness(self) -> float:
        """Get the fitness."""
        return self.scores['fitness']

    def get_scores(self) -> str:
        fitness = self.scores['fitness']
        player_wins = self.scores['Player']
        cpu_wins = self.scores['CPU']
        score_gap = player_wins - cpu_wins

        player_hits = self.scores['Player Hits']
        cpu_hits = self.scores['CPU Hits']
        hit_streak = self.scores['Hit Streak']
        winning_streak = self.scores['Winning Streak']

        score_ratio = self.fit_scores['Score Ratio']
        total_moves = self.fit_scores['Total Moves']
        move_ratio = self.fit_scores['Move Ratio']
        gap_score = self.fit_scores['Gap']
        move_bonus = self.fit_scores['Move-bonus']
        hits_score = self.fit_scores['Hits']
        hits_streak_score = self.fit_scores['Hits-streak']
        winning_streak_score = self.fit_scores['Winning-streak']

        other_scores = hits_score + hits_streak_score + winning_streak_score

        victory_bonus_multiplier = 1.25 if (player_wins > 1 and cpu_wins == 0) else 1.0

        return (
            f"Fitness: {fitness}\n"
            f"Scores -> "
            f"Player: {player_wins}, "
            f"CPU: {cpu_wins}, "
            f"Ratio: {score_ratio:0.2f}, "
            f"Gap: {score_gap}, "
            f"Longest Streak: {winning_streak},\n"

            f"Moves -> "
            f"Total: {total_moves}, "
            f"Ratio: {move_ratio:0.4f},\n"

            f"Hits -> "
            f"Player: {player_hits}, "
            f"CPU: {cpu_hits}, "
            f"Longest Streak: {hit_streak},\n"

            f"Others: {other_scores:0.4f}\n"

            f"Fitness math:\n"
            f"-------------\n"
            f"[ \t{gap_score} <gap>\n"
            f" +\t{move_bonus:0.2f} \t<move-bonus>\n"
            f" +\t{hits_score:0.2f} \t<hits>\n"
            f" +\t{hits_streak_score:0.2f} \t<hits-streak>\n"
            f" +\t{winning_streak_score:0.2f} \t<winning-streak> ] "
            f"* {(score_ratio + 1) ** 2:0.2f} "
            f"* {victory_bonus_multiplier:0.2f}\n"
        )

    def reset_hit_streak(self) -> None:
        """Reset the streak counter."""
        self.max_win_streak = max(self.max_hit_streak, self.current_hit_streak)
        self.current_hit_streak = 0

    # Winning streak. Counts the number of consecutive wins.
    def reset_winning_streak(self) -> None:
        """Reset the streak counter."""
        self.max_win_streak = max(self.max_win_streak, self.current_hit_streak)
        self.current_hit_streak = 0

    def reset_scores(self) -> None:
        """
        Specifically, reset game-play stats without wiping UID or Age.
        """
        self.current_hit_streak = 0
        self.max_hit_streak = 0
        self.current_win_streak = 0
        self.max_win_streak = 0
        self.scores = {
            'Player': 0,
            'CPU': 0,
            'Left Moves': 0,
            'Right Moves': 0,
            'Player Hits': 0,
            'CPU Hits': 0,
            'Hit Streak': 0,
            'Winning Streak': 0,
            'fitness': 0,
        }
        self.fit_scores = None

    def reset_defaults(self) -> None:
        """
        Reset stats and assign new UID for brand-new offspring/pupils.
        """
        self.uid = uuid.uuid4()
        self.age = 0
        self.reset_scores()  # Call the new helper here

    @property
    def __str__(self) -> str:
        """Readable identity."""
        return 'Player = {0}; fitness = {1}'.format(self.uid, self.scores['fitness'])

    def __repr__(self):
        return 'Player = {0}; fitness = {1}'.format(self.uid, self.scores['fitness'])