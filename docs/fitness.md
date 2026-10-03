# Fitness function

`IndividualPlayer.calculate_fitness()` in `src/ga/player.py` combines match results, movement, hits, and streaks. The terms are summed and then multiplied by a score-ratio multiplier and an optional shutout bonus.

Let `P` be the player's score, `C` the CPU score, and `squash(x, k) = k × tanh(x / k)`.

| Term | Calculation | Purpose |
| --- | --- | --- |
| Score gap | `(P - C) × 17` | Rewards winning by a larger margin and penalizes losses. |
| Movement bonus | `squash(left_moves + right_moves, 1000) × 2^(2 × move_ratio)` | Rewards useful movement, with a multiplier for moving in both directions. `move_ratio` is the smaller direction count divided by the larger, or zero if either direction has no moves. |
| Hits | `player_hits × (player_hits / (cpu_hits + 1)) × 2` | Rewards player paddle hits relative to CPU hits. |
| Hit streak | `longest_player_hit_streak × 2.5` | Rewards consecutive successful player hits. |
| Winning streak | `squash(longest_player_win_streak, 5)` | Rewards consecutive points while limiting this term's growth. |

The sum is multiplied by `(P / (C + 1) + 1)^2`. A further `1.25` multiplier applies when the player scores more than one point and the CPU scores none. This means match score dominance has a strong effect on the final fitness, while the other terms provide secondary incentives.
