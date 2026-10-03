# Architecture

## Evolution loop

`src/main.py` creates an initial population of `IndividualPlayer` instances and passes it to `GeneticAlgorithm` in `src/ga/ga_core.py`. Each generation:

1. Resets population scores and runs one game epoch.
2. Calculates and sorts individuals by fitness.
3. Keeps elites and selects other survivors.
4. Crosses over survivors, mutates offspring, and restores the configured population size.

The default population is 200, the GA retains an elite fraction of 0.1, and the crossover rate is 0.4. The run limit defaults to 1,000 generations.

## Neural network and batched brain

Each `IndividualPlayer` uses a fully connected `7 → 8 → 6 → 2` neural network. Its inputs describe ball/paddle positions and ball velocity. The layers use `tanh`, `ReLU`, and `sigmoid`; the two outputs represent left and right actions, thresholded at `0.5`.

`BatchedPopulationBrain` stacks the networks' weights and evaluates the population together with PyTorch batch matrix operations. This avoids making one separate forward-pass call per player at every simulation step.

## Simulation and rendering

`Game` owns one `PlayZone` per player. A game step gathers each player's observations, evaluates the batch, applies actions, and advances the zones. `TIME_OUT` ends an epoch after the configured elapsed duration.

The Arcade window renders only one representative zone alongside its neural-network visualization; the rest of the population continues to be simulated. Each run's settings, metrics, elite, and generation checkpoints are saved in its timestamped run folder. `python -m src.tester` loads the newest run's elite for a one-player, unbounded viewing game.
