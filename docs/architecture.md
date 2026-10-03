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

The Arcade window renders only one representative zone alongside its neural-network visualization; the rest of the population continues to be simulated. Use `python -m src.main` for the rendered training runner or `python -m src.train` for headless training with progress logs. Both continue until interrupted unless `--generations` sets a limit.

Rendering uses a high-contrast dark court with distinct player/CPU paddle colors and a bright ball halo. Court markings, palette, and network-panel styling are draw-only; they do not feed into observations or change simulation state.

Rendered games generate short paddle-hit, wall-bounce, and scoring tones with Pyglet's synthesis API, so no audio files or new dependencies are needed. Only the displayed zone plays audio, with per-event cooldowns to keep rapid rallies from becoming noisy. Sound is enabled by default in `src.main` and `src.tester`; pass `--no-sound` to either command to disable it. The headless trainer is silent, and audio events do not affect physics or fitness.

After each generation, the best individual is evaluated in fresh games with varied starting positions and directions, duration, and paddle width. Its win and CPU-shutout rates are recorded alongside each run's settings, metrics, elite, and generation checkpoints. Reaching the configured validation target sets a persistent green indicator in the rendered runner; training does not stop. `python -m src.tester` loads the newest run's elite for a one-player, unbounded viewing game.
