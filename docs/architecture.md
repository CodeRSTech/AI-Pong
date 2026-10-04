# Architecture

## Evolution loop

`src/main.py` creates an initial population of `IndividualPlayer` instances and passes it to `GeneticAlgorithm` in `src/ga/ga_core.py`. Headless generations continue synchronously through the evaluator. Rendered
training uses one Arcade window and one event loop for the full lifecycle:

1. Reset population scores and play generation gameplay.
2. Calculate fitness in small batches, then save metrics and checkpoints.
3. Play the elite through the configured fresh validation scenarios in the same window (2 by default).
4. Keep elites and select other survivors, then cross over, mutate, and restore the configured population.
5. Begin the next generation without closing or recreating the window.

Rendered updates advance only a bounded amount of gameplay or validation at a
time, leaving Arcade's draw and close events responsive during the validation
suite. The displayed phase, generation, scenario progress, and scores identify
what the runner is doing between epochs. Closing the window cancels an
in-progress suite and prevents another generation from starting.

The default population is 200, the GA retains an elite fraction of 0.1, and the crossover rate is 0.4. Training runs continuously unless `--generations` sets a limit.

## Neural network and batched brain

Each `IndividualPlayer` uses a fully connected `7 → 8 → 6 → 2` neural network. Its inputs describe ball/paddle positions and ball velocity. The layers use `tanh`, `ReLU`, and `sigmoid`; the two outputs represent left and right actions, thresholded at `0.5`.

`BatchedPopulationBrain` stacks the networks' weights and evaluates the population together with PyTorch batch matrix operations. This avoids making one separate forward-pass call per player at every simulation step.

## Simulation and rendering

`Game` owns one `PlayZone` per player. A game step gathers each player's observations, evaluates the batch, applies actions, and advances the zones. `TIME_OUT` ends an epoch after the configured elapsed duration.

The Arcade window renders only one representative zone alongside its neural-network visualization; the rest of the population continues to be simulated. Use `python -m src.main` for the rendered training runner or `python -m src.train` for headless training with progress logs. Both continue until interrupted unless `--generations` sets a limit.

The gameplay representative is a display choice, not a promise that the
fitness-ranked elite is being shown. After fitness ranking, the first-ranked
individual is checkpointed and used for validation; that checkpoint is written
before validation runs.

Rendering uses a high-contrast dark court with distinct player/CPU paddle colors and a bright ball halo. Court markings, palette, and network-panel styling are draw-only; they do not feed into observations or change simulation state.

Rendered games generate short paddle-hit, wall-bounce, and scoring tones with Pyglet's synthesis API, so no audio files or new dependencies are needed. Only the displayed zone plays audio, with per-event cooldowns to keep rapid rallies from becoming noisy. Sound is enabled by default in `src.main` and `src.tester`; pass `--no-sound` to either command to disable it. The headless trainer is silent, and audio events do not affect physics or fitness.

After each generation, the best individual is evaluated in fresh games with varied starting positions and directions, duration, and paddle width. The headless evaluator and rendered suite share the scenario setup and result aggregation. Every scenario uses the same fixed simulation delta in either mode, and Python, NumPy, and PyTorch RNG states are isolated and restored before selection and reproduction. Its win and CPU-shutout rates are recorded alongside each run's settings, metrics, elite, and generation checkpoints. Reaching the configured validation target sets a persistent green indicator in the rendered runner; training does not stop. `python -m src.tester` loads the newest run's elite for a one-player, unbounded viewing game.
