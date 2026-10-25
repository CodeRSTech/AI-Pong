# Arcade, simulation, and rendering

Arcade is a Python library for building and displaying 2D games. Here it
provides the rendered window, event loop, and drawing API; it is not the
neural-network or genetic-algorithm library.

`Game` owns one `PlayZone` per individual. At each simulation step it gathers
all players' observations into a NumPy batch, uses PyTorch to predict their
actions together, then applies the actions and advances every zone.
`TIME_OUT` ends an epoch after the configured elapsed duration.

The rendered window draws one representative play zone and its neural
network visualization; the remaining population continues to be simulated
without individual rendering. Use `python -m src.main` for rendered training
or `python -m src.train` for headless training with progress logs. Both run
until interrupted unless `--generations` sets a limit.

The gameplay representative is a display choice, not a promise that the
fitness-ranked elite is being shown. After fitness ranking, the first-ranked
individual is checkpointed and used for validation; that checkpoint is written
before validation runs.

Rendering uses a high-contrast dark court with distinct player/CPU paddle
colors and a bright ball halo. Court markings, palette, and network-panel
styling are draw-only; they do not feed into observations or change simulation
state.

Rendered games generate short paddle-hit, wall-bounce, and scoring tones with Pyglet's synthesis API, so no audio files or new dependencies are needed. Only the displayed zone plays audio, with per-event cooldowns to keep rapid rallies from becoming noisy. Sound is enabled by default in `src.main` and `src.tester`; pass `--no-sound` to either command to disable it. The headless trainer is silent, and audio events do not affect physics or fitness.

After each generation, the best individual is evaluated in fresh games with varied starting positions and directions, duration, and paddle width. The headless evaluator and rendered suite share the scenario setup and result aggregation. Every scenario uses the same fixed simulation delta in either mode, and Python, NumPy, and PyTorch RNG states are isolated and restored before selection and reproduction. Its win and CPU-shutout rates are recorded alongside each run's settings, metrics, elite, and generation checkpoints. Reaching the configured validation target sets a persistent green indicator in the rendered runner; training does not stop. `python -m src.tester` loads the newest run's elite for a one-player, unbounded viewing game.
