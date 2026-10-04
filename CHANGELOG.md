# Changelog

Notable project changes are recorded here. This changelog may not include every change.

## Unreleased

No unreleased changes.

## 1.4.0

See the [1.4.0 release notes](docs/releases/1.4.0.md) for CLI details, artifact
schemas and save timing, migration guidance, and known limitations.

### Training and reproducibility

- Added separate rendered (`python -m src.main`) and headless
  (`python -m src.train`) training commands with population, generation,
  random-seed, timeout, output-directory, simulation-rate, CPU-thread, and
  validation-suite options.
- Created timestamped per-run directories containing the run configuration,
  generation metrics, validation summaries and scenarios, the current
  generation's fitness winner, and the top two generation checkpoints.
- Added periodic headless progress reports and explicit newest-run or
  `--checkpoint` selection in `src.tester`.
- Seeded Python, NumPy, and PyTorch training randomness; validation scenarios
  are generated per generation and evaluation restores the training RNG state
  before selection and reproduction.
- Documented that each invocation starts a fresh population: run artifacts are
  for inspection and model playback, not a full evolution-resume snapshot.

### Genetic algorithm and simulation

- Corrected crossover inheritance to copy selected contiguous neuron rows and
  their outgoing weights, with architecture checks and a non-empty interval
  selected between distinct cut points.
- Made simulation steps per rendered frame configurable and retained batched
  PyTorch population inference; fitness calculation and rendered validation
  advance in bounded batches to keep the window responsive.
- Added randomized, fixed-step elite validation with win/shutout rates,
  per-scenario results, and a non-stopping success indicator.
- Defaulted validation to 2 scenarios in both training modes for faster feedback;
  `--validation-games` selects a larger suite. The unchanged 95% win/90% shutout
  targets require two wins and two shutouts at the default count; this small
  sample is not strong evidence of reliability.

### Rendered experience and lifecycle

- Refreshed the court, paddle contrast, ball visibility, and neural-network
  panel with a higher-contrast visual palette.
- Added generated, rate-limited paddle-hit, wall-bounce, and scoring tones to
  rendered play; `--no-sound` disables them.
- Kept rendered training in one Arcade window through gameplay, fitness and
  checkpoints, visible elite validation, selection, crossover, mutation, and
  the next generation; window close cancels future transitions.

### Project maintenance and documentation

- Added package metadata, development/docs extras, pytest and Ruff settings,
  CI, Dependabot, issue and pull-request templates, and contributor guidance.
- Added the MkDocs documentation site and GitHub Pages deployment workflow,
  with architecture, fitness, coordinate, training, and roadmap guides.
- Expanded automated coverage for simulation, training, validation, artifacts,
  crossover, audio, and the continuous rendered lifecycle.

## 1.3.0

- Migrated from Pygame to Arcade.
- Replaced the NumPy perceptron with a PyTorch neural network and updated the network structure and fitness function.
- Added a visual representation of the neural network to the UI.
- Added borders to the ball and paddles.
- Batched inference across the population for improved performance.
- Displayed only the elite player's play zone and added configurable frame skipping.
