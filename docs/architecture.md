# Architecture

## Evolution loop

`src.main` creates an initial population of `IndividualPlayer` instances and
passes it to `GeneticAlgorithm` in `src/ga/ga_core.py`. Each individual owns a
neural network and one `PlayZone`. Every generation:

1. The population plays for the configured generation duration; each player is
   evaluated in its own simulation zone.
2. Fitness is calculated and the population is ranked. The fitness winner is
   checkpointed and tested on fresh validation scenarios (2 by default).
3. Selection and reproduction produce the next population.

Rendered training advances through this lifecycle in one Arcade window and
event loop. Updates perform bounded gameplay, fitness, or validation work so
the window can continue drawing and processing close events. Closing it
cancels an in-progress validation and prevents another generation from
starting. Headless training runs the same simulation synchronously without a
window.

The default population is 200. Training continues until interrupted unless
`--generations` sets a limit.

### Selection, crossover, and mutation

After ranking, the GA seeds a survivor pool with three copies of the top 10%
of the population (at least one elite). It also considers the current
individuals for probabilistic retention: fitter individuals have a higher
chance, using fitness scaled between the weakest and strongest individual in
the generation. When every fitness value is equal, the retention probability
is 0.1. The population is then filled or trimmed to its configured size.

The first 40% of the resulting population (at least two individuals, rounded
to an even number) forms the mating pool. Its members are paired in shuffled
order. Each pair produces two reciprocal offspring with
`two_point_crossover`: two cut points define a contiguous interval in the
ordered list of layer-output neurons. Within that interval, a child inherits
each selected neuron's incoming weights and bias from one parent; outside it,
those values come from the other. Connections leaving selected neurons in
the preceding layer are inherited from the same parent. The second offspring
reverses the parent roles. This is a neuron-level crossover, not a raw slice
through the flattened weight tensors.

![Two-point crossover inheritance. Node colors show the parent supplying each neuron's bias; edge colors show the parent supplying each weight. The strip shows the exact concatenated output-neuron interval.](images/two-point-crossover.svg)

### Crossover animation

<video controls playsinline preload="none" width="100%" poster="../images/two-point-crossover.svg" aria-label="Two-point crossover parameter inheritance animation">
  <source src="../images/animations/two-point-crossover.mp4" type="video/mp4">
  Your browser does not support embedded video. Use the download link below.
</video>

[Watch or download the crossover animation](images/animations/two-point-crossover.mp4).
The static diagram above remains available without video playback.

The clip highlights the half-open interval `[3, 12)` in the ordered list of
layer-output neurons, separates copied segments, transports the donor segments
into new children while leaving the parents intact, and then colors the children's
biases and weights by their source parent (cyan for A, green for B). Input
observations are not inherited parameters. For illustration, both children
use the same cut points; the GA samples cut points independently for each
actual crossover call. The parameter colors are checked against children
produced by the real crossover function. Mutation occurs afterward and is
not shown.

The offspring are reset and mutated. Each network parameter independently
has a 0.1 probability of receiving Gaussian noise scaled by 0.2. Offspring
replace the tail of the selected pool, and the GA restores the configured
population size. These rates are implementation constants in
`src/ga/ga_core.py` and `src/ga/network.py`, not training CLI options.

## Neural network

Each `IndividualPlayer` uses a fully connected **7 → 8 → 6 → 2** network:
seven input values, hidden layers of eight and six neurons, and two action
outputs. The inputs are normalized ball-to-paddle distances, paddle and ball
positions, and the ball's normalized horizontal and vertical velocity. The
hidden layers use `tanh` and ReLU; the output layer uses sigmoid. Outputs
above 0.5 activate the corresponding **Left** or **Right** action.

![The current network topology, input labels, layer sizes, and activations.](images/neural-network.svg)

### Neural-network animation

<video controls playsinline preload="none" width="100%" poster="../images/neural-network.svg" aria-label="Neural-network observations, forward pass, and paddle action animation">
  <source src="../images/animations/neural-network.mp4" type="video/mp4">
  Your browser does not support embedded video. Use the download link below.
</video>

[Watch or download the network animation](images/animations/neural-network.mp4).
The static diagram above remains available without video playback.

The clip carries seven illustrative normalized inputs into the network, focuses
on an actual neuron's weighted contributions, bias, sum and activation, and then
propagates signals through the actual PyTorch model. Positive values are blue,
negative values are pink/coral, and zero is neutral; cyan/green remain portfolio
accents and crossover donor colors, not activation signs. It displays the computed
activations and applies the sigmoid output threshold to obtain a paddle
action. This is a freshly initialized network with seed `334`, **not a trained
elite or a gameplay performance demonstration**. Both active outputs, or
neither active output, result in no movement.

### Modular gameplay explanations

The three additional Manim scenes are driven by one validated gameplay-trace
JSON file and one explicitly selected decision index:

| Scene | Recorded information explained |
| --- | --- |
| `observations` | The y-down court state and the seven observation formulas, with the actual input vector. |
| `network-actions` | That same input, checkpoint activations and signed weights, and the actual two sigmoid decisions. |
| `actions-execution` | The matching action, measured paddle displacement, legal center bounds, CPU movement, and subsequent recorded states/events. |

These are independent modules rather than clips spliced from separate episodes.
The trace records checkpoint SHA-256, capture configuration and software
versions, parameter matrices/biases once, and pre-step/observation/activation/
action/post-step data per decision. Missing run, generation, and validation
provenance is explicitly marked `unknown`. The capture command requires a
specific checkpoint; it never selects the latest file from `runs/` and does
not change training or physics:

```powershell
python -m scripts.capture_gameplay_trace --checkpoint D:\models\selected-generation.pt --seed 42 --steps 90 --output D:\animation-work\gameplay-trace.json
```

To render the three modules from decision 12 of that same trace:

```powershell
.\.venv\Scripts\python.exe -m scripts.render_animations --scenes observations network-actions actions-execution --trace D:\animation-work\gameplay-trace.json --decision-index 12 --quality l --output-dir D:\animation-work\preview
```

The chosen decision controls presentation only; it is not a reliability
evaluation. A reported two-hour training run produced a player scoring below
the CPU; that observation is not a diagnosis, and no suitable performance
demonstration is assumed imminent. Trained gameplay publication remains deferred.
Seeded test checkpoints may verify capture and preview layout but are not
trained gameplay. The terminal-style scene backgrounds use the portfolio
tokens; crossover ownership uses cyan/green, while checkpoint weight signs
keep the source game's blue-positive/coral-negative colors and a separate
legend.

### Future checkpoint-to-animation workflow

The capture and rendering tools are ready for a future selected model. A
checkpoint need not be mathematically optimal to explain its decisions, but
presenting it as successful gameplay requires disclosed evaluation evidence.
Training duration alone does not establish quality.

1. **Preserve the model and evidence.** Copy a stable generation checkpoint to
   an authoring directory, together with its run's `settings.json`, relevant
   `metrics.csv` and `validation.csv` rows, and available
   `validation_scenarios.jsonl` entry. Avoid reading a live `elite_model.pt`
   while it is overwritten. The latest elite is not necessarily best-ever or
   validation-passing. Retain the generation identifier and actual scenario
   count; the default two validation games give limited reliability evidence.
2. **Capture an explicit example.** Supply that immutable checkpoint, a scenario
   seed, a bounded number of decisions (1–1000), and a trace output path. The
   trace records the model SHA-256, parameters, settings used for capture,
   observations, actual activations/actions and consecutive simulation states.
   It is an offline seeded example, not a replay of the original training run.
3. **Attach provenance honestly.** Optional `--run-id`, `--generation`, and
   `--validation-json` attach your supplied metadata. The JSON must be an
   object; it is not an automatic CSV/JSONL importer or independent verification
   of performance. Missing provenance stays `unknown`. Preserve original
   evidence separately rather than replacing it with a hand-written summary.
4. **Check capture settings.** Capture uses the current `src/variables.py`
   defaults, not automatic settings-file import. Its default step delta is
   `1 / (FPS × SPEED × STEPS_PER_FRAME)`. `--step-delta` changes the recorded
   simulation clock increment, not the physics' per-step movement rules.
   Document differences from the training configuration. Explanatory pauses
   and slow motion alter presentation timing only.
5. **Select one recorded decision.** Inspect the JSON's zero-based `decisions`
   array and choose an index that shows the intended lesson. Render all three
   modules with the same trace and `--decision-index`. Do not splice unrelated
   observations and actions or choose a flattering example as proof of general
   reliability. Trace validation rejects malformed dimensions/nonfinite data
   and checks recorded inference against the stored parameters.
6. **Preview, then render.** Use `--quality l` and a private `--output-dir`.
   Inspect positions, normalization, signed colors, output thresholds, actual
   displacement, annotation readability and causal continuity. Only afterward
   render with `--quality h` to produce 1080p/60fps final assets.
7. **Publish deliberately.** The modules produce `gameplay-observations.mp4`,
   `network-actions.mp4` and `actions-execution.mp4`. Add controlled video
   playback, descriptive text and corresponding poster/fallback assets in the
   docs or portfolio after approval. Keep the checkpoint hash and example
   provenance available and describe limitations. Rendering does not commit,
   push or integrate these new videos automatically.

For example, attach provenance while capturing:

```powershell
python -m scripts.capture_gameplay_trace --checkpoint D:\models\selected-generation.pt --seed 42 --steps 90 --output D:\animation-work\gameplay-trace.json --run-id selected-run --generation 120 --validation-json D:\animation-work\validation-summary.json
```

After reviewing the preview, render the same selected decision at final quality:

```powershell
.\.venv\Scripts\python.exe -m scripts.render_animations --scenes observations network-actions actions-execution --trace D:\animation-work\gameplay-trace.json --decision-index 12 --quality h --output-dir D:\animation-work\final
```

The model is implemented with PyTorch `nn.Linear` layers. For population
inference, `BatchedPopulationBrain` stacks the individuals' weights and
biases, then evaluates them together using batched matrix multiplication.
This lets one forward pass evaluate all players at each simulation step.
PyTorch uses CUDA when available and otherwise runs on the CPU.

NumPy is Python's array and numerical-computing library; here it is used for
reusable observation and population-input arrays, as well as helpers such as
the fitness function's squashing operation. PyTorch—not NumPy—stores and
evaluates the neural-network parameters.

## Arcade, simulation, and rendering

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

## Regenerating the diagrams

The network and crossover figures in this guide are generated from the
current `IndividualPlayer` architecture and the shared crossover interval
mask logic. From the repository root, run:

```console
python -m scripts.generate_diagrams
```

The script rewrites `docs/images/neural-network.svg` and
`docs/images/two-point-crossover.svg`. Keep the generated files checked in so
the documentation renders without running the generator during site builds.

### Regenerating the animations

Manim is optional authoring tooling: training, SVG generation, tests of the
animation data, and documentation builds do not require it. Install it in
the environment used for rendering:

```console
python -m pip install manim
python -m scripts.render_animations
```

On Windows, if Manim is installed in the project's virtual environment, use
`.\.venv\Scripts\python.exe -m scripts.render_animations`.
The renderer uses ordinary text rather than TeX, so these scenes do not
require LaTeX. It selects installed Fira Code when available and otherwise
prints that it is using the installed Consolas monospace fallback; fonts are
not installed or bundled by the renderer.

The command writes `neural-network.mp4` and `two-point-crossover.mp4` beneath
`docs/images/animations/`, using a temporary directory for intermediate
render files. Its default is 480p at 15 fps; use `--quality m` for 720p at
30 fps or `--quality h` for 1080p at 60 fps. Commit the finished clips so
GitHub Pages can serve them without installing Manim or rendering scenes.
