# Training tips

## Generation timeout

`TIME_OUT` in `src/variables.py` defaults to 12 seconds per generation. Increase it if players need more time to rally and score. `-1` means no timeout for an interactive `Game`; training commands require a finite positive timeout.

## Population size

The initial population size defaults to 200. Choose another size with `python -m src.train --population 100`. A larger population explores more candidate networks per generation but requires more computation and memory. The GA restores this target size between generations.

## Mutation and selection

Mutation settings are currently code constants rather than CLI options: offspring use a mutation scale of `0.2`, and each parameter has the network's default mutation probability of `0.1`. The GA's elite factor (`0.1`) and crossover rate (`0.4`) are in `src/ga/ga_core.py`. Each run records its seed and settings in `settings.json`, one training row per generation in `metrics.csv`, and validation summaries/scenarios in `validation.csv` and `validation_scenarios.jsonl`.

## Update rate, speed, and simulation steps

`FPS` is the base update rate; `SPEED` multiplies the Arcade update rate and is also passed to play zones to affect paddle movement. `STEPS_PER_FRAME` (default 15) determines the number of simulation steps evaluated on each Arcade update. The approximate simulation workload is `FPS × SPEED × STEPS_PER_FRAME` steps per second. The per-step time increment is scaled so `TIME_OUT` remains elapsed seconds, not a number of simulation steps.

Headless runs report progress every tenth of each generation, including measured simulation steps per second. `--torch-threads` controls PyTorch's CPU intra-op parallelism and defaults to one thread for these small batched networks.

Rendered `src.main` training and `src.tester` playback use synthesized paddle-hit, wall-bounce, and scoring sounds by default. Use `--no-sound` to mute either rendered command. Headless training does not initialize or play sounds.

## Rendered generation lifecycle

`python -m src.main` keeps one Arcade window open while it moves through
generation gameplay, fitness and checkpointing, the elite's configured visible
validation scenarios, selection, and reproduction. The window shows the
current generation and phase; during validation it also shows scenario
progress, the live score, and the accumulated win/loss/tie record. Closing the
window interrupts the run without launching another generation. A finite
`--generations` limit leaves the window open with a completion status.

Validation advances in fixed-delta batches so the rendered suite uses the same
scenario setup, timing, and metrics as headless validation without blocking
Arcade's event loop for the entire suite. All configured scenarios are played
visibly in rendered runs, so the between-generation portion takes longer than
headless validation.

## Checkpoints

Each training run creates a timestamped subdirectory under `runs/` (or the path specified with `--output-dir`). It contains `settings.json`, `metrics.csv`, `validation.csv`, `validation_scenarios.jsonl`, `elite_model.pt`, and top-player checkpoints under `checkpoints/`. Metrics and the generation checkpoints, including `elite_model.pt`, are saved after fitness ranking and before validation. The elite file is overwritten each generation with that generation's fitness winner; it is not the best-ever model and may not pass validation. Validation records are appended after the entire scenario suite finishes. See the [1.4.0 release notes](releases/1.4.0.md) for exact fields and timing.

Run `python -m src.tester` to load the newest run directory containing an elite checkpoint, or specify a file with `python -m src.tester --checkpoint path\to\model.pt`. The viewer also accepts `--runs-dir` to change the default search root (`runs`). Existing checkpoints are not used to resume evolution: each training invocation creates a fresh population and a new run directory.

## Validation target

**Recommended:** use `--validation-games 20` or a higher count when assessing an elite's reliability. The default of 2 is intended for quick developmental feedback; a green indicator from such a small suite is not a robust assessment. Larger suites take longer, especially when rendered, and still do not guarantee performance on every possible scenario.

```console
python -m src.main --validation-games 20
python -m src.train --validation-games 20
```

Every generation's elite is tested in 2 fresh scenarios by default in both training modes. Use `--validation-games 20` (or another positive count) for a larger suite. Ball and paddle positions and ball direction vary, game duration ranges from 6–18 seconds, and both paddles use a width from 50–120 pixels. The target remains at least 95% wins and 90% CPU shutouts. With 2 games, both must be wins and CPU shutouts, and the small sample provides limited evidence of reliability. Reaching the target is recorded and shown in green in the rendered `src.main` window; it is informational only, and training continues.

## Command-line examples

```bash
python -m src.train --population 100 --generations 50 --seed 12
python -m src.train --timeout 30 --output-dir D:\training-runs
python -m src.main --fps 60 --speed 1.5 --steps-per-frame 4
```
