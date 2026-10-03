# Training tips

## Generation timeout

`TIME_OUT` in `src/variables.py` defaults to 12 seconds per generation. Increase it if players need more time to rally and score. `-1` means no timeout for an interactive `Game`; headless training requires a finite positive timeout.

## Population size

The initial population size defaults to 200. Choose another size with `python -m src.main --population 100`. A larger population explores more candidate networks per generation but requires more computation and memory. The GA restores this target size between generations.

## Mutation and selection

Mutation settings are currently code constants rather than CLI options: offspring use a mutation scale of `0.2`, and each parameter has the network's default mutation probability of `0.1`. The GA's elite factor (`0.1`) and crossover rate (`0.4`) are in `src/ga/ga_core.py`. Each run now records its seed and settings in `settings.json` and writes one metrics row per generation to `metrics.csv`.

## Update rate, speed, and simulation steps

`FPS` is the base update rate; `SPEED` multiplies the Arcade update rate and is also passed to play zones to affect paddle movement. `STEPS_PER_FRAME` (default 15) determines the number of simulation steps evaluated on each Arcade update. The approximate simulation workload is `FPS × SPEED × STEPS_PER_FRAME` steps per second. The per-step time increment is scaled so `TIME_OUT` remains elapsed seconds, not a number of simulation steps.

## Checkpoints

Each training run creates a timestamped subdirectory under `runs/` (or the path specified with `--output-dir`). It contains `settings.json`, `metrics.csv`, `elite_model.pt`, and top-player checkpoints under `checkpoints/`. Run `python -m src.tester` to load the newest run's elite, or specify a file with `python -m src.tester --checkpoint path\to\model.pt`.

## Command-line examples

```bash
python -m src.main --population 100 --generations 50 --seed 12
python -m src.main --timeout 30 --no-render --output-dir D:\training-runs
python -m src.main --render --fps 60 --speed 1.5 --steps-per-frame 4
```
