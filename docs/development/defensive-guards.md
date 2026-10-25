# Defensive guards

AI-Pong fails fast with an explicit, descriptive exception at API boundaries
rather than silently continuing with invalid state. This page documents the
existing pattern so new code stays consistent with it.

## Guard against invalid configuration and empty state

`src/game.py`, `src/ga/ga_core.py`, and `src/ga/validation.py` validate their
constructor and method arguments before doing any work, for example:

```python
if not players:
    raise ValueError("Game requires at least one player.")
if fps <= 0 or speed <= 0 or steps_per_frame <= 0:
    raise ValueError("fps, speed, and steps_per_frame must be positive.")
if timeout <= 0 and timeout != -1:
    raise ValueError("timeout must be positive, or -1 for an unbounded game.")
```

The genetic algorithm applies the same pattern to population invariants it
depends on for correctness, raising `RuntimeError` when an operation is
requested on a population that cannot produce a meaningful result:

```python
if len(population) < 2:
    raise ValueError("The genetic algorithm requires at least two individuals.")
...
if not ranked_population:
    raise RuntimeError("Cannot calculate fitness for an empty population.")
```

`EliteValidator` (`src/ga/validation.py`) validates its scenario ranges once,
in `__init__`, so a misconfigured validator fails immediately rather than
producing nonsensical scenarios later:

```python
if games < 1:
    raise ValueError("Validation requires at least one game.")
if not 0 < win_rate_target <= 1 or not 0 < shutout_rate_target <= 1:
    raise ValueError("Validation rate targets must be in (0, 1].")
```

## Guard against architecture and activation mismatches

`src/utils/functions.py`'s crossover helpers check that both parent networks
share the same layer shapes before combining their parameters, and
`src/ga/network.py` rejects an unrecognized activation name at the point a
layer is added, instead of failing obscurely during a later forward pass:

```python
if parent_a_sizes != parent_b_sizes:
    raise ValueError("Parents must have matching neural-network architectures.")
...
raise ValueError(f"Unknown activation: {activation}")
```

`scripts/generate_diagrams.py` applies the same idea to the diagrams
generator: it refuses to draw a network with no layers, mismatched
layer dimensions, an unsupported activation, or observation/action labels
that do not match the model's actual input/output size, so a diagram can
never silently misrepresent the live architecture.

## Guard exported and captured artifacts

`scripts/export_web_model.py` only accepts a checkpoint whose state dict is
*exactly* the current 7 → 8 → 6 → 2 parameter set, with every tensor finite
float32 of the expected shape; anything else raises `ValueError` describing
what was expected. `scripts/gameplay_scenes.py` similarly validates its
required environment variables (trace path, decision index) before rendering
a scene.

## Guard documentation build inputs

`scripts/docs_reference.py` (an MkDocs hook) raises `PluginError` rather than
silently producing a broken or unsafe page: on a reserved-path collision
between a generated API page and a real doc, on a missing `API Reference`
navigation entry, and on an API-asset reference that uses a backslash,
absolute path, or `..` traversal segment.

## Why this matters for changes here

When you add a new constructor argument, CLI option, or exported artifact
format, add the same kind of explicit, descriptive guard rather than letting
invalid input fail deep inside a computation. Tests exercise these guards
directly (see `tests/test_ga.py`, `tests/test_game.py`, and
`tests/test_web_model.py`); add a test alongside any new guard.
