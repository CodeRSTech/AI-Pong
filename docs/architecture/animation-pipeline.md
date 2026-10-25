# Animation pipeline

AI-Pong's documentation diagrams and example gameplay clips are generated from
the live model and from explicitly captured simulation traces, never invented
or hand-drawn. This page covers both the static diagram generator and the
gameplay-trace/animation workflow described in
[Neural network](neural-network.md#modular-gameplay-explanations).

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

## Capturing a gameplay trace

The capture command requires a specific checkpoint; it never selects the
latest file from `runs/` and does not change training or physics:

```powershell
python -m scripts.capture_gameplay_trace --checkpoint D:\models\selected-generation.pt --seed 42 --steps 90 --output D:\animation-work\gameplay-trace.json
```

`--validation-json` is optional and must contain a JSON object. If run,
generation, or validation provenance is unavailable, it is recorded as
`unknown`; the checkpoint hash and capture settings are still recorded. The
trace step delta defaults to the rendered game's configured timing and can be
overridden with `--step-delta`. Capture is bounded to 1-1000 decisions, uses
`Game.step`, and restores Python, NumPy, and PyTorch RNG state after
recording.

```powershell
python -m scripts.capture_gameplay_trace --checkpoint D:\models\selected-generation.pt --seed 42 --steps 90 --output D:\animation-work\gameplay-trace.json --run-id selected-run --generation 120 --validation-json D:\animation-work\validation-summary.json
```

`scripts/gameplay_scenes.py` defines the three Manim scene classes
(`observations`, `network-actions`, `actions-execution`) that read a captured
trace and one decision index; a command-line entry point to render them is
not yet part of this repository. Treat gameplay-trace capture as available
today and scene rendering as in progress.

## Future checkpoint-to-animation workflow

The capture tooling is ready for a future selected model. A
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
   seed, a bounded number of decisions (1-1000), and a trace output path. The
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
   `1 / (FPS x SPEED x STEPS_PER_FRAME)`. `--step-delta` changes the recorded
   simulation clock increment, not the physics' per-step movement rules.
   Document differences from the training configuration. Explanatory pauses
   and slow motion alter presentation timing only.
5. **Select one recorded decision.** Inspect the JSON's zero-based `decisions`
   array and choose an index that shows the intended lesson. Render all three
   modules with the same trace and decision index once a rendering entry point
   exists. Do not splice unrelated observations and actions or choose a
   flattering example as proof of general reliability. Trace validation
   rejects malformed dimensions/nonfinite data and checks recorded inference
   against the stored parameters.
6. **Preview, then render.** Once rendering is available, preview at low
   quality in a private output directory before producing final 1080p/60fps
   assets. Inspect positions, normalization, signed colors, output thresholds,
   actual displacement, annotation readability and causal continuity.
7. **Publish deliberately.** The three planned modules are
   `gameplay-observations.mp4`, `network-actions.mp4`, and
   `actions-execution.mp4`. Add controlled video playback, descriptive text,
   and corresponding poster/fallback assets in the docs or portfolio only
   after approval, with the checkpoint hash and example provenance recorded
   alongside them and limitations described. Rendering must not commit, push,
   or integrate new videos automatically.

For example, attach provenance while capturing:

```powershell
python -m scripts.capture_gameplay_trace --checkpoint D:\models\selected-generation.pt --seed 42 --steps 90 --output D:\animation-work\gameplay-trace.json --run-id selected-run --generation 120 --validation-json D:\animation-work\validation-summary.json
```

A reported two-hour training run produced a player scoring below the CPU;
that observation is not a diagnosis, and no suitable performance
demonstration is assumed imminent. Trained gameplay publication remains
deferred. Seeded test checkpoints may verify capture and preview layout but
are not trained gameplay.
