# Copilot coding agent instructions for AI-Pong

AI-Pong evolves a population of small PyTorch neural networks with a genetic
algorithm to play Pong, rendered with Arcade. Read this before making changes.

## Repository layout

- `src/` — the application package.
  - `main.py` / `train.py` / `tester.py` — entry points: rendered training,
    headless training, and elite playback (`python -m src.<name>`).
  - `ga/` — `ga_core.py` (GeneticAlgorithm: selection, crossover, mutation,
    checkpointing), `network.py` (`NeuralNet`, `BatchedPopulationBrain`),
    `player.py` (`IndividualPlayer`, observation/action labels),
    `validation.py` (`EliteValidator`, randomized post-generation scenarios).
  - `components/` — Arcade game entities (`ball.py`, `paddle.py`,
    `playzone.py`, `rectangle.py`, `geometry.py`, `colors.py`).
  - `game.py` — `Game`: owns play zones, batches observations/actions each
    step, drives the rendered and headless update loops.
  - `variables.py` — global tunables (`WIDTH`, `HEIGHT`, `FPS`, `TIME_OUT`,
    `SPEED`, `STEPS_PER_FRAME`, `PANEL_WIDTH`).
  - `utils/` — `functions.py` (crossover mask helpers), `logger.py` (loguru
    wrapper).
- `scripts/` — standalone CLI/tooling modules run with `python -m
  scripts.<name>`: `generate_diagrams.py` (regenerates committed SVGs in
  `docs/images/`), `export_web_model.py` (exports a checkpoint to
  `assets/model.json` for the browser demo), `capture_gameplay_trace.py` /
  `gameplay_trace.py` (deterministic gameplay-trace capture),
  `gameplay_scenes.py` (Manim scene definitions — no CLI render entry point
  exists yet), `docs_reference.py` (MkDocs hook that generates the API
  reference from docstrings).
- `tests/` — pytest suite, one file per area; `conftest.py` seeds Python,
  NumPy, and PyTorch RNGs before every test.
- `docs/` — MkDocs (Material theme) site. `docs/architecture/` and
  `docs/development/` are split into topic pages; keep `mkdocs.yml`'s `nav`
  in sync with any page you add, move, or rename.
- `index.html` + `assets/` — the static GitHub Pages homepage with the
  in-browser model demo (separate from the MkDocs site, deployed together).

## Conventions to follow

- Python 3.10+. Formatting/linting is `ruff check .` with `line-length = 120`
  (`pyproject.toml`); the configured rule set (`E9, F63, F7, F82`) catches
  syntax and undefined-name errors, not general style.
- Dependencies: runtime deps are in `requirements.txt` and
  `pyproject.toml[project.dependencies]`; development extras
  (`pytest`, `pytest-cov`, `ruff`) are the `dev` extra; docs-build-only
  dependencies live in `requirements-docs.txt` (mirrored by the `docs` extra
  in `pyproject.toml`) and are installed separately from the runtime
  dependencies in CI and the Pages deploy workflow — do not merge them back
  into `requirements.txt`.
- Guard public entry points explicitly: raise `ValueError`/`RuntimeError`
  with a descriptive message for invalid arguments or empty/invalid state
  instead of letting an error surface deep in a computation. See
  `docs/development/defensive-guards.md` for the existing pattern and where
  it is used (`game.py`, `ga/ga_core.py`, `ga/validation.py`,
  `utils/functions.py`, `scripts/export_web_model.py`,
  `scripts/docs_reference.py`).
- Generated artifacts are committed, not built on demand: `docs/images/*.svg`
  (from `scripts/generate_diagrams.py`) and `assets/model.json` (from
  `scripts/export_web_model.py`). Regenerate and commit both when the
  network/crossover logic or the exported checkpoint changes.
- Simulation uses a y-down coordinate system (see `docs/coordinates.md`);
  Arcade rendering flips y for display only. Keep collision/physics code in
  simulation coordinates.
- Mutation/crossover rates, elite factor, and similar GA tuning values are
  implementation constants in `src/ga/ga_core.py` and `src/ga/network.py`,
  not CLI options — don't add CLI flags for them without discussion.

## Running checks before committing

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest --cov=src --cov-report=term-missing
```

If you changed anything under `docs/`, `mkdocs.yml`, or navigation-affecting
files, also validate the docs build (requires `requirements-docs.txt`); CI's
`docs` job runs a plain `mkdocs build`, and running `--strict` locally also
catches broken links/nav entries (it also reports some pre-existing,
unrelated docstring warnings — see `docs/development/development.md`):

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict --site-dir /tmp/site
```

Some tests are platform-gated and will skip rather than fail: the deployment
script tests (`tests/test_docs_deployment*.py`) require Bash and skip on
Windows; browser-parity tests (`tests/test_web_demo.py`) skip when no
Chromium-family browser is available. CI (`.github/workflows/ci.yml`) runs
the full test matrix (Ubuntu + Windows, Python 3.10 + 3.12) plus a `docs`
job; `.github/workflows/docs.yml` builds and deploys the MkDocs site and the
static homepage to GitHub Pages on pushes to `main`.

## Documentation

Update the relevant guide under `docs/` alongside any user-facing or
CLI-affecting change — see `docs/development/documentation-style.md` for this
repository's documentation conventions (admonitions, tables, stable heading
anchors, labeling planned-vs-implemented features honestly). Headings are
linked to from `README.md` and other docs pages by exact anchor text; update
every link if you reword or move a heading.
