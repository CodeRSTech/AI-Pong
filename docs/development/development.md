# Development

This page covers local setup and the day-to-day commands used while working
on AI-Pong: installing dependencies, running tests, training, and
regenerating generated documentation assets.

## Set up a development environment

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux), then install the project with
its development extras (`pytest`, `pytest-cov`, `ruff`):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch may need a CPU-only wheel if you do not have CUDA; see the
[PyTorch install selector](https://pytorch.org/get-started/locally/).

## Run the tests and linter

```bash
ruff check .
pytest --cov=src --cov-report=term-missing
```

`.github/workflows/ci.yml` runs the same two commands on Linux and Windows
across Python 3.10 and 3.12. Some tests are platform-gated: the
deployment-script tests in `tests/test_docs_deployment.py` and
`tests/test_docs_deployment_platform.py` require Bash and are skipped on
Windows, and the browser-parity tests in `tests/test_web_demo.py` skip
explicitly when no Chromium-family browser is available (override the
executable with `PONG_TEST_BROWSER`).

CI also includes a `docs` job that builds the MkDocs site with `mkdocs build`
so a structurally broken build (bad YAML, a missing nav target, a plugin
failure) fails CI. Run `mkdocs build --strict` locally before opening a
documentation pull request: it also fails on dangling internal links and
files missing from the nav, which is stricter than CI. Note that `--strict`
currently also reports pre-existing mkdocstrings docstring-format warnings in
`src/components/rectangle.py` and `src/utils/functions.py` unrelated to
Markdown content; those are a known, separate cleanup and do not block CI.

## Run training

Rendered training opens one Arcade window for the full evolution loop:

```bash
python -m src.main
```

Headless training runs the same simulation without a window and reports
progress periodically:

```bash
python -m src.train
```

Both commands run until interrupted (`Ctrl+C`) unless `--generations` sets a
limit; see `--help` on either command for population size, seed, timeout, and
validation-suite options. To watch the newest saved elite checkpoint:

```bash
python -m src.tester
```

See [training tips](../training.md) for tuning details and
[architecture](../architecture/index.md) for how the evolution loop, neural
network, and simulation fit together.

## Regenerate generated documentation assets

The neural-network and crossover diagrams in the architecture docs are
generated from the live model, not hand-drawn, and must be regenerated and
committed whenever the network or crossover logic changes:

```bash
python -m scripts.generate_diagrams
```

The source-driven gameplay-trace workflow (`python -m
scripts.capture_gameplay_trace`) captures an explicit checkpoint's decisions
for later animation; see
[animation pipeline](../architecture/animation-pipeline.md) for the full
workflow, including which parts (trace capture) are implemented today versus
planned (the Manim scene render entry point).

If you change the web-demo checkpoint, regenerate its exported JSON too:

```bash
python -m scripts.export_web_model --checkpoint assets/model.pt --output assets/model.json
```

## Preview the documentation site locally

Install the docs-only dependencies from `requirements-docs.txt` (kept
separate from the runtime `requirements.txt` so building the site does not
require installing PyTorch, Arcade, or any other runtime dependency):

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

`mkdocs serve` watches `src/` as well as `docs/`, because the API reference
pages under `api/` are generated from source docstrings by
`scripts/docs_reference.py`. Before opening a pull request that changes
documentation, run `mkdocs build --strict` locally to catch broken links or
navigation entries before CI does.
