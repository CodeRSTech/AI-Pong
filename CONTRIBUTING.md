# Contributing

Thanks for helping improve AI-Pong. Bug reports, documentation improvements, and focused code changes are welcome.

## Development setup

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the project and development tools:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

PyTorch may be installed separately with a CPU-only wheel if you do not need CUDA; see the [PyTorch install selector](https://pytorch.org/get-started/locally/).

## Before opening a pull request

- Run `ruff check .` and `pytest`.
- Keep changes focused and include or update tests when behavior changes.
- Update relevant documentation, including configuration and user-facing behavior.
- Describe the problem, the approach, and any manual verification in the pull-request description.

Use the repository's issue templates for bug reports and feature requests. Please do not include private checkpoints or other large generated model files unless they are necessary to explain the change.
