# Documentation overview

AI-Pong evolves neural networks with a genetic algorithm and uses Pong simulations to evaluate them. These docs cover how the project works, how to train and extend it, and the Python components behind its core behavior.

## Learn the project

- [Architecture](architecture/index.md) explains population evaluation, neural networks, simulation, and rendering.
- [Coordinate system](coordinates.md) describes simulation coordinates and their relationship to Arcade rendering.
- [Fitness function](fitness.md) explains how candidate players are scored.
- [Web demo](web-demo.md) covers browser inference and exporting model checkpoints.

## Run and extend AI-Pong

- [Training](training.md) covers training options, timeouts, population size, and mutation.
- [Animation pipeline](architecture/animation-pipeline.md) describes the source-driven animation workflow.
- [Roadmap](ROADMAP.md) records current project direction.
- [Release notes](releases/1.4.0.md) document the 1.4.0 CLI, saved-run formats, and migration notes.

## Reference

The [API reference](reference/api/index.md) is generated from the core Python modules with mkdocstrings.

For setup instructions and a short project introduction, see the [home page](index.md).
