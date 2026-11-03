# Architecture

This section explains how AI-Pong evolves and evaluates its neural-network
players. It is organized by concern:

- [Genetic algorithm](genetic-algorithm.md) covers selection, crossover, and
  mutation.
- [Neural network](neural-network.md) covers the player model, its
  activations, and the modular gameplay-explanation tooling.
- [Simulation and rendering](simulation-rendering.md) covers Arcade, batched
  inference, and audio.
- [Animation pipeline](animation-pipeline.md) covers regenerating the static
  diagrams and the source-driven gameplay trace/animation workflow.

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

???+ tip "Population size and duration"
   The default population is 200. Training continues until interrupted unless
   `--generations` sets a limit.
