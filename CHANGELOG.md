# Changelog

Notable project changes are recorded here. This changelog may not include every change.

## Unreleased

- Added a headless/rendered training CLI with configurable population, generations, seed, timeout, output root, and simulation timing.
- Saved settings, per-generation metrics, and checkpoints in timestamped run directories.
- Updated the viewer to locate the latest run's elite model or accept an explicit checkpoint path.
- Improved crossover to always select a non-empty contiguous neuron range and inherit the selected neurons' outgoing weights.
- Made simulation steps per rendered frame configurable.
- Added separate rendered and headless training commands, progress updates, and randomized elite validation.
- Refreshed the rendered court and neural-network panel with higher-contrast colors and distinct paddles.

## 1.3.0

- Migrated from Pygame to Arcade.
- Replaced the NumPy perceptron with a PyTorch neural network and updated the network structure and fitness function.
- Added a visual representation of the neural network to the UI.
- Added borders to the ball and paddles.
- Batched inference across the population for improved performance.
- Displayed only the elite player's play zone and added configurable frame skipping.
