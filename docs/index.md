# AI-Pong documentation

AI-Pong evolves a small neural network population to play Pong. The Python simulation and genetic algorithm use PyTorch, while Arcade renders one representative play zone and its network.

## Get started

Install the project requirements and launch training from the repository root:

```bash
python -m pip install -r requirements.txt
python -m src.main
```

`python -m src.main` opens the rendered training runner. For headless runs with periodic progress reports, use `python -m src.train`. Both run continuously until interrupted unless `--generations` sets a limit. To view the most recently saved elite model, run `python -m src.tester`.

## Publishing this site

The deployment workflow publishes the documentation on pushes to `main` and can also be run manually. Before its first run, a repository maintainer must set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**.

## Guides

- [Architecture](architecture.md): population evaluation, neural network, simulation, and rendering.
- [Fitness function](fitness.md): what the genetic algorithm rewards.
- [Training tips](training.md): timeouts, population size, and mutation.
- [Coordinate system](coordinates.md): simulation coordinates and Arcade rendering.
- [Roadmap](ROADMAP.md): current project direction.
