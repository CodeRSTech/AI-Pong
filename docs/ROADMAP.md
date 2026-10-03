# Roadmap

This roadmap summarizes the project's planned work. Priorities may change as the project evolves.

| Phase | Focus | Status | Related guide |
| --- | --- | --- | --- |
| 0 | Foundation: packaging, tests, and CI | Complete | [CI workflow](https://github.com/CodeRSTech/AI-Pong/blob/main/.github/workflows/ci.yml) |
| 1 | Documentation and repository hygiene | Complete | [Contributing](https://github.com/CodeRSTech/AI-Pong/blob/main/CONTRIBUTING.md) |
| 2 | Reproducible runs, correctness, and a training CLI | Complete | [Training tips](training.md) |
| 3 | Arcade visuals and effects | Complete | [Architecture](architecture.md) |
| 4 | Sound effects | Planned | [Architecture](architecture.md) |
| 5 | Lightweight browser demo | Planned | [README feature tracker](https://github.com/CodeRSTech/AI-Pong#feature-tracker) |
| 6 | Release process and distribution | Planned | [Changelog](https://github.com/CodeRSTech/AI-Pong/blob/main/CHANGELOG.md) |

The evolutionary design and current gameplay rules are described in the [architecture](architecture.md) and [fitness](fitness.md) guides. Proposed changes should be discussed in an issue before substantial work begins.

## Follow-up issue candidates

The following code-level TODOs were moved here for filing as focused GitHub issues:

- Revisit CPU paddle tracking so its response is more adaptive to the ball's distance and trajectory.
- Prevent the ball from sticking to a paddle after a collision by resolving overlap before the next update.
- Constrain respawn coordinates to a centered safe area that keeps the ball in play.
