# Documentation style

These are the conventions this repository's documentation already follows.
Match them when adding or editing pages instead of introducing a new style.

## Structure and navigation

- Every page is reachable from `mkdocs.yml`'s `nav`. When you add a page,
  add it to the nav in the same change; `mkdocs build --strict` (run
  locally before opening a documentation PR, see
  [development](development.md)) fails on files outside the nav or on
  broken internal links.
- Group related pages into folders (`architecture/`, `development/`,
  `releases/`) with a landing `index.md` or topic-named files, rather than one
  long page. Cross-link between them with relative Markdown links
  (`[text](../other-folder/page.md)`), not absolute site URLs.
- The API reference under `api/` (nav label "API Reference") is generated at
  build time by `scripts/docs_reference.py` from docstrings in `src/`. Do not
  hand-edit those pages or add files under `docs/api/` or `docs/reference/`;
  edit the source docstrings instead.

## Headings are stable anchors

Headings are linked to from other pages and from `README.md` (for example
`docs/architecture/genetic-algorithm.md#selection-crossover-and-mutation`).
Treat an existing heading's exact text as part of the public contract: if you
reword or move it, update every internal link and README anchor that points
at it in the same change.

## Prose conventions

- Use MkDocs Material admonitions for supplementary notes and callouts, for
  example `???+ tip "Title"` (collapsible, expanded by default) or
  `???+ info "Title"`, followed by an indented body.
- Use Markdown tables for structured, scannable data: CLI settings and their
  defaults, fitness terms and their purpose, scene names and what they
  explain. Prefer a table over a bulleted list once there are two or more
  consistent columns of information.
- Use fenced code blocks with an explicit language tag matching the shell the
  command targets: ```bash``` for cross-platform or POSIX examples,
  ```powershell``` for Windows-specific examples (this repo's primary
  development platform), and ```console``` for the diagram generator, which
  is invoked the same way everywhere.
- Mermaid diagrams are enabled via `pymdownx.superfences` in `mkdocs.yml`
  (fence name `mermaid`) for future use; prefer a generated SVG (see
  [animation pipeline](../architecture/animation-pipeline.md)) when the
  diagram should reflect the current source of truth, such as the network
  topology or crossover masks.

## Be explicit about what exists versus what is planned

Several docs describe tooling that is partially built, such as the gameplay
animation pipeline. Label forward-looking content clearly ("planned, not
delivered") rather than describing it as available, and update the label the
moment the capability ships. Do not present a seeded or placeholder example
as a trained-model performance claim.
