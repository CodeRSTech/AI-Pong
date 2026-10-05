# Homepage model demo

The homepage runs a live, watch-only Pong simulation in the browser. The saved
network controls the green/right paddle against the CPU's cyan/left paddle.
It does **not** train, mutate, or select a population, and the checkpoint's
performance is not a guarantee of reliability.

The browser uses the Python game's seven normalized observations, tanh/ReLU/
sigmoid layers, and two actions thresholded strictly above 0.5. Both active
actions cancel each other; neither active action leaves the paddle still.
Simulation uses the original 476 x 500 coordinates and is rotated only for
display in the existing CSS court.

## Export a new checkpoint

From the repository root, with the project's Python dependencies installed:

```bash
python -m scripts.export_web_model --checkpoint assets/model.pt --output assets/model.json
```

After replacing `assets/model.pt`, rerun this command and include both the
checkpoint and the generated JSON in your change. For a checkpoint elsewhere:

```bash
python -m scripts.export_web_model --checkpoint src/elite_model.pt --output assets/model.json
```

The exporter loads on CPU using PyTorch's `weights_only=True`. It accepts only
the current state dict's six finite float32 tensors with the 7 -> 8 -> 6 -> 2
shapes. Full pickled models, other architectures, and invalid weights fail
explicitly. Activations are specified by the current architecture, not stored
in the `.pt` file; changing them requires updating the exporter and browser
implementation together.

JSON version 1 contains architecture, layer weights/biases/activations, and the
source checkpoint's SHA-256 hash. Export is deterministic and does not round
away float32 values. The browser loads `assets/model.json`, not the `.pt` file,
and needs no external ML runtime. JavaScript and PyTorch arithmetic may differ
slightly; raw outputs are compared within tolerance in the parity coverage.

## Preview locally

Serve the repository root over HTTP (opening `index.html` directly as a
`file:` URL will not reliably allow model fetching):

```bash
python -m http.server 8000
```

Open `http://localhost:8000/`. The demo autoplays, displays actual AI/CPU scores,
and re-serves after misses. Pause/resume is available; reduced-motion visitors
start paused. Hidden tabs and offscreen courts suspend without accumulating
simulation time or discarding a manual pause.

The browser uses 144 logical decisions per second for watchable playback,
independent of display refresh rate. This is deliberately slower than the
Python training runner's accelerated multi-step updates. No audio is played.
Missing or malformed JSON produces an explicit unavailable message and a
console error, not fake model-controlled gameplay. Without JavaScript the court
remains a static illustration.

GitHub Pages publishes the committed JSON, homepage scripts, and demo GIF.
Deployment does not install PyTorch or regenerate weights. Keep the committed
JSON in sync using the export command above.

## Coverage

`python -m pytest tests/test_web_model.py tests/test_web_demo.py` covers export
validation and browser/PyTorch parity. Browser coverage uses an installed
Chromium-family browser (Edge, Chrome, or Chromium), or the executable selected
by `PONG_TEST_BROWSER`; it skips explicitly when none is available. Existing
deployment tests exercise asset assembly on Linux with Bash.
