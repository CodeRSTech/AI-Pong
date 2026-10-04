# Editing and rendering the Manim animations

This guide is for making small visual changes to the two existing explanatory
videos, previewing them safely, and only then replacing the published files.
You do not need to change the game or train a model to adjust titles, layout,
colors, pacing, or other presentation details.

## Find the part you want to change

Both videos are scenes in [`scripts/animate_diagrams.py`](https://github.com/CodeRSTech/AI-Pong/blob/main/scripts/animate_diagrams.py):

| Video | Scene class | What it explains |
| --- | --- | --- |
| `neural-network.mp4` | `NeuralNetworkAnimation` | A fixed input entering the network, one neuron's weighted sum, activations, and the resulting paddle action. |
| `two-point-crossover.mp4` | `CrossoverAnimation` | Parent copies, the selected interval, reciprocal child inheritance, and parameter ownership in the child networks. |

For most visual edits, start in the matching scene's `construct` method:

- Change titles, captions, and labels by editing the nearby text strings.
- Adjust layout by changing the coordinates passed to `move_to`, or spacing
  passed to `next_to` and `arrange`. Scene coordinates are relative to the
  center of the frame; small changes are easier to judge in a preview.
- Adjust text size using the size argument to `caption` or `text`.
- Change an animation's speed by adjusting its `run_time`; change a pause
  between beats with `self.wait(...)`.
- Shared colors, backgrounds, and reusable visual components live in
  [`scripts/animation_theme.py`](https://github.com/CodeRSTech/AI-Pong/blob/main/scripts/animation_theme.py). A change there
  may affect both videos, so preview both before keeping it.

The numerical examples are assembled in
[`scripts/animation_data.py`](https://github.com/CodeRSTech/AI-Pong/blob/main/scripts/animation_data.py) from the current
PyTorch model and crossover functions. Prefer changing the explanatory copy or
layout rather than typing replacement values into the scene: the displayed
values and diagrams are intended to stay connected to the source behavior.
If you change a mathematical explanation or crossover detail, check that its
labels, values, and animation still agree.

## Make one small change first

For example, to rename the network video, find its `terminal_header` call in
`NeuralNetworkAnimation.construct` and edit the title string:

```python
"AI-Pong: observations to paddle actions",
```

Save the file, render that scene to a separate preview folder using the
instructions below, and watch the resulting MP4. If it looks right, continue
with the next change. A useful first pass is to write down the one idea each
video should teach, then simplify crowded text, improve spacing, or slow down
only the transitions that are hard to follow. Making one kind of change at a
time makes it much easier to tell what helped.

## Set up Manim

Manim is optional; the game and the rest of the project do not need it. From
the repository root, install it in the Python environment you will use to
render. For the project's Windows virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install manim
```

The scenes use ordinary text, not TeX. The renderer uses installed Fira Code
when available, otherwise Consolas; it does not install or bundle fonts. If
neither is installed, install one of those fonts and render again.

## Preview without replacing the published videos

The renderer is [`scripts/render_animations.py`](https://github.com/CodeRSTech/AI-Pong/blob/main/scripts/render_animations.py).
It accepts the scene names `neural-network` and `two-point-crossover`, and
writes MP4s to the directory given by `--output-dir`. Always use a separate
preview directory while iterating:

```powershell
.\.venv\Scripts\python.exe -m scripts.render_animations `
  --scenes neural-network `
  --quality l `
  --output-dir D:\animation-work\preview
```

Open `D:\animation-work\preview\neural-network.mp4` and watch the whole clip.
For the crossover video, change the scene name:

```powershell
.\.venv\Scripts\python.exe -m scripts.render_animations `
  --scenes two-point-crossover `
  --quality l `
  --output-dir D:\animation-work\preview
```

You can render both scenes in one command by listing both names after
`--scenes`. The quality choices are `l` (480p/15 fps), `m` (720p/30 fps), and
`h` (1080p/60 fps). Use `l` for quick layout and timing previews; a higher
quality setting is not a substitute for reviewing the content.

If you omit `--output-dir`, the renderer writes to
`docs/images/animations/`, which contains the published videos. Avoid that
default while experimenting so a preview cannot replace those files.

## Review and publish

Before keeping an edit, review the preview at normal playback size and check:

- Text fits inside the frame, stays readable, and is not covered by other
  objects.
- Each step is on screen long enough to understand; particles and labels reach
  the intended objects.
- The order of events is clear and the colors still communicate the intended
  meaning.
- Statements and displayed values agree with the model or crossover behavior.

Once the preview is approved, render the final clips to a separate folder:

```powershell
.\.venv\Scripts\python.exe -m scripts.render_animations `
  --scenes neural-network two-point-crossover `
  --quality h `
  --output-dir D:\animation-work\final
```

Watch the final files in `D:\animation-work\final` too. Then copy only the
approved MP4s into `docs/images/animations/` if you intend to update the
published videos. The renderer does not update documentation or commit files
for you.

## Common problems

- **`No module named manim`:** install Manim in the same Python environment
  used to run the render command.
- **Missing font message:** install Fira Code or Consolas, then rerun.
- **Text is clipped after moving it:** render a low-quality preview and adjust
  the position or reduce the text size; changing one coordinate at a time
  makes the cause easier to spot.
- **A shared style edit changes both clips:** this is expected for helpers in
  `animation_theme.py`; render and review both scenes.

For the videos' current explanations and limitations, see the
[architecture guide](architecture.md#neural-network-animation).
