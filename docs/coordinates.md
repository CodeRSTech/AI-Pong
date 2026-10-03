# Coordinate system

## Simulation: origin at the top left

Game logic uses a y-down coordinate system: `(0, 0)` is the top-left corner, x increases to the right, and y increases downwards. Entity positions are their centers. For the ball, positive `speed.y` moves down and negative `speed.y` moves up.

Rectangles derive their edges from the center:

- `left = pos_x - width / 2`
- `right = pos_x + width / 2`
- `top = pos_y - height / 2`
- `bottom = pos_y + height / 2`

## Rendering: origin at the bottom left

Arcade uses y-up coordinates. Entity rendering flips the vertical position against the window height: a center at simulation y-coordinate `y` is rendered at `window_height - y`. For rectangles, each y-down edge is similarly converted before drawing. Collision checks remain entirely in simulation coordinates.
