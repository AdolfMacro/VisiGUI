# VisiGUI

VisiGUI is a desktop application for exploring a deterministic procedural 3D
world with two hands, a keyboard, or a mouse-free camera setup. The desktop
application is the default experience. A terminal-based Python project
structure browser is retained behind `--cli`.

This document explains how the application works, including the coordinate
systems, camera matrices, movement equations, geometry generation, tracking,
and rendering pipeline. The implementation is in `src/visigui/`.

## 1. Features and controls

- Starts the full-screen PyQt6 world by default and attempts to use camera
  device 0 for hand tracking.
- The left hand starts movement when it is clenched into a fist. The fist's
  displacement from the pose where it was clenched sets horizontal and
  vertical travel; opening the fist stops the movement target.
- With the right hand, touching thumb and index continuously zooms out. As
  soon as they separate, zoom-in continues through the middle range and at
  maximum spread.
- Shows a live camera preview with landmarks, a red tracking square, and
  `Left HAND - MOVE` / `Right HAND - ZOOM` labels.
- Supports keyboard flight when running with `--demo`.
- Keeps the world, camera, and generated sector cache alive when one or both
  hands disappear.
- Generates geometry deterministically from a world seed and integer sector
  coordinates. Geometry is cached and reused rather than rebuilt each frame.
- Includes `--cli`, a static Python project-structure explorer. It parses
  source into an AST and does not import or execute the project being scanned.

### Controls

| Input | Action |
|---|---|
| Left hand | Clench a fist to engage movement; move the fist to steer horizontally/vertically; open it to stop. |
| Right hand | Touch thumb/index and hold to zoom out; as they separate, zoom continuously in. No intermediate gap stops zoom-in. |
| `W` / `Up` | Move upward. |
| `S` / `Down` | Move downward. |
| `A` / `Left` | Move left. |
| `D` / `Right` | Move right. |
| `E` / `PageUp` | Move forward. |
| `Q` / `PageDown` | Move backward. |
| `Esc` | Exit. |
| `F11` | Toggle full-screen mode. |

Keyboard input adds to hand input. Camera speed limits still apply to the
combined target.

## 2. Installation and running

### Requirements

- Python 3.10 or newer.
- Linux for the supplied `install.sh` installer.
- A desktop session and an OpenGL 3.3-compatible driver for the 3D renderer.
- A camera for tracking mode.
- The MediaPipe Hand Landmarker model for camera tracking. The installer
  downloads it and verifies its pinned SHA-256 digest.

The project depends on NumPy, OpenCV, MediaPipe Tasks, PyQt6, PyOpenGL, and
Pytest for development tests.

### User installation on Linux

```bash
chmod +x install.sh
./install.sh
```

This creates a virtual environment under
`~/.local/share/visigui/venv/`, installs a staged copy of the package, downloads
the hand model, and exposes `visigui` in `~/.local/bin`. It does not use
`sudo`, install into system Python, or build in the source checkout. Start a
new shell or reload its configuration if `~/.local/bin` is not in `PATH`.

To install without downloading the hand model:

```bash
./install.sh --no-model
~/.local/share/visigui/venv/bin/python -m visigui.download_model
```

The installer supports these optional environment variables:

| Variable | Purpose |
|---|---|
| `VISIGUI_INSTALL_DIR` | User installation directory; defaults to `$XDG_DATA_HOME/visigui` or `~/.local/share/visigui`. |
| `VISIGUI_BIN_DIR` | Executable directory; defaults to `~/.local/bin`. |
| `PYTHON` | Python 3 executable; defaults to `python3`. |

Installations created before the rename are left untouched. If an older
launcher still runs, verify `command -v visigui` and inspect the installed
environment under `~/.local/share/visigui/venv/`.

### Development install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m visigui.download_model
```

### Run modes

```bash
# Default: full-screen 3D world with camera tracking
visigui

# Same application without a camera; use the keyboard
visigui --demo

# Pick a camera device; zero is the default
visigui --camera-index 1

# Keep tracking active but hide the camera image
visigui --no-camera-view

# Retained terminal Python project explorer
visigui --cli -p /path/to/python-project

# Print available options
visigui --help
```

Equivalent module invocations include `python -m visigui` and
`python -m visigui --demo`. The `--frames N` option selects the finite terminal
render path and is mainly useful for smoke testing.

The camera model defaults to
`~/.cache/visigui/hand_landmarker.task`. Override it with `--model PATH`.
Native Qt, OpenCV, and MediaPipe diagnostics are written to
`~/.cache/visigui/runtime.log` by default; use `--log-file PATH` to change the
destination.

## 3. High-level architecture

The camera and world run on separate Qt worker threads. The camera worker
captures and processes frames, then publishes immutable hand state. The world
worker consumes the latest hand state and keyboard target, updates simulation,
and publishes a world frame. The GUI thread displays status, camera preview,
hand drawings, and OpenGL output.

```text
Camera frame
    │
    ├─ BGR → RGB, horizontal mirror
    ├─ MediaPipe Hand Landmarker (up to 2 hands, 21 landmarks each)
    ├─ temporal left/right assignment
    ├─ adaptive landmark filter + finger/gesture analysis
    └─ TwoHandState
          ├─ left hand → planar navigation velocity
          └─ right pinch → depth velocity
                    │
Keyboard ───────────┤
                    ▼
            GenerativeWorld.update
             ├─ TwoHandWorldController
             ├─ CameraController
             └─ SectorStreamer
                    │
                    ▼
               WorldFrame
                 ├─ PyQt HUD and hand widgets
                 └─ OpenGL sector renderer
```

Important modules:

| Module | Responsibility |
|---|---|
| `src/visigui/app.py` | CLI argument parsing and dispatch between the GUI and retained terminal mode. |
| `src/visigui/camera/` | Camera capture and camera errors. |
| `src/visigui/vision/detector.py` | MediaPipe Tasks integration, hand observations, and finger analysis. |
| `src/visigui/vision/tracking.py` | Stable left/right assignment, hand state, pinch distance, and adaptive landmark filtering. |
| `src/visigui/gui/workers.py` | Camera and world worker lifecycles and thread handoff. |
| `src/visigui/interaction/world_controller.py` | Converts hand states to bounded movement targets. |
| `src/visigui/world/camera.py` | Camera basis, movement dynamics, position, and sector coordinate. |
| `src/visigui/world/procedural.py` | Deterministic sector generation and bounded LRU cache. |
| `src/visigui/world/simulation.py` | Persistent world state and immutable render-frame assembly. |
| `src/visigui/renderer/opengl_world.py` | Frustum culling, shaders, GPU buffers, and OpenGL draws. |
| `src/visigui/gui/world_window.py` | Full-screen layout, HUD, controls, and shutdown. |
| `src/visigui/project/`, `src/visigui/terminal/` | Retained static source explorer and terminal UI. |

## 4. Coordinate systems and vector conventions

### 4.1 World coordinates

The 3D world uses a right-handed Cartesian coordinate system:

- `+X`: world right.
- `+Y`: world up.
- `-Z`: the initial camera forward direction.
- Positions and directions are represented by three-component vectors.

The OpenGL shader treats positions as column vectors in homogeneous
coordinates:

```text
p_world = [x, y, z, 1]ᵀ
```

The final clip-space point is calculated in this order:

```text
p_clip = P · V · M · p_world
```

where `M` is the model matrix, `V` is the view matrix, and `P` is the
projection matrix. Sector vertices are already generated in world coordinates,
so the current model transform is the identity matrix `M = I`. The renderer
uploads `P · V` as `u_mvp`.

After the vertex shader writes `gl_Position`, OpenGL performs perspective
division:

```text
p_ndc = [x_clip / w_clip, y_clip / w_clip, z_clip / w_clip]
```

The normalized device coordinate (NDC) cube is clipped to `x,y,z ∈ [-1, 1]`.
The OpenGL viewport transform then maps NDC x/y to framebuffer pixels.

### 4.2 Camera orientation and basis

The camera stores position `C = (x, y, z)`, yaw `ψ`, and pitch `θ` in radians.
Yaw rotates around world Y; pitch is clamped to `[-1.35, 1.35]` radians to
avoid the singularity of looking exactly vertically. The unit forward vector
is:

```text
F = (-sin ψ cos θ, sin θ, -cos ψ cos θ)
```

The camera-right vector and camera-up vector are:

```text
R = (cos ψ, 0, -sin ψ)
U = R × F
```

At the initial orientation `ψ = θ = 0`, `F = (0, 0, -1)`,
`R = (1, 0, 0)`, and `U = (0, 1, 0)`.

For a world point `X`, its camera-relative coordinates are dot products:

```text
x_camera = (X - C) · R
y_camera = (X - C) · U
z_camera = (X - C) · F
```

The `lookAt` view transform expresses the same operation as a 4×4 matrix. In
row form, for camera axes `R`, `U`, `-F`:

```text
V = [ Rₓ   Rᵧ   R_z   -R·C ]
    [ Uₓ   Uᵧ   U_z   -U·C ]
    [-Fₓ  -Fᵧ  -F_z    F·C ]
    [  0    0    0      1  ]
```

The last row preserves homogeneous coordinates. Matrix APIs may store these
values column-major in memory; that storage detail does not change the
mathematical transform.

### 4.3 Perspective projection matrix

The renderer uses a vertical field of view of `68°`, aspect ratio
`a = viewport_width / viewport_height`, near plane `n = 0.08`, and far plane
`f = 210`. Define:

```text
q = 1 / tan(FOV_y / 2)
```

The standard OpenGL perspective matrix used by `QMatrix4x4.perspective` is:

```text
                 [ q/a    0          0                 0             ]
                 [  0     q          0                 0             ]
P =              [  0     0   -(f+n)/(f-n)      -2fn/(f-n)           ]
                 [  0     0         -1                 0             ]
```

The matrix scales x by both the field of view and viewport aspect ratio, so
objects do not become stretched when the window changes shape. Its third and
fourth rows map camera depth to the OpenGL clip-depth range. The near and far
planes reject geometry outside the useful viewing distance and provide depth
buffer precision.

The complete transform for a sector vertex `p` is therefore:

```text
p_clip = P · V · I · p_world
```

The identity `I` is omitted in the renderer's uniform because sector vertices
are already in world space. The shader computes `u_mvp * vec4(a_position, 1)`.

## 5. Hand tracking and interaction mathematics

### 5.1 Landmark coordinates and identity

MediaPipe returns up to two hands, each with 21 landmarks. Landmark x/y values
are normalized image coordinates; z is MediaPipe's relative depth coordinate.
The input image is mirrored before detection so its orientation matches the
preview. Each observation also carries a handedness label and confidence
metadata when MediaPipe provides them.

MediaPipe result order is not used as identity. The tracker assigns
observations to logical `Left` and `Right` tracks, preferring a valid
handedness label and otherwise minimizing distance to the previous palm
centers. If an assignment conflicts with a known handedness label, it incurs
an additional cost. A track tolerates up to eight missed frames before it is
reset.

Each hand's 21 x/y landmark pairs pass through a One Euro-style adaptive low
pass filter. For each scalar coordinate, the filter estimates velocity:

```text
dᵢ = (xᵢ - xᵢ₋₁) / Δt
```

It low-pass-filters that derivative to obtain `d̂`, then selects a position
cutoff that increases with speed:

```text
f_c = f_min + β · |d̂|
α(f, Δt) = 1 / (1 + 1/(2π f Δt))
x̂ᵢ = α xᵢ + (1 - α) x̂ᵢ₋₁
```

Small slow movements receive more smoothing to reduce jitter; fast movements
raise the cutoff so the hand display and controller respond with less lag.

The defaults are `f_min = 1.4 Hz`, `β = 0.2`, and derivative cutoff
`f_d = 1.0 Hz`.

### 5.2 Finger extension analysis

When MediaPipe world landmarks are available, finger analysis uses them;
otherwise x/y are scaled by image width/height and z by image width to give
the three axes comparable pixel-like units. The palm direction is the unit
vector from wrist to middle-finger MCP. For each non-thumb finger, the
analyzer checks that adjacent bone segments point in broadly the same
direction, then projects the MCP-to-tip vector onto the palm direction. A
finger is marked extended when:

```text
cosine(segment₁, segment₂) ≥ 0.25 for both adjacent segment pairs
tip_projection ≥ max(0.40 · palm_length, 0.58 · total_finger_length)
```

The thumb uses its three bone directions and lateral distance from the index
MCP. It must have adjacent segment cosine at least `0.05`, and its tip must
extend laterally by more than `max(0.18 · palm_width, 10⁻⁶)`. These geometric
rules are scale-aware heuristics, not a learned classifier. Gesture
recognition consumes the resulting five boolean finger states and confidence;
the right-hand depth control separately consumes the continuous pinch ratio.

### 5.3 Left-hand fist navigation

Movement is enabled only while the left-hand gesture classifier reports a
fist. The first fist frame captures normalized palm position `(x₀, y₀)`; this
is the steering origin, not the camera's absolute position. While the fist is
held, the controller calculates:

```text
Δx = x - x₀
Δy = y - y₀
```

Horizontal steering uses `Δx`; vertical steering uses `-Δy`, since image y
increases downward while world y increases upward. The dead-zone thresholds
are `0.08` of the normalized image span.

Opening the fist clears the reference and zeros the movement target. The next
fist creates a fresh origin, so changing gestures or temporarily losing the
hand cannot cause a large position jump when tracking resumes.

For a signed input `v`, dead zone `d`, and velocity limit `L`, the
implementation calculates:

```text
m = max(0, (|v| - d) / (1 - d))
u = sign(v) · min(L, m · s · L)
```

where `s` is the configured sensitivity. The default navigation parameters
are `d = 0.08`, `s = 2.0`, and `L = 192`. The `min` caps the result; because
the sensitivity is greater than one, the cap is reached before the input
reaches the edge of its normalized range.

### 5.4 Right-hand endpoint zoom

The controller measures the straight-line gap between the thumb tip and index
tip, divided by palm width (distance between index MCP and pinky MCP). When
MediaPipe world landmarks are available, both distances are Euclidean 3D
distances:

```text
g = ||thumb_tip - index_tip||₂
    / ||index_MCP - pinky_MCP||₂
```

Using 3D points reduces gap changes caused merely by rotating the hand. If
world landmarks are missing, the tracker falls back to x/y image distance.
Dividing by palm width makes the values less dependent on how close the hand
is to the camera; it remains a relative gesture measurement, not a physical
distance.

The normalized gap is smoothed with an exponential filter (`τ = 0.035 s`) and
must reach either endpoint for `0.12 s`:

| Gap state | Entry | Exit (hysteresis) | Intent | Depth target |
|---|---:|---:|---|---:|
| Thumb and index touching | `g ≤ 0.12` | `g > 0.20` | `ZOOM_OUT` | `-16` units/s |
| Fingers separated | `g > 0.12` on initial detection; above `0.20` when leaving contact | `g ≤ 0.12` | `ZOOM_IN` | `+16` units/s |
| Between contact thresholds | `0.12 < g ≤ 0.20` | Retains last stable intent | No stop while a mode is active | `±16` units/s |

Distinct contact entry/exit thresholds prevent noisy landmarks from rapidly
switching direction. Contact zooms out; every separated gap zooms in, including
the middle range and maximum spread. Both directions continue while their
corresponding gap mode is held; only a brief transition between contact and
separation is debounced. Tracking loss resets the zoom state. The intent
passes through the `WORLD` interaction resolver before becoming a camera
target.

The keyboard demo's `E`/`PageUp` and `Q`/`PageDown` depth controls use the
same 16-unit/s target speed as hand zoom, allowing direct speed comparison.

## 6. Camera motion and numerical integration

`WorldMotion` contains target velocities in camera-relative directions:
`(strafe, vertical, depth)`. The camera controller combines these with basis
vectors to form a world-space target velocity:

```text
T = R · strafe + U · vertical + F · depth
```

The horizontal part of `T` is scaled down if its magnitude exceeds the
horizontal speed cap. The depth component is independently clamped. Defaults
are a horizontal speed cap of `352`, a depth cap of `448`, acceleration rate
`λₐ = 22 s⁻¹`, and damping rate `λ_d = 6.8 s⁻¹`.

To make motion stable across frame rates, each update clamps elapsed time to
`0.25` seconds and integrates in substeps no longer than `1/120` second. Each
substep uses exponential smoothing:

```text
λ = λₐ, when the target is nonzero
λ = λ_d, when the target is zero
α = 1 - exp(-λ Δt)
v_new = v_old + α (T - v_old)
p_new = p_old + v_new Δt
```

The exponential term makes the response depend on elapsed time rather than a
fixed number of frames. Acceleration approaches the requested velocity;
damping approaches zero after the input disappears, producing a short,
smooth coast rather than an abrupt stop.

The camera sector coordinate is calculated independently on each axis:

```text
sectorᵢ = floor(positionᵢ / sector_size)
```

The default sector size is `48` world units. `floor`, rather than integer
truncation, ensures negative coordinates map to the correct neighboring
sector.

## 7. Procedural geometry and streaming

The world seed defaults to `731921`. Each sector is identified by an integer
coordinate `(i, j, k)` and receives an independent deterministic random
generator. The sector seed is produced by BLAKE2b over the world seed and
coordinate string:

```text
sector_seed = BLAKE2b("world_seed:i:j:k", digest_size=8)
```

Therefore, requesting the same sector again creates the same geometry,
regardless of camera travel order.

For a sector of size `S = 48`, its center is:

```text
C_sector = S · (i, j, k)
```

Each sector contains one smooth backbone that wanders through randomized 3D
control points, plus 13 return branches, for 14 connected lines total. Ten
branches form two-turn polygonal spirals with triangular, hexagonal,
octagonal, and nonagonal facets; three are direct straight connectors. Each
branch starts and ends on the backbone, making one tangled network rather
than separate decorative objects. Seeded palettes cycle through aurora,
ocean, and ember colors; route, polygon side count, and gradients vary
deterministically to avoid repeated X/Y motifs. A sparse halo of dim points
adds depth without competing with the lines.

Each sector seed fixes its anchor route, branch connections, colors, and dust.
The backbone is a Catmull–Rom spline through anchors `P₀ … Pₙ`; every
consecutive segment shares endpoint positions and tangents. A polygonal
branch builds a spiral from evenly spaced polygon corners whose radius
tightens over two turns. Each pair of corners is rendered as a straight line:

```text
θᵢ = θ₀ + 2πN·i/M
rᵢ = r₀(1 - 0.58i/M)
Qᵢ = C + rᵢ(cos(θᵢ)U + sin(θᵢ)V)
```

Here `N` is the number of polygon sides, `M = 2N` is the number of corner
steps, and `U,V` span a seeded plane. Straight connectors join each spiral
to its two backbone samples. Each backbone enters and exits at seeded points
on the sector's depth faces. The boundary coordinate is hashed from world
seed, x/y sector, and depth index; adjacent sectors share the exact same
world-space endpoint, so the main path remains joined through the world.

The default `SectorStreamer` requests a neighborhood of radius 2 in x, 2 in y,
and 3 in z. That is at most `5 × 5 × 7 = 175` sectors around the camera.
It generates at most three missing sectors per simulation update, preventing a
sector boundary from causing a large synchronous generation spike. The
initial world fills progressively over several updates. Generated geometry
is retained in a 192-entry least-recently-used cache. As a sector approaches
the outermost ring, its alpha eases to zero across the final sector-width;
new sectors enter at zero alpha and fade in as the camera approaches. Sector
coordinates and camera sector position are passed to the shader, so fading
updates continuously without re-uploading static geometry. The currently
requested coordinate tuple is memoized, so a camera remaining in the same
sector neighborhood does not regenerate or reassemble it.

## 8. Frustum culling and OpenGL rendering

The renderer uses a 68° vertical field of view. For aspect ratio `a`, the
horizontal and vertical half-frustum slopes are:

```text
t_y = tan(68° / 2)
t_x = a · t_y
```

Each sector has an axis-aligned bounding box with center `B` and half-extents
`E = (eₓ, eᵧ, e_z)`. Let `D = B - C`, and project its center into the camera
basis:

```text
d = D · F       (forward depth)
h = D · R       (horizontal offset)
v = D · U       (vertical offset)
```

The box's conservative projected radii are:

```text
r_d = |Fₓ|eₓ + |Fᵧ|eᵧ + |F_z|e_z
r_h = |Rₓ|eₓ + |Rᵧ|eᵧ + |R_z|e_z
r_v = |Uₓ|eₓ + |Uᵧ|eᵧ + |U_z|e_z
```

A 24-unit fade band surrounds the frustum. A sector is rejected only when it
lies completely outside the frustum and this extra band. For example, the
horizontal rejection is conservative and includes both the box depth radius
and the fade distance:

```text
|h| > d·t_x + r_h + r_d·t_x + 24√(1+t_x²)
```

The vertical test uses `t_y` and `r_v` in the same way. The shader computes
each line/particle's distance to the nearest of the four side planes and the
near/far planes, then smoothly fades alpha to zero at the frustum boundary.
Thus geometry leaving through the top, bottom, left, right, or depth limits
fades rather than vanishing with a hard sector-culling edge. This inexpensive
bounding-box test avoids sending clearly invisible geometry to the GPU; it is
not a per-triangle visibility test.

Visible line and point data are concatenated into a single vertex buffer when
the set of visible sector coordinates changes. Each vertex uses ten 32-bit
floats:

```text
x, y, z, red, green, blue, size, sector_x, sector_y, sector_z
```

The vertex shader transforms position with `u_mvp`. The renderer draws
connected line pairs with `GL_LINES` and depth testing, then round dust
particles with additive blending. For point primitives it sets a screen-space
size using:

```text
point_size = clamp(size · (0.72 · viewport_height) / max(1, clip_w), 1, 8)
```

For particles, the fragment shader discards fragments outside a radius-0.5
circle in `gl_PointCoord`. Depth testing preserves occlusion.

The renderer asks OpenGL for a 3.3 core shader context. It reports shader,
buffer, or context errors through the GUI rather than silently treating a
failed renderer as successful.

## 9. Threading, frame rates, and recovery

- The camera worker targets 30 frames per second and sleeps for the remainder
  of its frame interval.
- Camera preview rendering is coalesced to at most 15 frames per second and
  resized to no larger than 320×240 before annotation and display.
- The world worker updates simulation independently from the camera. It keeps
  only the newest hand state rather than queuing an unbounded history of old
  frames.
- The GUI receives worker output through Qt signals. OpenGL context operations
  remain in the renderer widget's GUI context.
- An isolated camera-frame processing error is reported and retried. Five
  consecutive failures are surfaced as a camera-worker error. Detector and
  camera resources are closed in the worker's cleanup path.
- Tracking loss sends a zero target for the missing hand; it does not recreate
  the world, reset position, or clear the sector cache.

These are configured targets and implementation behaviors, not guarantees of
actual frame rate. Camera hardware, MediaPipe, Qt, GPU driver, and generated
geometry load all affect observed performance.

## 10. Retained terminal project explorer

The `--cli` mode statically analyzes a Python project and renders a navigable
hierarchy of packages, files, classes, and functions. The analyzer uses Python
AST parsing; it does not import or execute scanned files. This mode is
independent from the default 3D world.

```bash
visigui --cli -p /path/to/python-project
```

`--frames N` selects finite terminal rendering and exits after `N` frames,
which is useful for scripts and smoke tests. Interactive terminal mode expects
a TTY.

## 11. Tests and validation

Run the test suite from the repository root:

```bash
PYTHONPATH=src python3 -m pytest -q
```

Run the most relevant subsystems only:

```bash
PYTHONPATH=src python3 -m pytest tests/interaction tests/world tests/gui tests/vision -q
```

Compile Python sources and tests:

```bash
python3 -m compileall -q src tests
```

Test command dispatch without launching the GUI:

```bash
PYTHONPATH=src python3 -m visigui --help
PYTHONPATH=src python3 -m visigui --cli --demo --frames 1 -p src
```

Unit tests use synthetic hands and frames. They verify software behavior but
cannot establish hand-recognition accuracy for a particular person or camera,
actual GPU performance, or compatibility with every OpenGL driver. Validate
the full-screen renderer and hand ergonomics on the target desktop and webcam.

## 12. Troubleshooting and limitations

- **No camera image:** check `--camera-index`, OS camera permission, and
  whether another process owns the camera. `--demo` verifies the world without
  camera hardware.
- **Model missing:** run
  `python -m visigui.download_model`, or use `--model PATH`.
- **Black/empty 3D panel:** verify that the desktop driver supports OpenGL 3.3
  core and inspect the renderer status and `~/.cache/visigui/runtime.log`.
  Qt's offscreen plugin cannot validate real GPU output.
- **Old command still runs:** check `command -v visigui`, use the installed
  environment's `bin/visigui`, and inspect `~/.local/bin` for older launchers.
- **Qt portal or MediaPipe native warnings:** a warning alone does not prove
  failure; check whether the camera status, tracking state, or renderer
  actually reports an error.

The pinch ratio is not metric depth, hand labels depend on detector
orientation/conventions, and performance varies across cameras, lighting,
processors, and GPU drivers. Synthetic tests do not replace real-device
validation.
