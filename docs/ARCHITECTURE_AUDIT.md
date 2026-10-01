# VisiGUI Architecture Audit

## Active product and runtime

The default product is a full-screen, two-hand-controlled, procedurally generated 3D world. It opens the webcam and starts vision automatically. `--demo` runs the same world with keyboard flight and no camera; `--cli` selects the preserved static Python project explorer.

```text
Default: python -m visigui
  ├─ CameraWorker (QThread) -> OpenCV camera -> MediaPipe (up to 2 hands)
  │  -> DualHandTracker -> stable left/right TwoHandState
  ├─ WorldSimulationWorker (QThread)
  │  -> TwoHandWorldController -> WorldMotion intents
  │  -> CameraController -> smoothed 3D camera state
  │  -> GenerativeWorld -> deterministic SectorStreamer/cache
  ├─ GenerativeWorldView (QOpenGLWidget) -> OpenGL 3.3 shaders/VBO batches
  └─ WorldWindow -> full-screen scene, status HUD and optional camera preview

--cli -> legacy Camera/vision -> GestureStabilizer -> ExplorerController
  -> bounded AST ProjectAnalyzer -> terminal project hierarchy
```

## Subsystem responsibilities

| Subsystem | Responsibility |
|---|---|
| `vision/detector.py` | MediaPipe Tasks adapter; observations and finger/gesture analysis. |
| `vision/tracking.py` | Stable logical left/right association, 21-point hand state, pinch/velocity, adaptive landmark filtering. |
| `gui/workers.py:CameraWorker` | Camera lifecycle, frame retry policy, tracking, annotated image and hand-state signals. Five consecutive processing failures are reported as an error; an isolated frame failure is retried. |
| `interaction/world_controller.py` | Converts left-fist displacement into bounded planar movement and right-hand thumb/index endpoint poses into zoom intents. |
| `world/camera.py` | Camera position and local basis; frame-rate-independent acceleration, velocity limits, damping, and sector coordinates. |
| `world/procedural.py` | Seeded Catmull–Rom backbone, connected polygonal spirals/straight branches, sparse particles, and bounded sector LRU cache. |
| `world/simulation.py` | Persistent camera/controller/streamer state; builds frames without resetting on tracking loss. |
| `gui/workers.py:WorldSimulationWorker` | Independent simulation loop and thread-safe hand/keyboard state handoff. |
| `renderer/opengl_world.py` | OpenGL shader setup, visible-sector line/point batching, VAO/VBO state, frustum and streamed-edge fades. |
| `gui/world_window.py` | Default full-screen layout: live left/right articulated hand vision and camera tracker occupy the upper quarter, with the world renderer in the lower three quarters, plus status, keyboard fallback, and orderly shutdown. |
| `project/` and `terminal/` | Separate legacy, bounded static Python-source explorer selected only by `--cli` (or `--frames`). |

## Tracking and input behavior

- MediaPipe provides zero, one, or two hand observations. Assignment prefers handedness and maintains continuity with recent spatial tracks instead of treating result index as identity.
- `TwoHandState` can represent either or neither hand. Camera/world updates continue when observations are missing.
- The left hand steers only while classified as a fist. Its fist pose establishes a relative origin; opening the hand clears the origin and zeros the movement target. A dead zone removes small movements and camera physics smooths acceleration/stopping.
- In `WORLD` context the right-hand normalized 3D thumb/index tip gap is filtered and debounced. Contact (`g ≤ 0.12`, exit above `0.20`) resolves to continuous `ZOOM_OUT`; every separated gap resolves to continuous `ZOOM_IN`, with hysteresis during the contact transition. Hand loss zeros and resets the target. The distance is normalized by palm width and is not metric.
- Gesture classification and smoothing are separate from drawing. Filtering is adaptive for landmarks; actions use gesture state and a sustained pinch gate.
- A single hand remains useful: available left/right channels continue to work independently. Losing both hands zeros their target input and does not recreate simulation or sector state.
- Keyboard fallback: `WASD`/arrows for planar movement, `E`/`PageUp` forward, `Q`/`PageDown` backward; `Esc` exits; `F11` toggles full-screen. `--demo` disables the camera.

## World and rendering

World geometry is determined by a global seed plus integer sector coordinates.
Each sector contains a Catmull–Rom backbone and 13 connected branches: ten
two-turn polygonal spirals with 3, 6, 8, or 9 sides and three straight
connectors. Their endpoints lie on the backbone, forming 14 connected lines
in one tangled wire-like network. Seeded palettes and route/shape variation
create structured randomness; a sparse halo of dim particles adds depth.
The backbone crosses each sector's depth faces at deterministic hashed
coordinates, so adjoining depth sectors share exact endpoints. Lines are
visual only, with no collision geometry, and the camera remains in free
flight.

Generated sector arrays are immutable; the streamer retains a deep
neighborhood in a bounded cache. Revisited coordinates reproduce exactly the
same geometry. The 5×5×7 default sector neighborhood contains 175 sectors
and is held in a 192-entry LRU cache; at most three missing sectors are
generated per simulation update to bound streaming work. A shader-side
one-sector fade at the streamed neighborhood boundary makes incoming and
outgoing sectors transition continuously without re-uploading geometry. The
renderer also culls against the view frustum with a 24-unit fade margin and
uploads a batched ten-float-per-vertex buffer only when the visible sector set
changes. Lines and particles fade at all frustum planes to avoid hard pops as
the camera turns or moves. Offscreen Qt tests cannot validate real GPU frame
pacing.

## Legacy CLI explorer boundaries

The older 2D Python project explorer is kept as an independent mode. It reads eligible files, parses them with `ast`, and represents packages/files/classes/functions plus conservative static relationships. It does not import or execute the inspected project. Traversal and source/symbol sizes have explicit limits. Static call/import edges are not runtime traces.

## Error handling and lifecycle

- Camera open/detector setup and cleanup failures are surfaced through worker signals; cleanup is attempted for each acquired resource.
- A single camera-frame failure does not end tracking; five consecutive errors end the worker with an explicit error.
- Camera and world workers stop before the window completes close. The renderer releases GL objects only while its context is current.
- OpenGL setup/draw failures remain visible in the HUD and are not overwritten by simulation status updates.
- Native library stderr diagnostics are retained at `~/.cache/visigui/runtime.log`.

## Verification and known limits

- Unit tests exercise handedness/order continuity, missing-hand behavior, pinch gating and mapping, camera dynamics, deterministic sectors/cache, interaction intent, and the legacy CLI.
- `QT_QPA_PLATFORM=offscreen` is suitable for Qt logic/widget tests, not GPU verification; this environment reports that `QOpenGLWidget` is unsupported on its offscreen platform. The renderer's actual shader/raster output must be validated on a desktop-capable OpenGL host.
- Synthetic observations do not prove accuracy across real cameras, lighting, mirrored views, hand sizes, or different users. The normalized thumb-index gap is a relative gesture control, not metric depth.
- MediaPipe may emit non-fatal native diagnostics; review the runtime log when investigating actual vision failures.

Architecture boundaries are tested in `tests/test_architecture_boundaries.py`. Update those tests when introducing new cross-subsystem imports or responsibilities.
