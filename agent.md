# VisiGUI — Agent Continuation Guide

Read this guide and the relevant source/tests before changing behavior. When docs and code disagree, code and verified tests are authoritative. Do not present an intended capability as verified unless it has been exercised.

Related references:

- [README.md](README.md): installation, controls, architecture, coordinate systems, camera matrices, simulation, and rendering.
- [docs/ARCHITECTURE_AUDIT.md](docs/ARCHITECTURE_AUDIT.md): active subsystem boundaries and validation limits.
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md): implementation status and remaining verification.

## Product identity

The default experience is a full-screen, two-hand-controlled procedural 3D world. A normal `visigui` or `python -m visigui` launch requests the webcam and starts the world without a project path or setup menu. `--demo` runs the world without camera input and supports keyboard flight. The former Python source explorer remains available separately with `--cli`.

The world consists of deterministic procedural sectors containing continuous, tangled 3D line networks and a sparse dusting of particles. A stable seed and integer sector coordinate make each network repeatable; a bounded cache retains nearby sectors. The left hand starts planar movement when clenched; moving the fist relative to its clench origin steers, and opening the hand stops its target. With the right hand, thumb/index contact held zooms out; any separated gap continuously zooms in. The normalized 3D tip gap is smoothed and contact entry/exit thresholds use hysteresis. Keyboard alternatives are `WASD`/arrows, `E`/`PageUp`, and `Q`/`PageDown`; `Esc` exits and `F11` toggles full-screen.

## Product and safety principles

- Keep vision, tracking identity, gesture recognition, intent, movement physics, world generation, rendering, and UI responsibilities separated.
- Missing or temporarily lost hands must not reset simulation/camera/world state. No-hand input should ease movement to a stop.
- Do not assume MediaPipe result index means anatomical left or right. Preserve temporal identity; handle zero, one, or two observations.
- Use explicit errors, status signals, and cleanup. Do not mask renderer/camera errors with success-shaped fallbacks or broad silent catches.
- The legacy analyzer treats scanned source as data. Never import, execute, install, or run code/tests/build scripts from the selected project.
- Keep generated environments, camera/model output, caches, build artifacts, credentials, and unrelated user changes out of commits.
- Do not use `sudo`, package-manager mutations, or real installer runs against the user's home as validation without explicit authorization.

## Product history

VisiGUI began as a hand-controlled CLI prototype and later became a PyQt6 hand-tracking project explorer. The user has since explicitly redirected the default product to a procedural two-hand 3D world. The old explorer still exists as a separate CLI mode; its prior menus/actions are not the current default GUI. The current world renderer is new and must not be confused with the retired gear simulation.

The user may rename the checkout's root directory independently; do not rename or move the root as part of an internal product rename. Work only in the active repository root. Preserve `.git` and `LICENSE`. The checkout may lack a project-local `.venv`; inspect before assuming one exists. Historical install failures included root-owned build and egg-info directories; do not change ownership or delete user data without authorization.

## Runtime architecture

```text
python -m visigui (default)
  ├─ CameraWorker (QThread)
  │  └─ OpenCV -> MediaPipe (up to two hands) -> DualHandTracker
  │     -> stable left/right TwoHandState -> hand/world signals
  ├─ WorldSimulationWorker (QThread)
  │  └─ TwoHandWorldController -> WorldMotion intents
  │     -> CameraController -> GenerativeWorld / SectorStreamer
  ├─ GenerativeWorldView (QOpenGLWidget) -> OpenGL 3.3 batches
  └─ WorldWindow (full-screen HUD, camera preview, keyboard, lifecycle)

--cli
  └─ legacy camera/gesture loop -> ExplorerController
     -> bounded AST ProjectAnalyzer -> terminal hierarchy cards
```

### Key files

- `src/visigui/app.py`, `src/visigui/__main__.py`: argument parsing, default world launch, `--cli` and finite-frame dispatch.
- `src/visigui/vision/tracking.py`: stable two-hand pairing/state and adaptive filtered landmark tracks.
- `src/visigui/interaction/world_controller.py`: left-fist relative navigation and debounced right-hand contact/spread zoom intents.
- `src/visigui/world/camera.py`: camera basis, position, frame-rate-independent acceleration/damping and sector location.
- `src/visigui/world/procedural.py`: deterministic geometry and bounded sector cache.
- `src/visigui/world/simulation.py`: persistent simulation state; combines camera, input and visible sectors.
- `src/visigui/renderer/opengl_world.py`: OpenGL context resources, shaders, sector VBO upload and drawing.
- `src/visigui/gui/workers.py`: camera/vision and world simulation worker lifecycle.
- `src/visigui/gui/world_window.py`: full-screen VisiGUI window, HUD, keyboard controls and shutdown.
- `src/visigui/project/`, `src/visigui/terminal/`: isolated legacy static project explorer.

## Interaction and tracking constraints

- `DualHandTracker` prefers confirmed handedness labels and uses recent spatial continuity for stable assignment when labels are absent/ambiguous.
- `HandState` carries handedness, confidence, 21 landmarks, normalized position/apparent size, velocity, finger state, gesture and pinch distance. Use the typed `TwoHandState` at the worker/world boundary.
- The left controller reference is initialized/rebased from observations. Do not interpret camera-relative x/y as absolute world position.
- The right-hand thumb-index gap is a relative gesture control, not calibrated metric distance. Pinch must be stable for the configured activation interval; its intent is resolved through the world interaction context and remains active until the release threshold is crossed.
- Filtered points are for visualization. Preserve low-latency classification/action input.
- One-hand and no-hand frames are expected, not exceptional worker conditions. An isolated camera frame exception is retried; repeated consecutive failures are surfaced.
- The world worker reads hand and keyboard state under a lock. Keep Qt widgets and OpenGL calls on the GUI/context thread.

## Procedural world and rendering constraints

- Keep generator output deterministic for a fixed seed and coordinate; test identical geometry and sector revisit/cache behavior.
- Geometry arrays are immutable after creation. Cache bounds and visible-neighborhood dimensions are deliberate memory/performance controls; cover changes with tests.
- Camera/simulation state must remain persistent through hand loss, worker updates, and sector changes.
- Keep rendering batched. Upload only when the visible sector-coordinate set changes; do not move GL objects across contexts or call GL from the simulation thread.
- OpenGL 3.3 core and PyOpenGL are runtime requirements. Qt's `offscreen` platform does not support a real `QOpenGLWidget` context on the current test host; widget tests are not shader/pixel proof.
- Surface context, shader, buffer, and draw failures visibly. Release GL resources with a current context.

## Legacy project analyzer safety

`ProjectAnalyzer` in `src/visigui/project/analyzer.py` statically reads eligible `.py` sources with `ast`. It creates package/file/class/function nodes and conservative containment/import/call edges. It must never import or execute scanned source. Preserve deterministic ordering, bounded traversal/source/symbol counts, and explicit parse/read/limit issues. An AST relationship is not a runtime trace.

Current declared scanner bounds include maximum file count, source size per file, total source bytes, definitions per file, and definitions across a scan. Check the analyzer and its tests before changing limits/exclusions. The project directory traversal itself should remain a future performance hardening target if untrusted massive trees become relevant.

## Installer and dependencies

- User-level Linux installer: `install.sh`; it creates an isolated venv under the configured user data path, installs the package and exposes `visigui` in the user bin path.
- The installer may edit user shell startup files and download the MediaPipe model. Test its mock/unit behavior first; do not run it in the user's home without permission.
- Runtime dependencies include Python >=3.10, NumPy, OpenCV, MediaPipe, PyQt6 and PyOpenGL; the `test` extra contains the test runner.
- Python wheel availability varies by Python version, architecture and distribution. Never claim all platforms are verified from `pyproject.toml` alone.
- After manifest changes, install only in an isolated project/test environment when available. Do not mutate system Python.

## Verification commands

Run the smallest relevant set first, then the full tests for meaningful cross-cutting changes:

```sh
PYTHONPATH=src python3 -m pytest tests/interaction tests/world tests/gui -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python3 -m pytest -q
python3 -m compileall -q src tests
PYTHONPATH=src python3 -m visigui --help
PYTHONPATH=src python3 -m visigui --cli --demo --frames 1 -p src
```

Run a full GUI/camera/OpenGL smoke test only where an actual desktop session, usable camera, and OpenGL-capable driver exist. Synthetic landmarks do not establish real-world gesture accuracy. The offscreen Qt platform on the current host explicitly reports that `QOpenGLWidget` is unsupported; report this limitation rather than interpreting an offscreen launch as proof of rendering.

Potential non-fatal MediaPipe and Qt portal warnings should remain inspectable in `~/.cache/visigui/runtime.log`. Do not suppress diagnostics as a substitute for fixing a verified cause.
