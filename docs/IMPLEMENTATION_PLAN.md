# VisiGUI Runtime Status

## Current product

The default invocation (`visigui` / `python -m visigui`) starts the full-screen VisiGUI 3D procedural world and webcam hand tracking. `--demo` starts the same world without a camera and uses keyboard flight. `--cli` retains the former terminal Python project explorer; `--frames N` selects its finite output path.

## Delivered behavior

- Dual-hand MediaPipe observations are paired into stable logical left/right states using handedness and spatial continuity, rather than trusting result ordering.
- Left fist and keyboard movement relative to a captured fist pose control bounded horizontal and vertical travel, with a dead zone and smoothed camera physics. Opening the left hand stops its movement target.
- Sustained right thumb/index contact zooms out; any separated thumb/index gap continuously zooms in. Filtered measurements and contact hysteresis prevent chatter; intent resolves through the world context.
- Zero/one/two hands are supported. Missing hands stop supplying input without restarting the world or resetting camera position, simulation, or sector cache.
- Camera processing retries isolated frame failures and surfaces five consecutive failures; worker cleanup is explicit.
- The world is deterministic by seed and sector coordinate. It combines procedural curves, helices, a spatial lattice, and point clouds in neighboring streamed sectors with a bounded cache.
- OpenGL renders lines and points in VBO batches. The HUD reports camera/tracking state and current sector; keyboard controls provide a non-camera fallback.
- The full-screen layout assigns one quarter to separate live left/right articulated hand views and the hand-tracker camera feed; the world renderer fills the larger lower three quarters.
- The independent legacy CLI keeps bounded AST project analysis without importing/executing scanned source.

## Relevant implementation

- `src/visigui/vision/tracking.py`: `HandState`, `TwoHandState`, `DualHandTracker`.
- `src/visigui/interaction/world_controller.py`: fist-gated navigation, stable endpoint zoom states, movement targets and depth intents.
- `src/visigui/world/camera.py`: free-flight 3D camera physics and snapshot basis.
- `src/visigui/world/procedural.py`: deterministic geometry and LRU streaming cache.
- `src/visigui/world/simulation.py`: persistent simulation and immutable render frames.
- `src/visigui/renderer/opengl_world.py`: shaders, sector batching and OpenGL rendering.
- `src/visigui/gui/workers.py`: camera and world workers.
- `src/visigui/gui/world_window.py`: full-screen runtime, HUD and lifecycle.
- `src/visigui/app.py`, `src/visigui/__main__.py`: default world dispatch and legacy CLI path.

## Test and run commands

```sh
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m pytest tests/interaction tests/world tests/gui -q
python3 -m compileall -q src tests
PYTHONPATH=src python3 -m visigui --help
PYTHONPATH=src python3 -m visigui --cli --demo --frames 1 -p src
```

For a camera-free desktop run: `python -m visigui --demo`. Real camera/gesture ergonomics require a physical camera and hands in view. OpenGL context/shader/pixel validation requires a desktop-capable GPU/driver; Qt's offscreen plugin cannot validate rendered output.

## Remaining validation boundaries

1. Confirm startup, shader compilation, smooth frame rate and visible geometry on a real OpenGL 3.3 desktop driver.
2. Exercise left/right hand assignment and relative-depth pinch control with real camera views, including mirrored camera orientations and tracking loss.
3. Measure generation/upload hitching while traversing sector boundaries on representative GPUs; the first sector neighborhood is generated synchronously on the world worker.
4. Test minimum Python and wheel availability across target Linux distributions. Python package declarations do not prove availability on every version/architecture.
5. Review MediaPipe native warnings and camera status on target hardware; synthetic tests cannot establish real-world recognition quality.
