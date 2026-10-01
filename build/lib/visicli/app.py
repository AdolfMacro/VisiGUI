from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
import select
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from .camera.camera import Camera, CameraError
from .core.gesture import Gesture
from .core.hand import FingerState
from .gesture.recognizer import GestureRecognizer
from .gesture.stabilizer import GestureStabilizer
from .project.analyzer import ProjectAnalyzer
from .project.explorer import ExplorerController
from .terminal.dashboard import DashboardState, TerminalDashboard
from .terminal.input import InputHandler
from .terminal.log_capture import StderrLogCapture, default_log_path
from .terminal.project_view import ProjectViewport
from .vision.camera_preview import CameraPreview
from .vision.detector import DEFAULT_MODEL_PATH, FingerAnalyzer, HandDetector, resolve_model_path

DEMO_GESTURES = {
    "0": (Gesture.FIST, FingerState()),
    "1": (Gesture.INDEX, FingerState(index=True)),
    "2": (Gesture.THUMB_INDEX, FingerState(thumb=True, index=True)),
    "3": (Gesture.INDEX_MIDDLE_RING, FingerState(index=True, middle=True, ring=True)),
    "4": (
        Gesture.INDEX_MIDDLE_RING_PINKY,
        FingerState(index=True, middle=True, ring=True, pinky=True),
    ),
    "5": (Gesture.OPEN_PALM, FingerState(True, True, True, True, True)),
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visicli",
        description="Explore a Python project with a full-screen hand-controlled GUI or terminal mode.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="use keyboard input instead of a camera")
    modes.add_argument("--camera", action="store_true", help="explicitly select webcam tracking")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="use the terminal interface instead of the default full-screen desktop GUI",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera device index")
    parser.add_argument(
        "--no-camera-view",
        action="store_true",
        help="hide the camera image while keeping tracking active",
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="MediaPipe hand model path")
    parser.add_argument("--frames", type=int, help="render finite terminal frames, then exit")
    parser.add_argument("--debug", action="store_true", help="show frame timing in the dashboard")
    parser.add_argument(
        "-p",
        "--project",
        type=Path,
        help="Python project directory to analyze into the interactive hierarchy",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=default_log_path(),
        help="save camera and native-library diagnostics here",
    )
    parser.add_argument(
        "--ascii-art",
        action="store_true",
        help="legacy compatibility flag; the active dashboard always shows the ASCII hand",
    )
    args = parser.parse_args(argv)
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.camera_index < 0:
        parser.error("--camera-index must be non-negative")
    if args.project is not None:
        args.project = args.project.expanduser()
        if not args.project.exists() or not args.project.is_dir():
            parser.error(f"-p/--project must name an existing directory: {args.project}")
    return args


def _terminal_size() -> tuple[int, int]:
    columns, rows = shutil.get_terminal_size(fallback=(80, 24))
    return max(1, columns), max(1, rows)


def _read_key() -> Optional[str]:
    data = os.read(sys.stdin.fileno(), 1)
    if not data:
        return None
    return data.decode("ascii", errors="ignore")


def run(
    demo: bool = False,
    camera_index: int = 0,
    model_path: str = str(DEFAULT_MODEL_PATH),
    max_frames: Optional[int] = None,
    debug: bool = False,
    ascii_art: bool = False,
    project_path: str | Path | None = None,
    log_path: str | Path | None = None,
    camera_view: bool = True,
) -> int:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if camera_index < 0:
        raise ValueError("camera_index must be non-negative")
    if max_frames is None and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        raise RuntimeError("Interactive mode needs a terminal; use --frames N for a finite run.")
    graph = ProjectAnalyzer().analyze(project_path) if project_path is not None else None
    if not demo:
        resolved_model_path = resolve_model_path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {resolved_model_path}. "
                "Run `python -m visicli.download_model` or pass --model."
            )
        model_path = str(resolved_model_path)

    dashboard = TerminalDashboard()
    controller = ExplorerController(graph)
    viewport = ProjectViewport()
    recognizer = GestureRecognizer()
    stabilizers: dict[str, GestureStabilizer] = {}
    analyzer = FingerAnalyzer()
    detector = HandDetector(model_path=model_path) if not demo else None
    camera = Camera(device_index=camera_index) if not demo else None
    preview = CameraPreview() if not demo and camera_view else None
    input_handler = InputHandler()
    log_capture = StderrLogCapture(log_path) if not demo else None

    hand_detected: Optional[bool] = False if demo else None
    camera_ready = False
    handedness: Optional[str] = None
    finger_state: Optional[FingerState] = None
    gesture_name: Optional[str] = None
    frame_count = 0
    last_time = time.monotonic()
    frame_time_ms: Optional[float] = None
    fps: Optional[float] = None
    last_state = DashboardState(mode="DEMO" if demo else "CAMERA")

    try:
        if log_capture is not None:
            log_capture.start()
        if camera is not None:
            camera.open()
            camera_ready = True
        input_handler.enable_raw_mode()
        if sys.stdout.isatty():
            width, height = _terminal_size()
            dashboard.present(
                DashboardState(
                    mode="DEMO" if demo else "CAMERA",
                    camera_index=camera_index,
                    camera_ready=camera_ready,
                    hand_detected=hand_detected,
                    ascii_art=True,
                    project_lines=viewport.render(
                        graph,
                        controller,
                        max(1, width - 35 if width >= 59 else width - 1),
                        height,
                        "KEYBOARD DEMO" if demo else f"CAMERA {camera_index} READY",
                        (
                            f"{fps:.1f} FPS | {frame_time_ms:.1f} MS"
                            if debug and fps is not None and frame_time_ms is not None
                            else None
                        ),
                    ),
                ),
                width,
                height,
            )
        while max_frames is None or frame_count < max_frames:
            now = time.monotonic()
            dt = max(0.0, now - last_time)
            last_time = now
            if dt > 0 and frame_count > 0:
                fps = 1.0 / dt

            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                key = _read_key()
                if key is None or key in ("\x03", "q"):
                    break
                if key in DEMO_GESTURES:
                    gesture, finger_state = DEMO_GESTURES[key]
                    gesture_name = gesture.name
                    if demo:
                        hand_detected = True
                    controller.choose_gesture(gesture.name)
                elif key in ("\r", "\n", "c"):
                    controller.confirm()
                elif key in ("m", "b"):
                    controller.back()

            if not demo:
                assert camera is not None and detector is not None
                camera_frame = camera.read()
                observations = detector.detect_all(camera_frame)
                if preview is not None and not preview.show(camera_frame, observations):
                    preview = None
                hand_detected = bool(observations)
                if not observations:
                    handedness = None
                    gesture_name = None
                    finger_state = None
                    for stabilizer in stabilizers.values():
                        stabilizer.reset()
                else:
                    active_tracks: set[str] = set()
                    for observation in observations[:1]:
                        track_id = observation.handedness or "hand-0"
                        active_tracks.add(track_id)
                        stabilizer = stabilizers.setdefault(
                            track_id,
                            GestureStabilizer(min_stable_duration=0.2, buffer_size=5),
                        )
                        observed_fingers = analyzer.analyze(observation)
                        handedness = observation.handedness
                        if observed_fingers is None:
                            stabilizer.reset()
                            finger_state = None
                            gesture_name = Gesture.UNKNOWN.name
                        else:
                            gesture_observation = recognizer.recognize(
                                observed_fingers,
                                confidence=observation.confidence,
                                timestamp=observation.timestamp,
                            )
                            event = stabilizer.update(gesture_observation, dt)
                            finger_state = observed_fingers
                            gesture_name = gesture_observation.gesture_name
                            if event is not None:
                                controller.choose_gesture(event.gesture.name)
                    for track_id, stabilizer in stabilizers.items():
                        if track_id not in active_tracks:
                            stabilizer.reset()

            processing_started = time.monotonic()
            frame_time_ms = (processing_started - now) * 1000.0
            primary_track_id = (
                handedness if handedness in stabilizers else "hand-0"
            )
            stable_gesture = stabilizers.get(primary_track_id).stable_gesture if (
                primary_track_id in stabilizers
            ) else None
            stable = (
                demo and gesture_name is not None
                or stable_gesture is not None
                and stable_gesture.gesture_name == gesture_name
            )
            width, height = _terminal_size()
            last_state = DashboardState(
                mode="DEMO" if demo else "CAMERA",
                camera_index=camera_index,
                camera_ready=camera_ready,
                hand_detected=hand_detected,
                handedness=handedness,
                finger_state=finger_state,
                gesture_name=gesture_name,
                stable=stable,
                intent=controller.last_action,
                fps=fps,
                frame_time_ms=frame_time_ms,
                debug=debug,
                ascii_art=True,
                project_lines=viewport.render(
                    graph,
                    controller,
                    max(1, width - 35 if width >= 59 else width - 1),
                    height,
                    "KEYBOARD DEMO" if demo else f"CAMERA {camera_index} READY",
                    (
                        f"{fps:.1f} FPS | {frame_time_ms:.1f} MS"
                        if debug and fps is not None and frame_time_ms is not None
                        else None
                    ),
                ),
            )
            if sys.stdout.isatty():
                dashboard.present(last_state, width, height)
            frame_count += 1
            time.sleep(max(0.0, 1.0 / 30.0 - (time.monotonic() - now)))
    finally:
        with ExitStack() as cleanup:
            if log_capture is not None:
                cleanup.callback(log_capture.close)
            if detector is not None:
                cleanup.callback(detector.close)
            if preview is not None:
                cleanup.callback(preview.close)
            if camera is not None:
                cleanup.callback(camera.close)
            cleanup.callback(dashboard.close)
            cleanup.callback(input_handler.disable_raw_mode)

    if not sys.stdout.isatty():
        width, height = _terminal_size()
        dashboard.present(last_state, width, height)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        if not args.cli and args.frames is None:
            from .gui.main_window import launch_gui

            return launch_gui(
                project_path=args.project,
                camera_index=args.camera_index,
                model_path=args.model,
                demo=args.demo,
                camera_view=not args.no_camera_view,
            )
        return run(
            demo=args.demo,
            camera_index=args.camera_index,
            model_path=args.model,
            max_frames=args.frames,
            debug=args.debug,
            ascii_art=args.ascii_art,
            project_path=args.project,
            log_path=args.log_file,
            camera_view=not args.no_camera_view,
        )
    except KeyboardInterrupt:
        return 130
    except (CameraError, FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"visicli: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
