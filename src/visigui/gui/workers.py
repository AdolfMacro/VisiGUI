from __future__ import annotations

from dataclasses import replace
from threading import Event, Lock
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ..camera.camera import Camera
from ..core.contracts import GestureObservation
from ..gesture.stabilizer import GestureStabilizer
from ..vision.camera_preview import CameraPreview
from ..vision.detector import HandDetector
from ..vision.tracking import DualHandTracker, TwoHandState
from ..world.simulation import GenerativeWorld, WorldFrame
from ..interaction.world_controller import WorldMotion


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object, object, str, str, bool, float, object)
    hands_ready = pyqtSignal(object)
    action_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera_index: int,
        model_path: str | Path,
        parent=None,
        preview_enabled: bool = True,
    ):
        super().__init__(parent)
        self.camera_index = camera_index
        self.model_path = str(model_path)
        self.preview_enabled = preview_enabled
        self._stop_requested = Event()
        self._last_preview_time = 0.0

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        camera = None
        detector = None
        self.status_changed.emit("Opening camera")
        try:
            camera = Camera(device_index=self.camera_index)
            detector = HandDetector(model_path=self.model_path)
            stabilizer = GestureStabilizer(min_stable_duration=0.18, buffer_size=5)
            dual_tracker = DualHandTracker()
            camera.open()
            self.status_changed.emit("Camera connected")
            last_time = time.monotonic()
            consecutive_frame_errors = 0
            while not self._stop_requested.is_set():
                started = time.monotonic()
                dt = max(0.0, started - last_time)
                last_time = started
                try:
                    self._process_frame(camera.read(), detector, dual_tracker, stabilizer, dt, started)
                except Exception as error:
                    consecutive_frame_errors += 1
                    self.status_changed.emit(
                        f"Frame processing warning {consecutive_frame_errors}/5: "
                        f"{type(error).__name__}: {error}"
                    )
                    if consecutive_frame_errors >= 5:
                        raise RuntimeError(
                            "Camera processing failed for five consecutive frames: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    self._stop_requested.wait(0.04)
                    continue
                consecutive_frame_errors = 0
                remaining = 1.0 / 30.0 - (time.monotonic() - started)
                if remaining > 0:
                    self._stop_requested.wait(remaining)
        except Exception as error:
            self.error_occurred.emit(f"{type(error).__name__}: {error}")
        finally:
            if detector is not None:
                try:
                    detector.close()
                except Exception as error:
                    self.error_occurred.emit(f"Detector cleanup failed: {type(error).__name__}: {error}")
            if camera is not None:
                try:
                    camera.close()
                except Exception as error:
                    self.error_occurred.emit(f"Camera cleanup failed: {type(error).__name__}: {error}")
            self.status_changed.emit("Camera stopped")

    def _process_frame(
        self,
        frame,
        detector: HandDetector,
        dual_tracker: DualHandTracker,
        stabilizer: GestureStabilizer,
        dt: float,
        started: float,
    ) -> None:
        observations = detector.detect_all(frame)
        hands = dual_tracker.update(observations, frame.timestamp)
        active_hands = tuple(
            hand for hand in (hands.left_hand, hands.right_hand)
            if hand is not None
        )
        if not active_hands:
            stabilizer.reset()
        else:
            primary = hands.left_hand or hands.right_hand
            assert primary is not None
            event = stabilizer.update(
                GestureObservation(
                    gesture_name=primary.gesture.name,
                    confidence=primary.confidence,
                    timestamp=frame.timestamp,
                ),
                dt,
            )
            if event is not None:
                self.action_ready.emit(event.gesture.name)
        self.hands_ready.emit(hands)
        if not self.preview_enabled:
            return
        preview_time = time.monotonic()
        if preview_time - self._last_preview_time < 1 / 15:
            return
        self._last_preview_time = preview_time
        annotated_observations = tuple(
            replace(
                dual_tracker.latest_observations[side],
                landmark_positions=hand.landmarks,
                handedness=side,
            )
            for side, hand in (("Left", hands.left_hand), ("Right", hands.right_hand))
            if hand is not None and side in dual_tracker.latest_observations
        )
        scale = min(320 / frame.width, 240 / frame.height, 1.0)
        preview_size = (round(frame.width * scale), round(frame.height * scale))
        preview_frame = cv2.resize(
            frame.image,
            preview_size,
            interpolation=cv2.INTER_AREA,
        )
        image = CameraPreview.annotate_frame(preview_frame, annotated_observations)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qt_image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        first_hand = active_hands[0] if active_hands else None
        landmark_positions = (
            tuple((1.0 - x, y) for x, y, _ in first_hand.landmarks)
            if first_hand is not None else None
        )
        elapsed = max(time.monotonic() - started, 1e-8)
        self.frame_ready.emit(
            qt_image,
            first_hand.finger_state if first_hand is not None else None,
            first_hand.gesture.name if first_hand is not None else "",
            first_hand.handedness if first_hand is not None else "",
            first_hand is not None,
            1.0 / elapsed,
            landmark_positions,
        )


class WorldSimulationWorker(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, world: Optional[GenerativeWorld] = None, parent=None):
        super().__init__(parent)
        self.world = world or GenerativeWorld()
        self._stop_requested = Event()
        self._state_lock = Lock()
        self._hands = TwoHandState(None, None, time.monotonic())
        self._keyboard_motion = WorldMotion()
        self._latest_frame: Optional[WorldFrame] = None
        self._frame_notification_pending = False

    def set_hands(self, hands: TwoHandState) -> None:
        with self._state_lock:
            self._hands = hands

    def set_keyboard_motion(self, motion: WorldMotion) -> None:
        with self._state_lock:
            self._keyboard_motion = motion

    def _publish_frame(self, frame: WorldFrame) -> None:
        with self._state_lock:
            self._latest_frame = frame
            if self._frame_notification_pending:
                return
            self._frame_notification_pending = True
        self.frame_ready.emit(frame)

    def take_latest_frame(self, fallback: WorldFrame) -> WorldFrame:
        with self._state_lock:
            frame = self._latest_frame or fallback
            self._frame_notification_pending = False
            return frame

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        last_time = time.monotonic()
        while not self._stop_requested.is_set():
            started = time.monotonic()
            dt = started - last_time
            last_time = started
            with self._state_lock:
                hands = self._hands
                keyboard = self._keyboard_motion
            frame = self.world.update(hands, dt, keyboard)
            self._publish_frame(frame)
            self._stop_requested.wait(max(0.0, 1 / 30 - (time.monotonic() - started)))


class ProjectScanWorker(QThread):
    project_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, project_path: str | Path, parent=None):
        super().__init__(parent)
        self.project_path = Path(project_path)

    def run(self) -> None:
        try:
            from ..project.analyzer import ProjectAnalyzer

            graph = ProjectAnalyzer().analyze(self.project_path)
        except Exception as error:
            self.error_occurred.emit(f"{type(error).__name__}: {error}")
        else:
            self.project_ready.emit(graph)
