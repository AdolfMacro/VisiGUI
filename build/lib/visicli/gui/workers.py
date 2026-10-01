from __future__ import annotations

from threading import Event
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from ..camera.camera import Camera
from ..gesture.recognizer import GestureRecognizer
from ..gesture.stabilizer import GestureStabilizer
from ..vision.camera_preview import CameraPreview
from ..vision.detector import FingerAnalyzer, HandDetector


class CameraWorker(QThread):
    frame_ready = pyqtSignal(object, object, str, str, bool, float, object)
    action_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index: int, model_path: str | Path, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.model_path = str(model_path)
        self._stop_requested = Event()

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        camera = None
        detector = None
        self.status_changed.emit("Opening camera")
        try:
            camera = Camera(device_index=self.camera_index)
            detector = HandDetector(model_path=self.model_path)
            analyzer = FingerAnalyzer()
            recognizer = GestureRecognizer()
            stabilizer = GestureStabilizer(min_stable_duration=0.18, buffer_size=5)
            camera.open()
            self.status_changed.emit("Camera connected")
            last_time = time.monotonic()
            while not self._stop_requested.is_set():
                started = time.monotonic()
                dt = max(0.0, started - last_time)
                last_time = started
                frame = camera.read()
                observations = detector.detect_all(frame)
                observation = observations[0] if observations else None
                finger_state = analyzer.analyze(observation)
                gesture_name = ""
                handedness = ""
                if observation is None or finger_state is None:
                    stabilizer.reset()
                else:
                    handedness = observation.handedness or ""
                    gesture = recognizer.recognize(
                        finger_state,
                        confidence=observation.confidence,
                        timestamp=observation.timestamp,
                    )
                    gesture_name = gesture.gesture_name
                    event = stabilizer.update(gesture, dt)
                    if event is not None:
                        self.action_ready.emit(event.gesture.name)

                image = CameraPreview.annotate_frame(frame.image, observations)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                qt_image = QImage(
                    rgb.data,
                    width,
                    height,
                    int(rgb.strides[0]),
                    QImage.Format.Format_RGB888,
                ).copy()
                landmark_positions = None
                if observation is not None:
                    image_width = observation.image_width or frame.width
                    image_height = observation.image_height or frame.height
                    landmark_positions = tuple(
                        ((1.0 - x) * image_width, y * image_height)
                        for x, y, _ in observation.landmark_positions
                    )
                elapsed = max(time.monotonic() - started, 1e-8)
                self.frame_ready.emit(
                    qt_image,
                    finger_state,
                    gesture_name,
                    handedness,
                    observation is not None,
                    1.0 / elapsed,
                    landmark_positions,
                )
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
