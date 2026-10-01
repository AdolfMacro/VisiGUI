import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np

from visigui.core.contracts import HandObservation
from visigui.gui import workers


def test_camera_worker_surfaces_capture_errors_and_closes_resources(monkeypatch):
    calls = []

    class FakeCamera:
        def __init__(self, device_index):
            calls.append(("camera-created", device_index))

        def open(self):
            calls.append(("camera-opened",))

        def read(self):
            raise RuntimeError("capture failed")

        def close(self):
            calls.append(("camera-closed",))

    class FakeDetector:
        def __init__(self, model_path):
            calls.append(("detector-created", model_path))

        def close(self):
            calls.append(("detector-closed",))

    monkeypatch.setattr(workers, "Camera", FakeCamera)
    monkeypatch.setattr(workers, "HandDetector", FakeDetector)
    worker = workers.CameraWorker(2, "model.task")
    errors = []
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert len(errors) == 1
    assert "five consecutive frames" in errors[0]
    assert ("detector-closed",) in calls
    assert ("camera-closed",) in calls


def test_camera_worker_surfaces_detector_startup_error_and_closes_camera(monkeypatch):
    calls = []

    class FakeCamera:
        def __init__(self, device_index):
            calls.append("camera-created")

        def close(self):
            calls.append("camera-closed")

    def fail_detector(model_path):
        raise FileNotFoundError(model_path)

    monkeypatch.setattr(workers, "Camera", FakeCamera)
    monkeypatch.setattr(workers, "HandDetector", fail_detector)
    worker = workers.CameraWorker(0, "missing.task")
    errors = []
    worker.error_occurred.connect(errors.append)

    worker.run()

    assert errors == ["FileNotFoundError: missing.task"]
    assert calls == ["camera-created", "camera-closed"]


def test_camera_worker_forwards_mirrored_live_landmarks(monkeypatch):
    points = tuple(
        (0.2 + index * 0.01, 0.3 + index * 0.005, 0.0)
        for index in range(21)
    )
    observation = HandObservation(
        landmark_positions=points,
        handedness="Right",
        image_width=640,
        image_height=480,
    )
    frame = SimpleNamespace(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        width=640,
        height=480,
        timestamp=1.0,
    )

    class FakeCamera:
        def __init__(self, device_index):
            pass

        def open(self):
            pass

        def read(self):
            return frame

        def close(self):
            pass

    class FakeDetector:
        def __init__(self, model_path):
            pass

        def detect_all(self, camera_frame):
            return (observation,)

        def close(self):
            pass

    monkeypatch.setattr(workers, "Camera", FakeCamera)
    monkeypatch.setattr(workers, "HandDetector", FakeDetector)
    worker = workers.CameraWorker(0, "model.task")
    frames = []
    hand_frames = []
    worker.frame_ready.connect(lambda *values: (frames.append(values), worker.stop()))
    worker.hands_ready.connect(hand_frames.append)

    worker.run()

    assert len(frames) == 1
    assert frames[0][0].width() == 320
    assert frames[0][0].height() == 240
    assert frames[0][-1] == tuple(
        (1.0 - x, y)
        for x, y, _ in points
    )
    assert len(hand_frames) == 1
    assert hand_frames[0].right_hand is not None
    assert hand_frames[0].right_hand.landmarks == points


def test_camera_worker_skips_preview_image_processing_when_disabled(monkeypatch):
    frame = SimpleNamespace(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        width=640,
        height=480,
        timestamp=1.0,
    )

    class FakeCamera:
        def __init__(self, device_index):
            pass

        def open(self):
            pass

        def read(self):
            return frame

        def close(self):
            pass

    class FakeDetector:
        def __init__(self, model_path):
            pass

        def detect_all(self, camera_frame):
            return ()

        def close(self):
            pass

    monkeypatch.setattr(workers, "Camera", FakeCamera)
    monkeypatch.setattr(workers, "HandDetector", FakeDetector)

    def fail_preview(*args):
        raise AssertionError("preview should be skipped")

    monkeypatch.setattr(workers.CameraPreview, "annotate_frame", classmethod(fail_preview))
    worker = workers.CameraWorker(0, "model.task", preview_enabled=False)
    hand_frames = []
    frame_signals = []
    worker.hands_ready.connect(lambda hands: (hand_frames.append(hands), worker.stop()))
    worker.frame_ready.connect(frame_signals.append)

    worker.run()

    assert len(hand_frames) == 1
    assert frame_signals == []


def test_world_worker_coalesces_pending_frames_to_the_latest_state():
    worker = workers.WorldSimulationWorker()
    emitted = []
    worker.frame_ready.connect(emitted.append)
    first = object()
    second = object()

    worker._publish_frame(first)
    worker._publish_frame(second)

    assert emitted == [first]
    assert worker.take_latest_frame(first) is second

    third = object()
    worker._publish_frame(third)
    assert emitted == [first, third]


def test_camera_worker_skips_one_bad_frame_and_resumes_tracking(monkeypatch):
    frame = SimpleNamespace(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        width=8,
        height=8,
        timestamp=1.0,
    )

    class RecoveringCamera:
        def __init__(self, device_index):
            self.read_count = 0

        def open(self):
            pass

        def read(self):
            self.read_count += 1
            if self.read_count == 1:
                raise RuntimeError("temporary camera hiccup")
            return frame

        def close(self):
            pass

    class EmptyDetector:
        def __init__(self, model_path):
            pass

        def detect_all(self, camera_frame):
            return ()

        def close(self):
            pass

    monkeypatch.setattr(workers, "Camera", RecoveringCamera)
    monkeypatch.setattr(workers, "HandDetector", EmptyDetector)
    worker = workers.CameraWorker(0, "model.task")
    errors = []
    frames = []
    worker.error_occurred.connect(errors.append)
    worker.frame_ready.connect(lambda *values: (frames.append(values), worker.stop()))

    worker.run()

    assert errors == []
    assert len(frames) == 1
