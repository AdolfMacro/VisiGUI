import numpy as np
import pytest

from visigui.camera.camera import Camera, CameraError


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.frames = [(True, np.zeros((12, 20, 3), dtype=np.uint8))]

    def isOpened(self):
        return self.opened

    def read(self):
        return self.frames.pop(0)

    def release(self):
        self.released = True


def test_camera_reads_frames_and_releases_device():
    capture = FakeCapture()
    camera = Camera(capture_factory=lambda _: capture)
    with camera:
        first = camera.read()
        assert (first.width, first.height, first.frame_id) == (20, 12, 1)
        assert first.timestamp > 0
        assert first.image.shape == (12, 20, 3)
    assert capture.released


def test_camera_initialization_failure_is_explicit():
    capture = FakeCapture(opened=False)
    camera = Camera(capture_factory=lambda _: capture)
    with pytest.raises(CameraError, match="Could not open webcam"):
        camera.open()
    assert capture.released


def test_camera_frame_failure_is_explicit():
    capture = FakeCapture()
    capture.frames = [(False, None)]
    camera = Camera(capture_factory=lambda _: capture)
    with pytest.raises(CameraError, match="Could not read a frame"):
        camera.read()
    camera.close()


def test_camera_rejects_negative_device_index():
    with pytest.raises(ValueError, match="non-negative integer"):
        Camera(device_index=-1)
