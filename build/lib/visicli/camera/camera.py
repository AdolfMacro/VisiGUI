from __future__ import annotations

from typing import Any, Callable, Optional
import time

from ..core.contracts import CameraFrame


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        capture_factory: Optional[Callable[[int], Any]] = None,
    ):
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("Camera device index must be a non-negative integer")
        if width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive")
        self.device_index = device_index
        self.width = width
        self.height = height
        self._capture_factory = capture_factory
        self._capture: Any = None
        self._frame_id = 0

    def open(self) -> None:
        if self._capture is not None:
            return

        if self._capture_factory is None:
            import cv2

            capture = cv2.VideoCapture(self.device_index)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        else:
            capture = self._capture_factory(self.device_index)

        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Could not open webcam device {self.device_index}")

        self._capture = capture

    def read(self) -> CameraFrame:
        self.open()
        success, image = self._capture.read()
        if not success or image is None:
            raise CameraError(f"Could not read a frame from webcam device {self.device_index}")

        shape = getattr(image, "shape", ())
        if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
            raise CameraError("Webcam returned an invalid image frame")

        self._frame_id += 1
        return CameraFrame(
            width=int(shape[1]),
            height=int(shape[0]),
            timestamp=time.monotonic(),
            image=image,
            frame_id=self._frame_id,
        )

    def capture(self) -> CameraFrame:
        return self.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
