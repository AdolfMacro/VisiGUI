from __future__ import annotations

from typing import Any

from ..core.contracts import CameraFrame, HandObservation


_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class CameraPreview:
    """Show a mirrored webcam popup with a live landmark skeleton and square track box."""

    WINDOW_TITLE = "VisiGUI | Hand Tracker"

    def __init__(self, cv2_module: Any = None):
        self._cv2 = cv2_module
        self._opened = False
        self._active = True

    def show(
        self,
        frame: CameraFrame,
        observations: tuple[HandObservation, ...],
    ) -> bool:
        if not self._active:
            return False
        cv2 = self._cv2
        if cv2 is None:
            import cv2 as cv2_module

            cv2 = cv2_module
            self._cv2 = cv2

        image = self.annotate_frame(frame.image, observations, cv2)
        if not self._opened:
            try:
                cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WINDOW_TITLE, min(frame.width, 960), min(frame.height, 720))
            except cv2.error as error:
                raise RuntimeError(
                    "Could not open the camera preview window. "
                    "Use --no-camera-view in a headless environment."
                ) from error
            self._opened = True

        try:
            cv2.imshow(self.WINDOW_TITLE, image)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as error:
            raise RuntimeError(
                "Could not update the camera preview window. "
                "Use --no-camera-view in a headless environment."
            ) from error
        if key in (27, ord("q")):
            self.close()
            return False
        return True

    @classmethod
    def annotate_frame(
        cls,
        image: Any,
        observations: tuple[HandObservation, ...],
        cv2_module: Any = None,
    ) -> Any:
        cv2 = cv2_module
        if cv2 is None:
            import cv2 as cv2_module

            cv2 = cv2_module
        mirrored = cv2.flip(image, 1)
        for observation in observations:
            cls._draw_hand(mirrored, observation, cv2)
        if not observations:
            cv2.putText(
                mirrored,
                "NO HAND TRACKED",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return mirrored

    @staticmethod
    def _draw_hand(image: Any, observation: HandObservation, cv2: Any) -> None:
        height, width = image.shape[:2]
        points = tuple(
            (
                min(width - 1, max(0, round(x * (width - 1)))),
                min(height - 1, max(0, round(y * (height - 1)))),
            )
            for x, y, _ in observation.landmark_positions
        )

        for start, end in _HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (70, 220, 70), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, point, 3, (255, 180, 0), -1, cv2.LINE_AA)

        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        side = min(
            max(max_x - min_x, max_y - min_y) + 2 * max(12, round(max(width, height) * 0.025)),
            min(width, height),
        )
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        left = min(max(0, center_x - side // 2), width - side)
        top = min(max(0, center_y - side // 2), height - side)
        right = left + side
        bottom = top + side
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3, cv2.LINE_AA)
        role = {
            "Left": "MOVE",
            "Right": "ZOOM",
        }.get(observation.handedness or "", "UNASSIGNED")
        label = f"{observation.handedness or 'HAND'} HAND - {role}"
        cv2.putText(
            image,
            label,
            (left, max(22, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    def close(self) -> None:
        if self._opened and self._cv2 is not None:
            self._cv2.destroyWindow(self.WINDOW_TITLE)
            self._opened = False
        self._active = False
