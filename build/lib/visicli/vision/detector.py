from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import math

from ..core.contracts import CameraFrame, HandObservation
from ..core.hand import FingerState, HandLandmark, HandLandmarks


DEFAULT_MODEL_PATH = Path.home() / ".cache" / "visicli" / "hand_landmarker.task"
LEGACY_MODEL_PATH = Path.home() / ".cache" / "eyehand" / "hand_landmarker.task"


def resolve_model_path(model_path: str | Path) -> Path:
    requested_path = Path(model_path).expanduser()
    if (
        requested_path == DEFAULT_MODEL_PATH
        and not requested_path.is_file()
        and LEGACY_MODEL_PATH.is_file()
    ):
        return LEGACY_MODEL_PATH
    return requested_path


class HandDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        landmarker: Any = None,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        for value in (
            min_hand_detection_confidence,
            min_hand_presence_confidence,
            min_tracking_confidence,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("MediaPipe confidence thresholds must be between 0 and 1")
        self.model_path = Path(model_path)
        self._landmarker = landmarker
        self._last_timestamp_ms = -1
        self._min_hand_detection_confidence = min_hand_detection_confidence
        self._min_hand_presence_confidence = min_hand_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence

    def _create_landmarker(self) -> Any:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {self.model_path}. "
                "Run `python -m visicli.download_model` to install it."
            )

        import mediapipe as mp

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self._min_hand_detection_confidence,
            min_hand_presence_confidence=self._min_hand_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: CameraFrame) -> Optional[HandObservation]:
        observations = self.detect_all(frame)
        return observations[0] if observations else None

    def detect_all(self, frame: CameraFrame) -> tuple[HandObservation, ...]:
        """Detect up to two hands, preserving MediaPipe's per-hand metadata."""
        if frame.image is None:
            raise ValueError("CameraFrame.image is required for hand detection")
        if self._landmarker is None:
            self._landmarker = self._create_landmarker()

        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        rgb = cv2.flip(rgb, 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame.timestamp * 1000)
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return ()

        observations: list[HandObservation] = []
        for index, hand_landmarks in enumerate(result.hand_landmarks[:2]):
            landmarks = tuple(
                (float(point.x), float(point.y), float(point.z))
                for point in hand_landmarks
            )
            HandLandmarks(landmarks)
            world_landmarks = None
            world_hands = getattr(result, "hand_world_landmarks", None)
            if world_hands is not None and index < len(world_hands):
                world_landmarks = tuple(
                    (float(point.x), float(point.y), float(point.z))
                    for point in world_hands[index]
                )
                HandLandmarks(world_landmarks)

            handedness: Optional[str] = None
            handedness_score: Optional[float] = None
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = category.category_name
                score = float(category.score)
                handedness_score = score if math.isfinite(score) else None

            observations.append(HandObservation(
                landmark_positions=landmarks,
                handedness=handedness,
                confidence=None,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                handedness_score=handedness_score,
                image_width=frame.width,
                image_height=frame.height,
                world_landmark_positions=world_landmarks,
            ))
        return tuple(observations)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class FingerAnalyzer:
    _MIN_PALM_LENGTH = 1e-5
    _MIN_FINGER_ALIGNMENT = 0.25
    _MIN_FINGER_LENGTH_RATIO = 0.58
    _MIN_FINGER_PALM_RATIO = 0.4
    _MIN_THUMB_ALIGNMENT = 0.05
    _MIN_THUMB_ABDUCTION_RATIO = 0.18

    _FINGER_JOINTS = (
        (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP),
        (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP),
        (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP),
        (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    )

    def analyze(self, observation: Optional[HandObservation]) -> Optional[FingerState]:
        if observation is None:
            return None
        source_positions = (
            observation.world_landmark_positions
            if observation.world_landmark_positions is not None
            else observation.landmark_positions
        )
        landmarks = HandLandmarks(source_positions)
        if (
            observation.world_landmark_positions is None
            and observation.image_width is not None
            and observation.image_height is not None
        ):
            positions = tuple(
                (x * observation.image_width, y * observation.image_height, z * observation.image_width)
                for x, y, z in landmarks.positions
            )
        else:
            positions = landmarks.positions

        wrist = positions[HandLandmark.WRIST]
        middle_mcp = positions[HandLandmark.MIDDLE_MCP]
        palm_axis = self._vector(wrist, middle_mcp)
        palm_length = self._magnitude(palm_axis)
        palm_width = self._distance(
            positions[HandLandmark.INDEX_MCP],
            positions[HandLandmark.PINKY_MCP],
        )
        if palm_length < self._MIN_PALM_LENGTH:
            return None
        palm_axis = tuple(component / palm_length for component in palm_axis)

        extended: list[bool] = []
        for mcp, pip, dip, tip in self._FINGER_JOINTS:
            points = [positions[index] for index in (mcp, pip, dip, tip)]
            segments = tuple(
                tuple(points[i + 1][axis] - points[i][axis] for axis in range(3))
                for i in range(3)
            )
            aligned = all(
                self._alignment(first, second) >= self._MIN_FINGER_ALIGNMENT
                for first, second in zip(segments, segments[1:])
            )
            finger_length = sum(self._magnitude(segment) for segment in segments)
            tip_vector = self._vector(positions[mcp], positions[tip])
            projected_extension = self._dot(tip_vector, palm_axis)
            extended.append(
                aligned
                and projected_extension >= max(
                    palm_length * self._MIN_FINGER_PALM_RATIO,
                    finger_length * self._MIN_FINGER_LENGTH_RATIO,
                )
            )

        thumb = self._thumb_extended(positions, palm_width)
        return FingerState(thumb, *extended)

    @staticmethod
    def _vector(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return tuple(end[index] - start[index] for index in range(3))

    @staticmethod
    def _magnitude(vector: tuple[float, float, float]) -> float:
        return math.sqrt(sum(component * component for component in vector))

    @staticmethod
    def _dot(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> float:
        return sum(first[index] * second[index] for index in range(3))

    @staticmethod
    def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    @staticmethod
    def _alignment(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        length_a = math.sqrt(sum(value * value for value in a))
        length_b = math.sqrt(sum(value * value for value in b))
        if length_a <= 1e-8 or length_b <= 1e-8:
            return -1.0
        return sum(a[i] * b[i] for i in range(3)) / (length_a * length_b)

    def _thumb_extended(
        self,
        positions: tuple[tuple[float, float, float], ...],
        palm_width: float,
    ) -> bool:
        cmc, mcp, ip, tip = (
            positions[HandLandmark.THUMB_CMC],
            positions[HandLandmark.THUMB_MCP],
            positions[HandLandmark.THUMB_IP],
            positions[HandLandmark.THUMB_TIP],
        )
        segments = (
            tuple(mcp[i] - cmc[i] for i in range(3)),
            tuple(ip[i] - mcp[i] for i in range(3)),
            tuple(tip[i] - ip[i] for i in range(3)),
        )
        aligned = all(
            self._alignment(a, b) >= self._MIN_THUMB_ALIGNMENT
            for a, b in zip(segments, segments[1:])
        )
        index_mcp = positions[HandLandmark.INDEX_MCP]
        lateral_extension = self._distance(tip, index_mcp) - self._distance(mcp, index_mcp)
        return aligned and lateral_extension > max(
            palm_width * self._MIN_THUMB_ABDUCTION_RATIO,
            1e-6,
        )
