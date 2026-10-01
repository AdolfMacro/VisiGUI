from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Optional

from ..core.contracts import HandObservation
from ..core.gesture import Gesture
from ..core.hand import FingerState, HandLandmark
from ..gesture.recognizer import GestureRecognizer
from .detector import FingerAnalyzer


@dataclass(frozen=True)
class HandState:
    handedness: str
    confidence: float
    landmarks: tuple[tuple[float, float, float], ...]
    palm_position: tuple[float, float, float]
    normalized_position: tuple[float, float, float]
    velocity: tuple[float, float]
    gesture: Gesture
    pinch_distance: float
    finger_state: FingerState

    @property
    def pinch_active(self) -> bool:
        return self.pinch_distance <= 0.12


@dataclass(frozen=True)
class TwoHandState:
    left_hand: Optional[HandState]
    right_hand: Optional[HandState]
    timestamp: float

    @property
    def hands_detected(self) -> int:
        return int(self.left_hand is not None) + int(self.right_hand is not None)


class HandTrackSelector:
    """Keep selecting the same detected hand when MediaPipe changes result order."""

    MAX_MISSED_FRAMES = 8
    _HANDEDNESS_PENALTY = 0.18

    def __init__(self) -> None:
        self._center: Optional[tuple[float, float]] = None
        self._handedness: Optional[str] = None
        self._missed_frames = 0

    def select(
        self,
        observations: tuple[HandObservation, ...],
    ) -> Optional[HandObservation]:
        if not observations:
            self._missed_frames += 1
            if self._missed_frames > self.MAX_MISSED_FRAMES:
                self.reset()
            return None
        centers = tuple(self._center_of(observation) for observation in observations)
        if self._center is None:
            selected_index = max(
                range(len(observations)),
                key=lambda index: self._initial_score(observations[index]),
            )
        else:
            def score(index: int) -> float:
                observation = observations[index]
                penalty = (
                    self._HANDEDNESS_PENALTY
                    if self._handedness
                    and observation.handedness
                    and self._handedness != observation.handedness
                    else 0.0
                )
                return math.dist(self._center, centers[index]) + penalty

            selected_index = min(range(len(observations)), key=score)
            if score(selected_index) > 0.55:
                selected_index = max(
                    range(len(observations)),
                    key=lambda index: self._initial_score(observations[index]),
                )
        selected = observations[selected_index]
        self._center = centers[selected_index]
        self._handedness = selected.handedness or self._handedness
        self._missed_frames = 0
        return selected

    def reset(self) -> None:
        self._center = None
        self._handedness = None
        self._missed_frames = 0

    @staticmethod
    def _center_of(observation: HandObservation) -> tuple[float, float]:
        return _palm_center(observation.landmark_positions)

    @staticmethod
    def _initial_score(observation: HandObservation) -> float:
        points = observation.landmark_positions
        area = (max(point[0] for point in points) - min(point[0] for point in points)) * (
            max(point[1] for point in points) - min(point[1] for point in points)
        )
        return area + (observation.handedness_score or 0.0) * 0.01


class OneEuroLandmarkFilter:
    """Low-pass landmark jitter while adapting quickly to intentional movement."""

    def __init__(
        self,
        min_cutoff: float = 1.4,
        beta: float = 0.2,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if min(min_cutoff, derivative_cutoff) <= 0 or beta < 0:
            raise ValueError("Landmark filter cutoffs must be positive and beta non-negative")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._previous: Optional[tuple[tuple[float, float], ...]] = None
        self._derivative: Optional[tuple[tuple[float, float], ...]] = None
        self._timestamp: Optional[float] = None

    def update(
        self,
        landmarks: tuple[tuple[float, float], ...],
        timestamp: float,
    ) -> tuple[tuple[float, float], ...]:
        if len(landmarks) != 21 or any(
            len(point) != 2 or not all(math.isfinite(value) for value in point)
            for point in landmarks
        ):
            raise ValueError("A landmark filter requires 21 finite 2D points")
        if not math.isfinite(timestamp):
            raise ValueError("Landmark timestamp must be finite")
        if self._previous is None or self._timestamp is None or timestamp <= self._timestamp:
            self._previous = landmarks
            self._derivative = tuple((0.0, 0.0) for _ in landmarks)
            self._timestamp = timestamp
            return landmarks
        dt = max(timestamp - self._timestamp, 1e-4)
        filtered: list[tuple[float, float]] = []
        derivatives: list[tuple[float, float]] = []
        for raw, previous, previous_derivative in zip(
            landmarks, self._previous, self._derivative or ()
        ):
            point_values: list[float] = []
            derivative_values: list[float] = []
            for value, previous_value, previous_speed in zip(
                raw, previous, previous_derivative
            ):
                derivative = (value - previous_value) / dt
                speed = self._low_pass(
                    derivative,
                    previous_speed,
                    self._alpha(self.derivative_cutoff, dt),
                )
                cutoff = self.min_cutoff + self.beta * abs(speed)
                point_values.append(
                    self._low_pass(value, previous_value, self._alpha(cutoff, dt))
                )
                derivative_values.append(speed)
            filtered.append((point_values[0], point_values[1]))
            derivatives.append((derivative_values[0], derivative_values[1]))
        self._previous = tuple(filtered)
        self._derivative = tuple(derivatives)
        self._timestamp = timestamp
        return self._previous

    def reset(self) -> None:
        self._previous = None
        self._derivative = None
        self._timestamp = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _low_pass(value: float, previous: float, alpha: float) -> float:
        return alpha * value + (1.0 - alpha) * previous


class DualHandTracker:
    """Pair MediaPipe observations into stable logical left/right hand states."""

    _MAX_MISSED_FRAMES = 8
    _HANDEDNESS_PENALTY = 0.24

    def __init__(
        self,
        analyzer: Optional[FingerAnalyzer] = None,
        recognizer: Optional[GestureRecognizer] = None,
    ) -> None:
        self.analyzer = analyzer or FingerAnalyzer()
        self.recognizer = recognizer or GestureRecognizer()
        self._centers: dict[str, tuple[float, float]] = {}
        self._filters = {
            "Left": OneEuroLandmarkFilter(),
            "Right": OneEuroLandmarkFilter(),
        }
        self._previous_positions: dict[str, tuple[float, float]] = {}
        self._missed_frames = {"Left": 0, "Right": 0}
        self._latest_observations: dict[str, HandObservation] = {}

    def update(
        self,
        observations: tuple[HandObservation, ...],
        timestamp: float,
    ) -> TwoHandState:
        assignments = self._assign(observations)
        states: dict[str, HandState] = {}
        for side in ("Left", "Right"):
            observation = assignments.get(side)
            if observation is None:
                self._missed_frames[side] += 1
                self._latest_observations.pop(side, None)
                if self._missed_frames[side] > self._MAX_MISSED_FRAMES:
                    self._reset_side(side)
                continue
            self._missed_frames[side] = 0
            states[side] = self._state_for(side, observation, timestamp)
            self._latest_observations[side] = observation
        return TwoHandState(states.get("Left"), states.get("Right"), timestamp)

    @property
    def latest_observations(self) -> dict[str, HandObservation]:
        return dict(self._latest_observations)

    def reset(self) -> None:
        self._latest_observations.clear()
        for side in ("Left", "Right"):
            self._reset_side(side)

    def _assign(
        self,
        observations: tuple[HandObservation, ...],
    ) -> dict[str, HandObservation]:
        if not observations:
            return {}
        candidates = observations[:2]
        centers = tuple(_palm_center(item.landmark_positions) for item in candidates)
        sides = ("Left", "Right")
        if not self._centers:
            result: dict[str, HandObservation] = {}
            remaining = list(range(len(candidates)))
            for index, observation in enumerate(candidates):
                side = _canonical_handedness(observation.handedness)
                if side is not None and side not in result:
                    result[side] = observation
                    remaining.remove(index)
            if remaining:
                unfilled = [side for side in sides if side not in result]
                remaining.sort(key=lambda index: centers[index][0])
                if len(unfilled) == 2 and len(remaining) == 2:
                    result[unfilled[0]] = candidates[remaining[1]]
                    result[unfilled[1]] = candidates[remaining[0]]
                else:
                    for side, index in zip(unfilled, remaining):
                        result[side] = candidates[index]
            return result

        if len(candidates) == 1:
            observation = candidates[0]
            reported = _canonical_handedness(observation.handedness)
            available = [side for side in sides if side in self._centers]
            if reported in available:
                side = reported
            elif available:
                side = min(available, key=lambda item: math.dist(self._centers[item], centers[0]))
            else:
                side = reported or ("Left" if centers[0][0] > 0.5 else "Right")
            return {side: observation}

        assignments = tuple(itertools.permutations(sides, len(candidates)))

        def cost(assignment: tuple[str, ...]) -> float:
            total = 0.0
            for index, side in enumerate(assignment):
                if side in self._centers:
                    total += math.dist(self._centers[side], centers[index])
                label = _canonical_handedness(candidates[index].handedness)
                if label is not None and label != side:
                    total += self._HANDEDNESS_PENALTY
            return total

        selected = min(assignments, key=cost)
        return {side: candidates[index] for index, side in enumerate(selected)}

    def _state_for(self, side: str, observation: HandObservation, timestamp: float) -> HandState:
        raw = observation.landmark_positions
        center = _palm_center(raw)
        self._centers[side] = center
        normalized = self._filters[side].update(
            tuple((point[0], point[1]) for point in raw),
            timestamp,
        )
        previous = self._previous_positions.get(side, center)
        dt = max(1e-4, timestamp - getattr(self, f"_{side.lower()}_timestamp", timestamp))
        velocity = ((center[0] - previous[0]) / dt, (center[1] - previous[1]) / dt)
        self._previous_positions[side] = center
        setattr(self, f"_{side.lower()}_timestamp", timestamp)
        landmarks = tuple(
            (point[0], point[1], original[2])
            for point, original in zip(normalized, raw)
        )
        finger_state = self.analyzer.analyze(observation)
        if finger_state is None:
            finger_state = FingerState()
        pinch_landmarks = observation.world_landmark_positions or landmarks
        pinch_distance = _pinch_distance(
            pinch_landmarks,
            include_depth=observation.world_landmark_positions is not None,
        )
        recognized = self.recognizer.recognize(
            finger_state,
            confidence=observation.confidence,
            timestamp=timestamp,
        )
        gesture = (
            Gesture.PINCH
            if pinch_distance <= 0.12
            else Gesture[recognized.gesture_name]
        )
        palm = _palm_center(landmarks)
        box_width = max(point[0] for point in landmarks) - min(point[0] for point in landmarks)
        box_height = max(point[1] for point in landmarks) - min(point[1] for point in landmarks)
        apparent_size = math.sqrt(max(box_width * box_height, 1e-8))
        return HandState(
            handedness=side,
            confidence=observation.handedness_score or observation.confidence or 0.0,
            landmarks=landmarks,
            palm_position=(palm[0], palm[1], sum(point[2] for point in landmarks) / 21),
            normalized_position=(palm[0], palm[1], apparent_size),
            velocity=velocity,
            gesture=gesture,
            pinch_distance=pinch_distance,
            finger_state=finger_state,
        )

    def _reset_side(self, side: str) -> None:
        self._centers.pop(side, None)
        self._latest_observations.pop(side, None)
        self._previous_positions.pop(side, None)
        self._missed_frames[side] = 0
        self._filters[side].reset()
        setattr(self, f"_{side.lower()}_timestamp", 0.0)


def _canonical_handedness(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "left":
        return "Left"
    if normalized == "right":
        return "Right"
    return None


def _palm_center(landmarks: tuple[tuple[float, float, float], ...]) -> tuple[float, float]:
    indices = (
        HandLandmark.WRIST,
        HandLandmark.INDEX_MCP,
        HandLandmark.MIDDLE_MCP,
        HandLandmark.RING_MCP,
        HandLandmark.PINKY_MCP,
    )
    return (
        sum(landmarks[index][0] for index in indices) / len(indices),
        sum(landmarks[index][1] for index in indices) / len(indices),
    )


def _pinch_distance(
    landmarks: tuple[tuple[float, float, float], ...],
    include_depth: bool = False,
) -> float:
    thumb = landmarks[HandLandmark.THUMB_TIP]
    index = landmarks[HandLandmark.INDEX_TIP]
    index_mcp = landmarks[HandLandmark.INDEX_MCP]
    pinky_mcp = landmarks[HandLandmark.PINKY_MCP]
    dimensions = 3 if include_depth else 2
    thumb_tip = thumb[:dimensions]
    index_tip = index[:dimensions]
    index_base = index_mcp[:dimensions]
    pinky_base = pinky_mcp[:dimensions]
    palm_width = math.dist(index_base, pinky_base)
    if palm_width <= 1e-6:
        return math.inf
    return math.dist(thumb_tip, index_tip) / palm_width
