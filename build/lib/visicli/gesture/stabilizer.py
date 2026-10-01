from collections import deque
import math
from typing import Optional

from ..core.contracts import GestureObservation, StableGesture
from ..core.gesture import Gesture, GestureEvent


class GestureStabilizer:
    def __init__(self, min_stable_duration: float = 0.15, buffer_size: int = 5):
        if min_stable_duration < 0 or not math.isfinite(min_stable_duration):
            raise ValueError("min_stable_duration must be finite and non-negative")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        self.min_stable_duration = min_stable_duration
        self.buffer_size = buffer_size
        self._history: deque[str] = deque(maxlen=buffer_size)
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0
        self._last_emitted: Optional[str] = None
        self._elapsed = 0.0
        self._stable_gesture: Optional[StableGesture] = None

    @property
    def stable_gesture(self) -> Optional[StableGesture]:
        return self._stable_gesture

    def update(self, observation: GestureObservation, dt: float) -> Optional[GestureEvent]:
        if dt < 0 or not math.isfinite(dt):
            raise ValueError("dt must be finite and non-negative")
        if not math.isfinite(observation.timestamp):
            raise ValueError("observation timestamp must be finite")
        self._elapsed += dt

        name = observation.gesture_name
        if name not in Gesture.__members__ or name == Gesture.UNKNOWN.name:
            self._history.clear()
            self._candidate = None
            self._last_emitted = None
            self._stable_gesture = None
            return None

        self._history.append(name)
        counts = {gesture: self._history.count(gesture) for gesture in self._history}
        majority = max(counts, key=lambda gesture: (counts[gesture], gesture == name))

        if majority != self._candidate:
            self._candidate = majority
            self._candidate_since = self._elapsed
            return None
        if majority == self._last_emitted:
            return None

        duration = self._elapsed - self._candidate_since
        if duration < self.min_stable_duration:
            return None

        previous = self._last_emitted or Gesture.UNKNOWN.name
        self._last_emitted = majority
        self._stable_gesture = StableGesture(
            gesture_name=majority,
            stable_duration=duration,
            timestamp=observation.timestamp,
            confidence=observation.confidence,
        )
        return GestureEvent(
            gesture=Gesture[majority],
            previous_gesture=Gesture[previous],
            timestamp=self._stable_gesture.timestamp,
            confidence=observation.confidence,
        )

    def reset(self) -> None:
        self._history.clear()
        self._candidate = None
        self._last_emitted = None
        self._elapsed = 0.0
        self._stable_gesture = None
