from ..core.contracts import GestureObservation
from ..core.gesture import Gesture
from ..core.hand import FingerState


class GestureRecognizer:
    def recognize(
        self,
        finger_state: FingerState,
        confidence: float | None = None,
        timestamp: float = 0.0,
    ) -> GestureObservation:
        return GestureObservation(
            gesture_name=self._fingers_to_gesture_name(finger_state),
            confidence=confidence,
            timestamp=timestamp,
        )

    @staticmethod
    def _fingers_to_gesture_name(state: FingerState) -> str:
        if not any((state.thumb, state.index, state.middle, state.ring, state.pinky)):
            return Gesture.FIST.name
        if state.thumb and state.index and state.middle and state.ring and state.pinky:
            return Gesture.OPEN_PALM.name
        if state.thumb and state.index and not any((state.middle, state.ring, state.pinky)):
            return Gesture.THUMB_INDEX.name
        if state.index and not any((state.thumb, state.middle, state.ring, state.pinky)):
            return Gesture.INDEX.name
        if state.index and state.middle and not any((state.thumb, state.ring, state.pinky)):
            return Gesture.INDEX_MIDDLE.name
        if state.index and state.middle and state.ring and not any((state.thumb, state.pinky)):
            return Gesture.INDEX_MIDDLE_RING.name
        if state.index and state.middle and state.ring and state.pinky and not state.thumb:
            return Gesture.INDEX_MIDDLE_RING_PINKY.name
        return Gesture.UNKNOWN.name
