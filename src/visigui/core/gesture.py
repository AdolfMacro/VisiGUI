from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Gesture(Enum):
    FIST = auto()
    OPEN_PALM = auto()
    INDEX = auto()
    INDEX_MIDDLE = auto()
    INDEX_MIDDLE_RING = auto()
    INDEX_MIDDLE_RING_PINKY = auto()
    INDEX_RING = auto()
    INDEX_PINKY = auto()
    MIDDLE_RING = auto()
    THUMB_INDEX = auto()
    THUMB_MIDDLE = auto()
    PINCH = auto()
    FIVE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class GestureEvent:
    gesture: Gesture
    previous_gesture: Gesture
    timestamp: float
    confidence: Optional[float] = None
