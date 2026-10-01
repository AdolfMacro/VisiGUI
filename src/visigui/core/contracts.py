from dataclasses import dataclass, field
import math
from typing import Any, Optional


@dataclass(frozen=True)
class CameraFrame:
    width: int
    height: int
    timestamp: float
    image: Any = field(default=None, repr=False, compare=False)
    frame_id: Optional[int] = None


@dataclass(frozen=True)
class HandObservation:
    landmark_positions: tuple[tuple[float, float, float], ...]
    handedness: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: float = 0.0
    frame_id: Optional[int] = None
    handedness_score: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    world_landmark_positions: Optional[tuple[tuple[float, float, float], ...]] = None

    def __post_init__(self) -> None:
        if (self.image_width is None) != (self.image_height is None):
            raise ValueError("Hand image width and height must be provided together")
        if self.image_width is not None:
            dimensions = (self.image_width, self.image_height)
            if any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in dimensions
            ):
                raise ValueError("Hand image dimensions must be positive integers")
        if self.world_landmark_positions is not None and (
            len(self.world_landmark_positions) != 21
            or any(
                len(point) != 3 or not all(math.isfinite(value) for value in point)
                for point in self.world_landmark_positions
            )
        ):
            raise ValueError("World hand landmarks must contain 21 finite 3D points")


@dataclass(frozen=True)
class GestureObservation:
    gesture_name: str
    confidence: Optional[float]
    timestamp: float


@dataclass(frozen=True)
class StableGesture:
    gesture_name: str
    stable_duration: float
    timestamp: float
    confidence: Optional[float] = None
