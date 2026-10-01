from dataclasses import dataclass
from enum import IntEnum
import math


class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True)
class HandLandmarks:
    positions: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if len(self.positions) != 21:
            raise ValueError("A hand must contain exactly 21 landmarks")
        if any(
            len(point) != 3 or not all(math.isfinite(value) for value in point)
            for point in self.positions
        ):
            raise ValueError("Each hand landmark must contain three finite coordinates")

    def __getitem__(self, landmark: HandLandmark) -> tuple[float, float, float]:
        return self.positions[landmark]


@dataclass(frozen=True)
class FingerState:
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    @property
    def count(self) -> int:
        return sum((self.thumb, self.index, self.middle, self.ring, self.pinky))
