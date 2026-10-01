from enum import Enum, auto


class Intent(Enum):
    SELECT = auto()
    OPEN = auto()
    CLOSE = auto()
    EXPAND = auto()
    COLLAPSE = auto()
    INSPECT = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    ROTATE_LEFT = auto()
    ROTATE_RIGHT = auto()
    PAN_LEFT = auto()
    PAN_RIGHT = auto()
    NONE = auto()
