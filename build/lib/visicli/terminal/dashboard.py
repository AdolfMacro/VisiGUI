from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional, TextIO

from ..core.hand import FingerState

_FINGER_NAMES = (
    ("thumb", "T", "THUMB"),
    ("index", "I", "INDEX"),
    ("middle", "M", "MIDDLE"),
    ("ring", "R", "RING"),
    ("pinky", "P", "PINKY"),
)
_GESTURE_LABELS = {
    "FIST": "FIST",
    "OPEN_PALM": "OPEN PALM",
    "INDEX": "INDEX",
    "THUMB_INDEX": "THUMB + INDEX",
    "INDEX_MIDDLE": "INDEX + MIDDLE",
    "INDEX_MIDDLE_RING": "INDEX + MIDDLE + RING",
    "INDEX_MIDDLE_RING_PINKY": "INDEX + MIDDLE + RING + PINKY",
    "UNKNOWN": "UNSUPPORTED SHAPE",
}
_HAND_ART_WIDTH = 31


def render_hand_art(state: DashboardState) -> tuple[str, ...]:
    """Draw an original, pose-driven ASCII hand using the observed finger states."""
    rows = [[" "] * _HAND_ART_WIDTH for _ in range(15)]

    def put(row: int, column: int, text: str) -> None:
        if 0 <= row < len(rows):
            for offset, char in enumerate(text):
                target = column + offset
                if 0 <= target < _HAND_ART_WIDTH:
                    rows[row][target] = char

    count, _ = _finger_summary(state.finger_state)
    put(0, 0, f"HAND {count}")

    if state.finger_state is None:
        if state.hand_detected:
            put(5, 2, "HAND GEOMETRY INVALID")
            put(7, 2, "REPOSITION HAND")
        else:
            put(5, 3, "NO HAND FOUND")
            put(7, 2, "HOLD HAND IN VIEW")
        return tuple("".join(row).rstrip() for row in rows)

    finger_values = (
        state.finger_state.index,
        state.finger_state.middle,
        state.finger_state.ring,
        state.finger_state.pinky,
    )
    centers = (8, 13, 18, 23)
    heights = (2, 0, 1, 3)
    palm_top = 6
    for extended, center, top in zip(finger_values, centers, heights):
        if extended:
            put(top, center - 2, "[##]")
            for row in range(top + 1, palm_top):
                put(row, center - 2, "|##|")
        else:
            put(5, center - 2, "(__)")

    put(6, 6, "+##################+")
    for row in range(7, 10):
        put(row, 6, "|##################|")
    put(10, 6, "+##################+")

    if state.finger_state.thumb:
        put(7, 2, "\\##")
        put(8, 0, "[####]")
        put(9, 2, "/##")
    else:
        put(8, 1, "(__)")

    put(11, 10, "|##########|")
    put(12, 10, "|##########|")
    put(13, 10, "|##########|")
    put(14, 10, "'----------'")
    return tuple("".join(row).rstrip() for row in rows)


@dataclass(frozen=True)
class DashboardState:
    mode: str
    camera_index: int = 0
    camera_ready: bool = False
    hand_detected: Optional[bool] = None
    handedness: Optional[str] = None
    finger_state: Optional[FingerState] = None
    gesture_name: Optional[str] = None
    stable: bool = False
    intent: Optional[str] = None
    fps: Optional[float] = None
    frame_time_ms: Optional[float] = None
    debug: bool = False
    ascii_art: bool = False
    project_lines: Optional[tuple[str, ...]] = None


def _gesture_label(name: Optional[str], mode: str = "CAMERA") -> str:
    if name is None:
        return "PRESS 0-5" if mode == "DEMO" else "WAITING FOR HAND"
    return _GESTURE_LABELS.get(name, name.replace("_", " "))


def _finger_summary(state: Optional[FingerState]) -> tuple[str, str]:
    if state is None:
        return "--", "--"
    values = tuple(getattr(state, attribute) for attribute, _, _ in _FINGER_NAMES)
    detail = " ".join(
        f"{code}:{'ON' if value else 'OFF'}"
        for value, (_, code, _) in zip(values, _FINGER_NAMES)
    )
    return f"{sum(values)}/5", detail


def _input_label(state: DashboardState) -> str:
    if state.mode == "DEMO":
        return "KEYBOARD DEMO"
    camera = f"CAMERA {state.camera_index}"
    return f"{camera} READY" if state.camera_ready else f"{camera} STARTING"


def _hand_label(state: DashboardState) -> str:
    if state.mode == "DEMO":
        return "KEYBOARD INPUT"
    if state.hand_detected is None:
        return "WAITING"
    return "DETECTED" if state.hand_detected else "NOT DETECTED"


def _fps_label(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value) or value <= 0:
        return "--"
    return f"{value:.1f}"


def _wide_rows(state: DashboardState) -> list[str]:
    finger_count, fingers = _finger_summary(state.finger_state)
    stable = "BYPASS" if state.mode == "DEMO" and state.gesture_name else (
        "STABLE" if state.stable else "WAITING"
    )
    rate = f"{_fps_label(state.fps)} FPS"
    if state.debug and state.frame_time_ms is not None and math.isfinite(state.frame_time_ms):
        rate += f" | {state.frame_time_ms:.1f} MS"
    side = state.handedness or "--"
    if state.mode != "DEMO" and state.hand_detected is False:
        side = "--"
    return [
        f"INPUT      {_input_label(state)}",
        f"HAND       {_hand_label(state)}   SIDE: {side}",
        f"FINGERS    {fingers}   COUNT: {finger_count}",
        f"GESTURE    {_gesture_label(state.gesture_name, state.mode)}",
        f"STABILITY  {stable}",
        f"LAST INTENT {state.intent or '--'}",
        f"RATE       {rate}",
    ]


def _compact_rows(state: DashboardState, inner_width: int) -> list[str]:
    count, _ = _finger_summary(state.finger_state)
    hand = "DEMO INPUT" if state.mode == "DEMO" else (
        "HAND YES" if state.hand_detected else
        "HAND NO" if state.hand_detected is False else "HAND WAIT"
    )
    if inner_width >= 30:
        detail = " ".join(
            label
            for enabled, (_, _, label) in zip(
                tuple(getattr(state.finger_state, attribute) for attribute, _, _ in _FINGER_NAMES)
                if state.finger_state else (False,) * 5,
                _FINGER_NAMES,
            )
            if enabled
        ) or "--"
        side = f" | {state.handedness}" if state.handedness else ""
        stable = "BYPASS" if state.mode == "DEMO" and state.gesture_name else (
            "STABLE" if state.stable else "WAITING"
        )
        rows = [
            f"INPUT: {_input_label(state)}",
            f"{hand}{side if state.mode != 'DEMO' else ''}",
            f"FINGERS: {detail} ({count})" if state.finger_state else "FINGERS: --",
            f"GESTURE: {_gesture_label(state.gesture_name, state.mode)}",
            f"{stable} | INTENT: {state.intent or '--'}",
        ]
    else:
        count_label = count if count != "--" else "?"
        stable = "YES" if state.stable or (state.mode == "DEMO" and state.gesture_name) else "NO"
        rows = [
            f"{state.mode} {hand}",
            f"FINGERS {count_label}",
            f"GESTURE {_gesture_label(state.gesture_name, state.mode)}",
            f"STABLE {stable}",
            f"INTENT {state.intent or '--'}",
        ]
    return rows


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _art_dashboard(state: DashboardState, width: int, height: int) -> tuple[str, ...]:
    usable_width = width - 1
    art = render_hand_art(state)
    rows = _compact_rows(state, max(1, usable_width - _HAND_ART_WIDTH - 3))
    status = [
        "+" + "-" * (usable_width - _HAND_ART_WIDTH - 4) + "+",
        "| " + _fit("VISICLI / STATUS", usable_width - _HAND_ART_WIDTH - 6).ljust(
            usable_width - _HAND_ART_WIDTH - 6
        ) + " |",
        "+" + "-" * (usable_width - _HAND_ART_WIDTH - 4) + "+",
    ]
    status.extend(
        "| " + _fit(row, usable_width - _HAND_ART_WIDTH - 6).ljust(
            usable_width - _HAND_ART_WIDTH - 6
        ) + " |"
        for row in rows
    )
    controls = "0-5: SHAPES | Q: QUIT" if state.mode == "DEMO" else "Q: QUIT"
    status.extend((
        "+" + "-" * (usable_width - _HAND_ART_WIDTH - 4) + "+",
        "| " + _fit(controls, usable_width - _HAND_ART_WIDTH - 6).ljust(
            usable_width - _HAND_ART_WIDTH - 6
        ) + " |",
        "+" + "-" * (usable_width - _HAND_ART_WIDTH - 4) + "+",
    ))
    height_needed = max(len(art), len(status))
    if height_needed <= height:
        top = (height - height_needed) // 2
        canvas = [""] * height
        for index in range(height_needed):
            art_line = art[index] if index < len(art) else ""
            status_line = status[index] if index < len(status) else ""
            canvas[top + index] = f"{art_line:<{_HAND_ART_WIDTH}}   {status_line}".rstrip()
        return tuple(canvas)

    stacked = [*art, *status]
    return tuple(_fit(line, usable_width) for line in stacked[:height])


def _project_dashboard(state: DashboardState, width: int, height: int) -> tuple[str, ...]:
    usable_width = width - 1
    input_panel = _input_panel(state, _HAND_ART_WIDTH)
    panel_lines = state.project_lines or ()
    if usable_width >= _HAND_ART_WIDTH + 24 and height >= 20:
        panel_width = usable_width - _HAND_ART_WIDTH - 3
        canvas = [""] * height
        panel_top = max(0, (height - len(input_panel)) // 2)
        for index in range(height):
            hand_line = (
                input_panel[index - panel_top]
                if panel_top <= index < panel_top + len(input_panel)
                else ""
            )
            project_line = panel_lines[index] if index < len(panel_lines) else ""
            canvas[index] = (
                f"{hand_line:<{_HAND_ART_WIDTH}}   {_fit(project_line, panel_width)}"
            )
        return tuple(_fit(line, usable_width) for line in canvas)

    compact_status = _compact_input_status(state, usable_width, height)
    combined = [*compact_status, *panel_lines]
    return tuple(_fit(line, usable_width) for line in combined[:height])


def _input_panel(state: DashboardState, width: int) -> tuple[str, ...]:
    inner_width = max(1, width - 4)
    title = "┌─ HAND "
    top = title + "─" * max(0, width - len(title) - 1) + "┐"
    art = render_hand_art(state)
    status = (
        f"POSE {_gesture_label(state.gesture_name, state.mode)}",
        f"ACTION {_fit(state.intent or '--', max(1, inner_width - 7))}",
        _input_label(state),
    )
    return (
        top,
        *(
            f"│ {_fit(row, inner_width).ljust(inner_width)} │"
            for row in art
        ),
        *(f"│ {_fit(row, inner_width).ljust(inner_width)} │" for row in status),
        "└" + "─" * (width - 2) + "┘",
    )


def _compact_input_status(
    state: DashboardState,
    width: int,
    height: int,
) -> tuple[str, ...]:
    count, _ = _finger_summary(state.finger_state)
    rows = (
        f"HAND {count} {_hand_label(state)}",
        f"POSE {_gesture_label(state.gesture_name, state.mode)}",
        f"ACTION {state.intent or '--'}",
    )
    return tuple(_fit(row, width) for row in rows[:height])


def _compact_art_dashboard(state: DashboardState, width: int, height: int) -> tuple[str, ...]:
    count, _ = _finger_summary(state.finger_state)
    values = (
        state.finger_state.thumb,
        state.finger_state.index,
        state.finger_state.middle,
        state.finger_state.ring,
        state.finger_state.pinky,
    ) if state.finger_state else (False,) * 5
    marks = " ".join("|" if active else "_" for active in values)
    rows = [
        f"HAND {count}",
        "T I M R P",
        marks,
        "+-------+",
        f"GESTURE {_gesture_label(state.gesture_name, state.mode)}",
        f"INTENT {state.intent or '--'}",
        "0-5: SHAPES | Q: QUIT" if state.mode == "DEMO" else "Q: QUIT",
    ]
    return tuple(_fit(line, width) for line in rows[:height])


def render_dashboard(state: DashboardState, width: int, height: int) -> tuple[str, ...]:
    """Render a cell-sized dashboard; terminal font pixels are intentionally irrelevant."""
    if width <= 0 or height <= 0:
        raise ValueError("Dashboard dimensions must be positive")

    usable_width = width - 1
    if state.project_lines is not None:
        return _project_dashboard(state, width, height)
    if state.ascii_art:
        if usable_width >= _HAND_ART_WIDTH + 28 and height >= 14:
            return _art_dashboard(state, width, height)
        return _compact_art_dashboard(state, usable_width, height)
    if usable_width < 18 or height < 11:
        count, _ = _finger_summary(state.finger_state)
        rows = [
            "VISICLI",
            _input_label(state),
            _hand_label(state),
            f"FINGERS {count}",
            _gesture_label(state.gesture_name, state.mode),
            f"INTENT {state.intent or '--'}",
            "0-5 INPUT  Q QUIT" if state.mode == "DEMO" else "Q QUIT",
        ]
        visible = rows[:height]
        return tuple(_fit(row, usable_width) for row in visible)

    panel_width = min(usable_width, 88)
    inner_width = panel_width - 4
    if inner_width >= 52 and height >= 13:
        rows = _wide_rows(state)
    else:
        rows = _compact_rows(state, inner_width)

    if state.mode == "DEMO" and inner_width >= 42:
        controls = "KEYS  0-5: TEST HAND SHAPES   |   Q: QUIT"
    elif state.mode == "DEMO":
        controls = "0-5: SHAPES  |  Q: QUIT"
    elif inner_width >= 42:
        controls = "Q: QUIT   |   HOLD ONE HAND CLEARLY IN VIEW"
    else:
        controls = "Q: QUIT  |  HOLD HAND IN VIEW"

    title = "VISICLI  /  HAND CONTROL"
    panel = [
        "+" + "-" * (panel_width - 2) + "+",
        "| " + _fit(title, inner_width).center(inner_width) + " |",
        "+" + "-" * (panel_width - 2) + "+",
    ]
    panel.extend("| " + _fit(row, inner_width).ljust(inner_width) + " |" for row in rows)
    panel.append("+" + "-" * (panel_width - 2) + "+")
    panel.append("| " + _fit(controls, inner_width).ljust(inner_width) + " |")
    panel.append("+" + "-" * (panel_width - 2) + "+")
    panel = panel[:height]

    canvas = [""] * height
    top = (height - len(panel)) // 2
    left = (usable_width - panel_width) // 2
    for index, line in enumerate(panel):
        canvas[top + index] = " " * left + line
    return tuple(canvas)


class TerminalDashboard:
    def __init__(self, stream: Optional[TextIO] = None):
        self.stream = stream
        self._terminal_active = False

    def present(self, state: DashboardState, width: int, height: int) -> tuple[str, ...]:
        stream = self.stream if self.stream is not None else sys.stdout
        lines = render_dashboard(state, width, height)
        if not stream.isatty():
            stream.write("\n".join(line.rstrip() for line in lines).rstrip() + "\n")
            stream.flush()
            return lines

        if not self._terminal_active:
            stream.write("\x1b[2J\x1b[?25l")
            self._terminal_active = True
        stream.write(
            "\x1b[H"
            + "\x1b[K\r\n".join(line.rstrip() for line in lines)
            + "\x1b[K\x1b[J"
        )
        stream.flush()
        return lines

    def close(self) -> None:
        if self._terminal_active:
            stream = self.stream if self.stream is not None else sys.stdout
            stream.write("\x1b[?25h")
            stream.flush()
            self._terminal_active = False

    def __enter__(self) -> TerminalDashboard:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
