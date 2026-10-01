from io import StringIO

import pytest

from visigui.core.hand import FingerState
from visigui.terminal.dashboard import (
    DashboardState,
    TerminalDashboard,
    render_dashboard,
    render_hand_art,
)


@pytest.mark.parametrize("width", (1, 7, 17, 18, 24, 40, 80, 120, 240))
@pytest.mark.parametrize("height", (1, 5, 8, 10, 11, 13, 24, 50))
def test_dashboard_respects_terminal_cell_dimensions(width, height):
    state = DashboardState(
        mode="CAMERA",
        camera_index=2,
        camera_ready=True,
        hand_detected=True,
        handedness="Right",
        finger_state=FingerState(index=True, middle=True),
        gesture_name="INDEX_MIDDLE",
        stable=True,
        intent="INSPECT",
        fps=29.97,
    )

    lines = render_dashboard(state, width, height)

    assert len(lines) <= height
    assert all(len(line) <= width for line in lines)
    assert all(line.isascii() for line in lines)


@pytest.mark.parametrize("width", (1, 18, 40, 56, 57, 72, 80, 120))
@pytest.mark.parametrize("height", (1, 8, 13, 14, 24))
def test_ascii_art_respects_terminal_dimensions(width, height):
    state = DashboardState(
        mode="CAMERA",
        hand_detected=True,
        finger_state=FingerState(index=True, middle=True),
        gesture_name="INDEX_MIDDLE",
        stable=True,
        intent="INSPECT",
        ascii_art=True,
    )

    lines = render_dashboard(state, width, height)

    assert len(lines) <= height
    assert all(len(line) <= width for line in lines)
    assert all(line.isascii() for line in lines)


def test_project_panel_replaces_status_box_and_keeps_hand_on_left():
    lines = render_dashboard(
        DashboardState(
            mode="CAMERA",
            hand_detected=True,
            finger_state=FingerState(index=True, middle=True),
            gesture_name="INDEX_MIDDLE",
            project_lines=("PROJECT MAP | sample", "2 files | 5 symbols"),
        ),
        100,
        24,
    )
    output = "\n".join(lines)
    first_project = next(line for line in lines if "PROJECT MAP" in line)

    assert "HAND 2/5" in output
    assert "PROJECT MAP" in output
    assert "2 files | 5 symbols" in output
    assert "VISIGUI / STATUS" not in output
    assert "[##]" in output
    assert first_project.index("PROJECT MAP") >= 34
    assert len(lines) <= 24
    assert all(len(line) <= 100 for line in lines)


@pytest.mark.parametrize(
    ("finger_state", "count"),
    (
        (FingerState(), "0/5"),
        (FingerState(index=True), "1/5"),
        (FingerState(index=True, middle=True), "2/5"),
        (FingerState(index=True, middle=True, ring=True), "3/5"),
        (FingerState(index=True, middle=True, ring=True, pinky=True), "4/5"),
        (FingerState(True, True, True, True, True), "5/5"),
    ),
)
def test_ascii_hand_tracks_the_exact_extended_fingers(finger_state, count):
    art_lines = render_hand_art(
        DashboardState(
            mode="CAMERA",
            hand_detected=True,
            finger_state=finger_state,
        )
    )
    art = "\n".join(art_lines)
    raised_fingers = sum((
        finger_state.index,
        finger_state.middle,
        finger_state.ring,
        finger_state.pinky,
    ))
    output = "\n".join(
        render_dashboard(
            DashboardState(
                mode="CAMERA",
                hand_detected=True,
                finger_state=finger_state,
                gesture_name="INDEX",
                ascii_art=True,
            ),
            100,
            24,
        )
    )

    assert f"HAND {count}" in art
    assert art.count("[##]") == raised_fingers
    assert (art_lines[8][0:6] == "[####]") is finger_state.thumb
    assert f"HAND {count}" in output


def test_compact_ascii_art_marks_each_finger_state():
    output = "\n".join(
        render_dashboard(
            DashboardState(
                mode="CAMERA",
                finger_state=FingerState(thumb=True, middle=True),
                gesture_name="UNKNOWN",
                ascii_art=True,
            ),
            40,
            12,
        )
    )

    assert "HAND 2/5" in output
    assert "T I M R P" in output
    assert "| _ | _ _" in output


def test_ascii_hand_waits_for_landmarks_instead_of_faking_a_pose():
    art = "\n".join(render_hand_art(DashboardState(mode="CAMERA")))

    assert "NO HAND FOUND" in art
    assert "^" not in art


def test_wide_dashboard_shows_hand_gesture_and_intent_without_scene_art():
    lines = render_dashboard(
        DashboardState(
            mode="CAMERA",
            camera_index=0,
            camera_ready=True,
            hand_detected=True,
            handedness="Left",
            finger_state=FingerState(index=True),
            gesture_name="INDEX",
            stable=True,
            intent="SELECT",
            fps=30.0,
        ),
        100,
        24,
    )
    output = "\n".join(lines)

    assert "VISIGUI" in output
    assert "DETECTED" in output
    assert "INDEX" in output
    assert "SELECT" in output
    assert "GEAR" not in output


def test_compact_dashboard_keeps_core_status_readable():
    output = "\n".join(
        render_dashboard(
            DashboardState(
                mode="DEMO",
                finger_state=FingerState(index=True, middle=True),
                gesture_name="INDEX_MIDDLE",
                stable=True,
                intent="INSPECT",
            ),
            42,
            14,
        )
    )

    assert "FINGERS: INDEX MIDDLE (2/5)" in output
    assert "GESTURE: INDEX + MIDDLE" in output
    assert "INTENT: INSPECT" in output
    assert "..." not in output


def test_dashboard_rejects_non_positive_dimensions():
    state = DashboardState(mode="CAMERA")
    with pytest.raises(ValueError, match="dimensions must be positive"):
        render_dashboard(state, 0, 24)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        render_dashboard(state, 80, 0)


def test_non_tty_present_prints_a_single_snapshot():
    stream = StringIO()
    dashboard = TerminalDashboard(stream)
    dashboard.present(
        DashboardState(mode="DEMO", gesture_name="FIST", intent="CLOSE"),
        60,
        14,
    )

    output = stream.getvalue()
    assert "KEYBOARD DEMO" in output
    assert "FIST" in output
    assert output.endswith("\n")
    assert "\x1b[" not in output


def test_tty_present_redraws_without_scrolling_and_restores_cursor():
    class TtyBuffer(StringIO):
        def isatty(self):
            return True

    stream = TtyBuffer()
    dashboard = TerminalDashboard(stream)
    dashboard.present(DashboardState(mode="CAMERA"), 80, 24)
    dashboard.present(DashboardState(mode="CAMERA", hand_detected=False), 80, 24)
    dashboard.close()

    output = stream.getvalue()
    assert output.count("\x1b[2J") == 1
    assert output.count("\x1b[H") == 2
    assert output.count("\x1b[K") == 48
    assert output.endswith("\x1b[?25h")


def test_tty_redraw_clears_stale_suffixes_when_project_rows_shrink():
    class TtyBuffer(StringIO):
        def isatty(self):
            return True

    stream = TtyBuffer()
    dashboard = TerminalDashboard(stream)
    dashboard.present(
        DashboardState(
            mode="CAMERA",
            project_lines=(
                "PROJECT MAP | sample",
                "PAGE 14/14 | SELECTED src/really_long_module_name.py:120 | Zoom src/really_long_module_name.py",
            ),
        ),
        100,
        16,
    )
    previous_length = len(stream.getvalue())
    dashboard.present(
        DashboardState(
            mode="CAMERA",
            project_lines=("PROJECT MAP | sample", "Back to map"),
        ),
        50,
        8,
    )
    redraw = stream.getvalue()[previous_length:]
    dashboard.close()

    assert redraw.count("\x1b[K") == redraw.count("\r\n") + 1
    assert redraw.count("\x1b[K") <= 8
    assert "Zoom src/really_long_module_name.py" not in redraw
    assert redraw.endswith("\x1b[K\x1b[J")
