import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from visigui.core.hand import FingerState
from visigui.gui.main_window import HandPoseWidget, MainWindow, ProjectCard
from visigui.project.analyzer import ProjectAnalyzer
from visigui.project.explorer import ProjectAction

_APP = None


def _application():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


@pytest.fixture(autouse=True)
def _delete_qt_windows():
    yield
    app = QApplication.instance()
    if app is not None:
        for window in QApplication.topLevelWidgets():
            window.close()
            window.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_default_desktop_window_has_brand_and_camera_tracker_layout():
    _application()
    window = MainWindow(demo=True, camera_view=False)

    assert window.windowTitle() == "VisiGUI · VisiGUI"
    assert window.findChild(type(window.footer_status), "developerCredit").text() == "dev : ManiKamran"
    assert "PRESS 0–5" in window.camera_panel.image.text()
    assert window.map_view.graph is None
    window.close()


def test_project_map_renders_selectable_cards_and_click_enters_hierarchy(tmp_path):
    _application()
    (tmp_path / "alpha.py").write_text(
        "class Alpha:\n"
        "    def run(self):\n"
        "        return 1\n"
    )
    (tmp_path / "beta.py").write_text("def helper():\n    return 2\n")
    graph = ProjectAnalyzer().analyze(tmp_path)
    window = MainWindow(demo=True, camera_view=False)
    window._on_project_loaded(graph)

    cards = [
        item for item in window.map_view.scene.items()
        if isinstance(item, ProjectCard)
    ]
    assert len(cards) == len(window._controller.nodes)

    window._on_card_activated(0, True)

    assert window._controller.current_parent_id == window._controller.graph.nodes[0].node_id
    assert window.breadcrumb.text().startswith("PROJECT")
    window.close()


def test_navigation_buttons_and_keyboard_share_explorer_actions(tmp_path):
    _application()
    (tmp_path / "alpha.py").write_text("")
    (tmp_path / "beta.py").write_text("")
    graph = ProjectAnalyzer().analyze(tmp_path)
    window = MainWindow(demo=True, camera_view=False)
    window._on_project_loaded(graph)

    window.next_button.click()
    assert window._controller.selected_index == 1

    selected_id = window._controller.selected_node.node_id
    window.enter_button.click()
    assert window._controller.current_parent_id == selected_id
    window._apply_action(ProjectAction.BACK)
    assert window._controller.current_parent_id is None
    window.close()


def test_fullscreen_is_requested_when_window_is_shown():
    app = _application()
    window = MainWindow(demo=True, camera_view=False)

    window.show()
    app.processEvents()

    assert window.isFullScreen()
    window.close()


def test_project_scan_error_remains_visible_while_camera_frames_continue():
    _application()
    window = MainWindow(demo=True, camera_view=False)

    window._on_project_error("project directory is unreadable")
    window._on_camera_frame(
        QImage(2, 2, QImage.Format.Format_RGB32),
        None,
        "",
        "",
        False,
        30.0,
    )
    window._refresh_map()
    assert "PROJECT SCAN ERROR" in window.project_stats.text()
    window.close()


def test_camera_error_state_is_not_overwritten_by_worker_stopped_status():
    _application()
    window = MainWindow(demo=True, camera_view=False)

    window._on_camera_error("camera is busy")
    window._on_camera_status("Camera stopped")

    assert window.camera_panel.status.text() == "●  CAMERA ERROR"
    assert window.camera_panel.image.text() == "camera is busy"
    window.close()


def test_hand_graphic_tracks_articulated_landmarks_and_smoothly_interpolates():
    _application()
    widget = HandPoseWidget()
    start = tuple((float(index % 5), float(index // 5)) for index in range(21))
    finish = tuple((x + 20.0, y + 12.0) for x, y in start)
    widget.set_pose(FingerState(index=True), "INDEX", True, start)
    widget._animation.stop()
    widget._current_landmarks = start
    widget.set_pose(FingerState(index=True), "INDEX", True, finish)

    widget._animate_pose()

    assert widget._current_landmarks != start
    assert widget._current_landmarks != finish
    assert len(widget._current_landmarks) == 21

    widget.resize(360, 270)
    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    widget.render(painter)
    painter.end()
    assert not image.isNull()
    widget.close()


def test_live_hand_graphic_receives_camera_landmarks():
    _application()
    window = MainWindow(demo=True, camera_view=False)
    landmarks = tuple(
        (float(index * 7), float(index * 5))
        for index in range(21)
    )

    window._on_camera_frame(
        QImage(2, 2, QImage.Format.Format_RGB32),
        FingerState(index=True, middle=True),
        "INDEX_MIDDLE",
        "Right",
        True,
        30.0,
        landmarks,
    )

    assert window.hand_graphic._target_landmarks == landmarks
    window.close()
