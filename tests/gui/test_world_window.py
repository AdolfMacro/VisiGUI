import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication

from visigui.core.gesture import Gesture
from visigui.core.hand import FingerState
from visigui.gui.world_window import WorldWindow
from visigui.interaction.world_controller import (
    DEFAULT_MOTION_SPEED,
    DEFAULT_ZOOM_SPEED,
    TwoHandWorldController,
)
from visigui.vision.tracking import HandState, TwoHandState


def test_renderer_error_remains_visible_when_world_frames_continue():
    app = QApplication.instance() or QApplication([])
    window = WorldWindow(demo=True, camera_view=False)
    window._on_renderer_error("OpenGL vertex array initialization failed")

    frame = SimpleNamespace(
        camera=SimpleNamespace(position=(0.0, 0.0, 26.0), sector=(0, 0, 0)),
        hands=SimpleNamespace(left_hand=None, right_hand=None),
        hands_detected=0,
    )
    window._on_world_frame(frame)

    assert "RENDERER ERROR" in window.world_status.text()
    assert "vertex array" in window.world_status.text()
    window._world_worker.stop()
    assert window._world_worker.wait(2_000)
    window.close()
    app.processEvents()


def test_world_window_places_two_hand_views_and_camera_above_world():
    app = QApplication.instance() or QApplication([])
    window = WorldWindow(demo=True, camera_view=False)
    try:
        window.resize(1280, 800)
        window.show()
        app.processEvents()

        world_top = window.renderer.mapTo(window, QPoint(0, 0)).y()
        left_hand_top = window.left_hand_view.mapTo(window, QPoint(0, 0)).y()
        right_hand_top = window.right_hand_view.mapTo(window, QPoint(0, 0)).y()
        camera_top = window.camera_image.mapTo(window, QPoint(0, 0)).y()
        left_hand_x = window.left_hand_view.mapTo(window, QPoint(0, 0)).x()
        camera_x = window.camera_image.mapTo(window, QPoint(0, 0)).x()
        assert left_hand_top < world_top
        assert right_hand_top < world_top
        assert camera_top < world_top
        assert left_hand_x < camera_x
        assert window.renderer.geometry().height() > 0
        assert "DEMO MODE" in window.camera_image.text()
        assert window.world_panel.height() >= window.vision_panel.height() * 2.5
        assert window.camera_image.height() >= 96
    finally:
        window._world_worker.stop()
        window._world_worker.wait(2_000)
        window.close()
        app.processEvents()


def test_hand_vision_panels_update_from_stable_left_and_right_states():
    app = QApplication.instance() or QApplication([])
    window = WorldWindow(demo=True, camera_view=False)
    landmarks = tuple((0.25 + index * 0.01, 0.3 + index * 0.005, 0.0) for index in range(21))

    left = HandState(
        "Left", 0.95, landmarks, (0.4, 0.5, 0.0), (0.4, 0.5, 0.2),
        (0.0, 0.0), Gesture.INDEX, 0.6, FingerState(index=True),
    )
    right = HandState(
        "Right", 0.92, landmarks, (0.6, 0.5, 0.0), (0.6, 0.5, 0.25),
        (0.0, 0.0), Gesture.PINCH, 0.1, FingerState(thumb=True, index=True),
    )
    try:
        window._on_hands(TwoHandState(left, right, 1.0))

        assert window.left_hand_graphic.detected
        assert window.left_hand_graphic.gesture_name == Gesture.INDEX.name
        assert window.left_hand_graphic._target_landmarks[0] == landmarks[0][:2]
        assert window.right_hand_graphic.detected
        assert window.right_hand_graphic.gesture_name == Gesture.PINCH.name
        assert window.right_hand_graphic._target_landmarks[0] == landmarks[0][:2]
    finally:
        window._world_worker.stop()
        window._world_worker.wait(2_000)
        window.close()
        app.processEvents()


def test_keyboard_and_hand_movement_and_zoom_use_identical_speed():
    app = QApplication.instance() or QApplication([])
    window = WorldWindow(demo=True, camera_view=False)
    right = HandState(
        "Right", 0.95, ((0.5, 0.5, 0.0),) * 21, (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.2), (0.0, 0.0), Gesture.OPEN_PALM, 0.3,
        FingerState(thumb=True, index=True),
    )
    try:
        window._held_keys.update((Qt.Key.Key_D, Qt.Key.Key_E))
        keyboard_motion = window._keyboard_motion()
        hand_controller = TwoHandWorldController(pinch_activation_seconds=0.0)
        hand_controller.update(TwoHandState(None, right, 1.0), 1 / 30)
        hand_speed = hand_controller.update(
            TwoHandState(None, right, 1.03),
            1 / 30,
        ).depth

        left = HandState(
            "Left", 0.95, ((0.5, 0.5, 0.0),) * 21, (0.5, 0.5, 0.0),
            (0.5, 0.5, 0.2), (0.0, 0.0), Gesture.FIST, 0.3,
            FingerState(),
        )
        navigation_controller = TwoHandWorldController()
        navigation_controller.update(TwoHandState(left, None, 1.0), 1 / 30)
        navigation_speed = navigation_controller.update(
            TwoHandState(
                HandState(
                    "Left", 0.95, ((0.5, 0.5, 0.0),) * 21, (0.5, 0.5, 0.0),
                    (1.0, 0.5, 0.2), (0.0, 0.0), Gesture.FIST, 0.3,
                    FingerState(),
                ),
                None,
                1.03,
            ),
            1 / 30,
        ).strafe

        assert DEFAULT_ZOOM_SPEED == DEFAULT_MOTION_SPEED == 50.0
        assert keyboard_motion.strafe == keyboard_motion.depth == DEFAULT_MOTION_SPEED
        assert navigation_controller.max_navigation_speed == DEFAULT_MOTION_SPEED
        assert navigation_speed == hand_speed == DEFAULT_MOTION_SPEED
    finally:
        window._world_worker.stop()
        window._world_worker.wait(2_000)
        window.close()
        app.processEvents()
