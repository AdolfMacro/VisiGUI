from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap, QSurfaceFormat
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..interaction.world_controller import DEFAULT_MOTION_SPEED, WorldMotion
from ..renderer.opengl_world import GenerativeWorldView
from ..terminal.log_capture import StderrLogCapture
from ..vision.detector import DEFAULT_MODEL_PATH
from ..vision.tracking import HandState, TwoHandState
from ..world.simulation import WorldFrame
from .main_window import HandPoseWidget
from .workers import CameraWorker, WorldSimulationWorker


_COLORS = {
    "background": "#080D17",
    "surface": "#101827",
    "border": "#26344A",
    "text": "#EEF4FF",
    "muted": "#9AAAC1",
    "cyan": "#42E4D2",
    "blue": "#5BA8FF",
    "violet": "#A98BFF",
    "orange": "#FFB86B",
    "red": "#FF647C",
    "green": "#5DE0A0",
}


class WorldWindow(QMainWindow):
    """Full-screen world view with live hand-vision and camera-tracker panels."""

    def __init__(
        self,
        camera_index: int = 0,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        camera_view: bool = True,
        demo: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("VisiGUI · Generative World")
        self.setMinimumSize(960, 640)
        self._camera_index = camera_index
        self._model_path = str(model_path)
        self._camera_view = camera_view
        self._demo = demo
        self._closing = False
        self._camera_failed = False
        self._renderer_error_message: Optional[str] = None
        self._camera_worker: Optional[CameraWorker] = None
        self._world_worker = WorldSimulationWorker(parent=self)
        self._held_keys: set[int] = set()
        self._last_frame_time = time.monotonic()
        self._display_fps = 30.0

        self.renderer = GenerativeWorldView()
        self.renderer.renderer_error.connect(self._on_renderer_error)
        self.setCentralWidget(self._build_surface())
        self._apply_style()

        self._world_worker.frame_ready.connect(self._on_world_frame)
        self._world_worker.start()
        if demo:
            self.camera_status.setText("DEMO MODE · CAMERA OFF")
            self.hands_status.setText("KEYBOARD FLIGHT · WASD / ARROWS")
        else:
            self._start_camera()

    def _build_surface(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        upper_panels = QHBoxLayout()
        upper_panels.setSpacing(12)

        self.vision_panel = QFrame()
        self.vision_panel.setObjectName("panel")
        vision_layout = QVBoxLayout(self.vision_panel)
        vision_layout.setContentsMargins(16, 12, 16, 12)
        vision_layout.setSpacing(8)
        vision_header = QHBoxLayout()
        vision_title = QLabel("HAND VISION")
        vision_title.setObjectName("panelTitle")
        vision_header.addWidget(vision_title)
        vision_header.addStretch(1)
        self.hands_status = QLabel("SEARCHING FOR HANDS")
        self.hands_status.setObjectName("status")
        vision_header.addWidget(self.hands_status)
        vision_layout.addLayout(vision_header)

        hand_views = QHBoxLayout()
        hand_views.setSpacing(8)
        self.left_hand_view, self.left_hand_graphic = self._hand_view("LEFT HAND · FIST TO MOVE")
        self.right_hand_view, self.right_hand_graphic = self._hand_view("RIGHT HAND · TOUCH OUT / SPREAD IN")
        hand_views.addWidget(self.left_hand_view, 1)
        hand_views.addWidget(self.right_hand_view, 1)
        vision_layout.addLayout(hand_views, 1)
        upper_panels.addWidget(self.vision_panel, 1)

        self.tracker_panel = QFrame()
        self.tracker_panel.setObjectName("panel")
        tracker_layout = QVBoxLayout(self.tracker_panel)
        tracker_layout.setContentsMargins(16, 12, 16, 12)
        tracker_layout.setSpacing(8)
        tracker_header = QHBoxLayout()
        tracker_title = QLabel("LIVE HAND TRACKER")
        tracker_title.setObjectName("panelTitle")
        tracker_header.addWidget(tracker_title)
        tracker_header.addStretch(1)
        self.camera_status = QLabel("STARTING CAMERA")
        self.camera_status.setObjectName("status")
        tracker_header.addWidget(self.camera_status)
        tracker_layout.addLayout(tracker_header)
        self.camera_image = QLabel("WAITING FOR CAMERA FRAME")
        self.camera_image.setObjectName("cameraImage")
        self.camera_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_image.setMinimumSize(240, 96)
        self.camera_image.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        tracker_layout.addWidget(self.camera_image, 1)
        upper_panels.addWidget(self.tracker_panel, 1)
        layout.addLayout(upper_panels, 1)

        self.world_panel = QFrame()
        self.world_panel.setObjectName("worldPanel")
        world_layout = QVBoxLayout(self.world_panel)
        world_layout.setContentsMargins(16, 10, 16, 10)
        world_layout.setSpacing(6)
        world_header = QHBoxLayout()
        brand = QLabel("VisiGUI")
        brand.setObjectName("brand")
        subbrand = QLabel("TWO-HAND GENERATIVE WORLD")
        subbrand.setObjectName("eyebrow")
        title_stack = QVBoxLayout()
        title_stack.setSpacing(1)
        title_stack.addWidget(brand)
        title_stack.addWidget(subbrand)
        world_header.addLayout(title_stack)
        world_header.addStretch(1)
        self.world_status = QLabel("GENERATING WORLD")
        self.world_status.setObjectName("muted")
        world_header.addWidget(self.world_status)
        world_layout.addLayout(world_header)
        self.renderer.setMinimumSize(320, 180)
        self.renderer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        world_layout.addWidget(self.renderer, 1)

        hints = QLabel(
            "LEFT HAND · HOLD FIST + MOVE     RIGHT HAND · TOUCH = OUT · SPREAD = IN     "
            "WASD / ARROWS · FLIGHT     ESC · EXIT"
        )
        hints.setObjectName("hints")
        hints.setAlignment(Qt.AlignmentFlag.AlignCenter)
        world_layout.addWidget(hints)
        layout.addWidget(self.world_panel, 3)
        self.camera_image.setText(
            "DEMO MODE · CAMERA OFF" if self._demo
            else "PREVIEW HIDDEN · TRACKING ACTIVE" if not self._camera_view
            else "WAITING FOR CAMERA FRAME"
        )
        return root

    @staticmethod
    def _hand_view(title: str) -> tuple[QWidget, HandPoseWidget]:
        panel = QFrame()
        panel.setObjectName("handPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("eyebrow")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        hand = HandPoseWidget()
        hand.setMinimumSize(160, 96)
        hand.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(hand, 1)
        return panel, hand

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{ background: {_COLORS['background']}; color: {_COLORS['text']}; }}
            QFrame#panel, QFrame#worldPanel {{ background: {_COLORS['surface']}; border: 1px solid {_COLORS['border']}; border-radius: 14px; }}
            QFrame#handPanel {{ background: rgba(8, 13, 23, 150); border: 1px solid {_COLORS['border']}; border-radius: 10px; }}
            QLabel#brand {{ color: {_COLORS['text']}; font-size: 23px; font-weight: 900; letter-spacing: 4px; background: transparent; }}
            QLabel#eyebrow {{ color: {_COLORS['cyan']}; font-size: 9px; font-weight: 800; letter-spacing: 2px; background: transparent; }}
            QLabel#panelTitle {{ color: {_COLORS['text']}; font-size: 11px; font-weight: 900; letter-spacing: 2px; background: transparent; }}
            QLabel#status {{ color: {_COLORS['green']}; font-size: 10px; font-weight: 800; letter-spacing: 1.2px; background: transparent; }}
            QLabel#muted {{ color: {_COLORS['muted']}; font-size: 10px; font-weight: 700; background: transparent; }}
            QLabel#hints {{ color: {_COLORS['muted']}; font-size: 9px; font-weight: 700; letter-spacing: 1px; background: transparent; }}
            QLabel#cameraImage {{ color: {_COLORS['muted']}; background: {_COLORS['background']}; border: 1px solid {_COLORS['cyan']}; border-radius: 10px; font-size: 10px; }}
            """
        )

    def _start_camera(self) -> None:
        self.camera_status.setText(f"OPENING CAMERA {self._camera_index}")
        self.camera_status.setStyleSheet(f"color: {_COLORS['orange']};")
        worker = CameraWorker(
            self._camera_index,
            self._model_path,
            self,
            preview_enabled=self._camera_view,
        )
        worker.frame_ready.connect(self._on_camera_frame)
        worker.hands_ready.connect(self._on_hands)
        worker.status_changed.connect(self._on_camera_status)
        worker.error_occurred.connect(self._on_camera_error)
        self._camera_worker = worker
        worker.start()

    def _on_camera_frame(
        self,
        image: QImage,
        finger_state,
        gesture_name: str,
        handedness: str,
        detected: bool,
        fps: float,
        landmark_positions,
    ) -> None:
        if not self._camera_view:
            return
        self.camera_image.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.camera_image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_hands(self, hands: TwoHandState) -> None:
        self._world_worker.set_hands(hands)
        self._update_hand_pose(self.left_hand_graphic, hands.left_hand)
        self._update_hand_pose(self.right_hand_graphic, hands.right_hand)
        if not hands.hands_detected:
            self.hands_status.setText("SEARCHING FOR HANDS")
            self.hands_status.setStyleSheet(f"color: {_COLORS['orange']};")

    @staticmethod
    def _update_hand_pose(widget: HandPoseWidget, hand: Optional[HandState]) -> None:
        if hand is None:
            widget.set_pose(None, "", False)
            return
        widget.set_pose(
            hand.finger_state,
            hand.gesture.name,
            True,
            tuple((x, y) for x, y, _ in hand.landmarks),
        )

    def _on_camera_status(self, status: str) -> None:
        if status == "Camera connected":
            self._camera_failed = False
            self.camera_status.setText("CAMERA ONLINE")
            self.camera_status.setStyleSheet(f"color: {_COLORS['green']};")

    def _on_camera_error(self, message: str) -> None:
        self._camera_failed = True
        self.camera_status.setText(f"CAMERA / TRACKING ERROR · {message}")
        self.camera_status.setStyleSheet(f"color: {_COLORS['red']};")
        if self._camera_view:
            self.camera_image.setText(message)

    def _on_renderer_error(self, message: str) -> None:
        self._renderer_error_message = message
        self.world_status.setText(f"RENDERER ERROR · {message}")
        self.world_status.setStyleSheet(f"color: {_COLORS['red']};")

    def _on_world_frame(self, frame: WorldFrame) -> None:
        frame = self._world_worker.take_latest_frame(frame)
        self.renderer.set_world_frame(frame)
        now = time.monotonic()
        dt = max(now - self._last_frame_time, 1e-6)
        self._last_frame_time = now
        self._display_fps = (
            self._display_fps * 0.85 + min(120.0, 1.0 / dt) * 0.15
        )
        snapshot = frame.camera
        if self._renderer_error_message is not None:
            return
        self.world_status.setText(
            f"{self._display_fps:4.0f} FPS  ·  SECTOR "
            f"{snapshot.sector[0]:+d} / {snapshot.sector[1]:+d} / {snapshot.sector[2]:+d}"
        )
        hands = frame.hands
        parts = []
        for label, hand in (("LEFT", hands.left_hand), ("RIGHT", hands.right_hand)):
            if hand is None:
                parts.append(f"{label} HAND LOST")
            else:
                parts.append(f"{label} {hand.gesture.name} · {hand.confidence:.0%}")
        self.hands_status.setText("     ".join(parts))
        self.hands_status.setStyleSheet(
            f"color: {_COLORS['green'] if frame.hands_detected else _COLORS['orange']};"
        )

    def _keyboard_motion(self) -> WorldMotion:
        keys = self._held_keys
        return WorldMotion(
            strafe=DEFAULT_MOTION_SPEED * (
                float(Qt.Key.Key_D in keys or Qt.Key.Key_Right in keys)
                - float(Qt.Key.Key_A in keys or Qt.Key.Key_Left in keys)
            ),
            vertical=DEFAULT_MOTION_SPEED * (
                float(Qt.Key.Key_W in keys or Qt.Key.Key_Up in keys)
                - float(Qt.Key.Key_S in keys or Qt.Key.Key_Down in keys)
            ),
            depth=DEFAULT_MOTION_SPEED * (
                float(Qt.Key.Key_E in keys or Qt.Key.Key_PageUp in keys)
                - float(Qt.Key.Key_Q in keys or Qt.Key.Key_PageDown in keys)
            ),
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        if event.key() in (
            Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D,
            Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right,
            Qt.Key.Key_E, Qt.Key.Key_Q, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
        ):
            self._held_keys.add(event.key())
            self._world_worker.set_keyboard_motion(self._keyboard_motion())
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        self._held_keys.discard(event.key())
        self._world_worker.set_keyboard_motion(self._keyboard_motion())
        super().keyReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "camera_image"):
            self.camera_image.move(self.width() - self.camera_image.width() - 28, 72)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.renderer.setFocus()

    def closeEvent(self, event) -> None:
        running = self._world_worker.isRunning() or (
            self._camera_worker is not None and self._camera_worker.isRunning()
        )
        if running:
            self._closing = True
            self._world_worker.stop()
            if self._camera_worker is not None:
                self._camera_worker.stop()
            QTimer.singleShot(40, self._finish_close)
            event.ignore()
            return
        self.renderer.cleanup()
        event.accept()

    def _finish_close(self) -> None:
        if self._world_worker.isRunning() or (
            self._camera_worker is not None and self._camera_worker.isRunning()
        ):
            QTimer.singleShot(40, self._finish_close)
        else:
            self.close()


def launch_world_gui(
    camera_index: int = 0,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    camera_view: bool = True,
    demo: bool = False,
) -> int:
    with StderrLogCapture():
        surface = QSurfaceFormat()
        surface.setVersion(3, 3)
        surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface.setDepthBufferSize(24)
        surface.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(surface)
        app = QApplication.instance() or QApplication([])
        app.setApplicationName("VisiGUI")
        app.setOrganizationName("ManiKamran")
        window = WorldWindow(
            camera_index=camera_index,
            model_path=model_path,
            camera_view=camera_view,
            demo=demo,
        )
        window.showFullScreen()
        return app.exec()
