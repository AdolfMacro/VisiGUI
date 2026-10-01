from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.gesture import Gesture
from ..core.hand import FingerState
from ..project.explorer import ExplorerController, ProjectAction
from ..project.model import ProjectGraph, ProjectNode
from ..terminal.log_capture import StderrLogCapture
from ..vision.detector import DEFAULT_MODEL_PATH
from .workers import CameraWorker, ProjectScanWorker


_COLORS = {
    "background": "#080D17",
    "surface": "#101827",
    "surface_alt": "#151F31",
    "border": "#26344A",
    "text": "#EEF4FF",
    "muted": "#8494AD",
    "cyan": "#42E4D2",
    "blue": "#5BA8FF",
    "violet": "#A98BFF",
    "orange": "#FFB86B",
    "red": "#FF647C",
    "green": "#5DE0A0",
}

_KIND_COLORS = {
    "package": _COLORS["violet"],
    "file": _COLORS["cyan"],
    "class": _COLORS["blue"],
    "function": _COLORS["orange"],
}

_DEMO_GESTURES = {
    Qt.Key.Key_0: (Gesture.FIST, FingerState()),
    Qt.Key.Key_1: (Gesture.INDEX, FingerState(index=True)),
    Qt.Key.Key_2: (Gesture.THUMB_INDEX, FingerState(thumb=True, index=True)),
    Qt.Key.Key_3: (Gesture.INDEX_MIDDLE_RING, FingerState(index=True, middle=True, ring=True)),
    Qt.Key.Key_4: (
        Gesture.INDEX_MIDDLE_RING_PINKY,
        FingerState(index=True, middle=True, ring=True, pinky=True),
    ),
    Qt.Key.Key_5: (Gesture.OPEN_PALM, FingerState(True, True, True, True, True)),
}


def _kind_color(kind: str) -> QColor:
    return QColor(_KIND_COLORS.get(kind, _COLORS["muted"]))


class BrandMark(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(62, 62)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(3, 3, 56, 56)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor(_COLORS["cyan"]))
        gradient.setColorAt(1, QColor(_COLORS["violet"]))
        painter.setPen(QPen(QColor(_COLORS["cyan"]), 1.5))
        painter.setBrush(gradient)
        painter.drawEllipse(rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(_COLORS["cyan"]), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(13, 15, 36, 31), 25 * 16, 130 * 16)
        painter.drawArc(QRectF(13, 15, 36, 31), 205 * 16, 130 * 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_COLORS["violet"]))
        painter.drawEllipse(QPointF(31, 30), 5.5, 5.5)
        painter.setBrush(QColor(_COLORS["cyan"]))
        painter.drawEllipse(QPointF(31, 30), 2, 2)


class HandPoseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.finger_state: Optional[FingerState] = None
        self.gesture_name = ""
        self.detected = False
        self._target_landmarks = self._make_pose(FingerState(True, True, True, True, True))
        self._current_landmarks = self._target_landmarks
        self._last_live_landmark_time: Optional[float] = None
        self._last_animation_time = time.monotonic()
        self._animation = QTimer(self)
        self._animation.setInterval(16)
        self._animation.timeout.connect(self._animate_pose)
        self.setMinimumSize(300, 230)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_pose(
        self,
        finger_state: Optional[FingerState],
        gesture_name: str,
        detected: bool,
        landmark_positions: Optional[tuple[tuple[float, float], ...]] = None,
    ) -> None:
        self.finger_state = finger_state
        self.gesture_name = gesture_name
        self.detected = detected
        if landmark_positions is not None and len(landmark_positions) == 21:
            now = time.monotonic()
            if (
                self._last_live_landmark_time is not None
                and now - self._last_live_landmark_time > 0.35
            ):
                self._current_landmarks = landmark_positions
            target = landmark_positions
            self._last_live_landmark_time = now
        else:
            if self._last_live_landmark_time is None:
                target = self._make_pose(
                    finger_state or FingerState(True, True, True, True, True)
                )
            else:
                target = self._target_landmarks
        if not self._animation.isActive():
            self._last_animation_time = time.monotonic()
            self._animation.start()
        self._target_landmarks = target
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = QRectF(self.rect())
        painter.fillRect(area, QColor(_COLORS["surface"]))
        scale = min((area.width() - 28) / 340, (area.height() - 72) / 250)
        painter.save()
        painter.translate((area.width() - 340 * scale) / 2, 8)
        painter.scale(scale, scale)
        self._draw_hand(
            painter,
            self._fit_landmarks(self._current_landmarks),
            1.0 if self.detected else 0.28,
        )
        painter.restore()
        painter.setPen(QColor(_COLORS["text"] if self.gesture_name else _COLORS["muted"]))
        painter.setFont(QFont("Inter", 10, QFont.Weight.DemiBold))
        title = self.gesture_name.replace("_", " ").title() if self.gesture_name else (
            "Hand not detected" if not self.detected else "Reading hand pose"
        )
        painter.drawText(QRectF(12, area.height() - 55, area.width() - 24, 20), Qt.AlignmentFlag.AlignCenter, title)
        painter.setPen(QColor(_COLORS["muted"]))
        painter.setFont(QFont("Inter", 8))
        if self.finger_state is not None:
            details = "  ".join(
                f"{name} {'●' if enabled else '○'}"
                for name, enabled in zip(
                    ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"),
                    (
                        self.finger_state.thumb,
                        self.finger_state.index,
                        self.finger_state.middle,
                        self.finger_state.ring,
                        self.finger_state.pinky,
                    ),
                )
            )
        else:
            details = "WAITING FOR TRACKER" if not self.detected else "POSE UNAVAILABLE"
        painter.drawText(
            QRectF(8, area.height() - 30, area.width() - 16, 16),
            Qt.AlignmentFlag.AlignCenter,
            details,
        )

    def _animate_pose(self) -> None:
        current = self._current_landmarks
        target = self._target_landmarks
        now = time.monotonic()
        dt = max(1e-4, now - self._last_animation_time)
        self._last_animation_time = now
        alpha = 1.0 - math.exp(-dt / 0.045)
        updated = tuple(
            (
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha,
            )
            for start, end in zip(current, target)
        )
        self._current_landmarks = updated
        if max(
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(updated, target)
        ) < 0.001:
            self._current_landmarks = target
            self._animation.stop()
        self.update()

    @staticmethod
    def _make_pose(state: FingerState) -> tuple[tuple[float, float], ...]:
        points = [(0.5, 0.91)] * 21
        thumb = (
            ((0.44, 0.72), (0.35, 0.64), (0.29, 0.57), (0.21, 0.50))
            if state.thumb
            else ((0.44, 0.72), (0.40, 0.66), (0.43, 0.61), (0.47, 0.62))
        )
        points[1:5] = thumb
        fingers = (
            (5, 0.37, 0.55, 0.35),
            (9, 0.48, 0.52, 0.32),
            (13, 0.59, 0.54, 0.35),
            (17, 0.69, 0.59, 0.41),
        )
        extended = (state.index, state.middle, state.ring, state.pinky)
        for (base, x, root_y, tip_y), is_extended in zip(fingers, extended):
            points[base] = (x, root_y)
            if is_extended:
                points[base + 1] = (x - 0.005, root_y - 0.18)
                points[base + 2] = (x - 0.012, tip_y + 0.07)
                points[base + 3] = (x - 0.018, tip_y)
            else:
                points[base + 1] = (x + 0.02, root_y + 0.12)
                points[base + 2] = (x + 0.005, root_y + 0.19)
                points[base + 3] = (x - 0.04, root_y + 0.17)
        return tuple(points)

    @staticmethod
    def _fit_landmarks(
        landmarks: tuple[tuple[float, float], ...],
    ) -> tuple[QPointF, ...]:
        min_x = min(point[0] for point in landmarks)
        max_x = max(point[0] for point in landmarks)
        min_y = min(point[1] for point in landmarks)
        max_y = max(point[1] for point in landmarks)
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        scale = min(300 / width, 222 / height)
        offset_x = (340 - width * scale) / 2
        offset_y = (250 - height * scale) / 2
        return tuple(
            QPointF(offset_x + (x - min_x) * scale, offset_y + (y - min_y) * scale)
            for x, y in landmarks
        )

    def _draw_hand(
        self,
        painter: QPainter,
        points: tuple[QPointF, ...],
        opacity: float,
    ) -> None:
        painter.save()
        painter.setOpacity(opacity)
        outline_color = QColor("#8E4D3D")
        highlight_color = QColor("#FFD0A7")
        fingers = (
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        )
        palm_width = max(38.0, points[17].x() - points[5].x())

        palm = self._palm_path(points, palm_width)
        painter.setPen(Qt.PenStyle.NoPen)
        shadow_path = QPainterPath(palm)
        shadow_path.translate(0, 5)
        painter.setBrush(QColor(0, 0, 0, 70))
        painter.drawPath(shadow_path)

        for finger_index, chain in enumerate(fingers):
            width = palm_width * (0.205 if finger_index == 0 else 0.18 - finger_index * 0.008)
            self._draw_finger(painter, [points[index] for index in chain], width, finger_index)

        shadow = QPainterPath(palm)
        shadow.translate(0, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 35))
        painter.drawPath(shadow)
        skin = QLinearGradient(85, 65, 260, 220)
        skin.setColorAt(0, QColor("#FFD0A7"))
        skin.setColorAt(0.38, QColor("#E8A77D"))
        skin.setColorAt(1, QColor("#BA6D51"))
        painter.setBrush(skin)
        painter.setPen(QPen(outline_color, 1.8))
        painter.drawPath(palm)

        self._draw_palm_creases(painter, points, palm_width, highlight_color, outline_color)
        for finger_index, chain in enumerate(fingers):
            dip, tip = points[chain[-2]], points[chain[-1]]
            nail_width = palm_width * (0.12 - finger_index * 0.006)
            if math.hypot(tip.x() - dip.x(), tip.y() - dip.y()) > nail_width * 0.9:
                self._draw_nail(painter, dip, tip, nail_width)
        painter.restore()

    @staticmethod
    def _smooth_closed_path(points: list[QPointF]) -> QPainterPath:
        path = QPainterPath()
        if len(points) < 3:
            return path
        midpoint = lambda a, b: QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
        path.moveTo(midpoint(points[-1], points[0]))
        for index, point in enumerate(points):
            path.quadTo(point, midpoint(point, points[(index + 1) % len(points)]))
        path.closeSubpath()
        return path

    @staticmethod
    def _palm_path(points: tuple[QPointF, ...], palm_width: float) -> QPainterPath:
        across = points[17] - points[5]
        length = max(math.hypot(across.x(), across.y()), 1e-6)
        direction = across / length
        wrist_left = points[0] - direction * palm_width * 0.34
        wrist_right = points[0] + direction * palm_width * 0.34
        contour = [
            wrist_left,
            points[1],
            points[2],
            points[5],
            points[9],
            points[13],
            points[17],
            wrist_right,
            points[0],
        ]
        return HandPoseWidget._smooth_closed_path(contour)

    @staticmethod
    def _draw_finger(
        painter: QPainter,
        points: list[QPointF],
        base_width: float,
        finger_index: int,
    ) -> None:
        outline = QColor("#96523F")
        shades = ("#F6C39A", "#EAA77D", "#D8906D", "#BD7457")
        for segment in range(3):
            start, end = points[segment], points[segment + 1]
            width = base_width * (1.0 - segment * 0.19)
            painter.setPen(QPen(outline, width + 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start, end)
            gradient = QLinearGradient(
                start + QPointF(-width * 0.35, -width * 0.35),
                end + QPointF(width * 0.3, width * 0.4),
            )
            gradient.setColorAt(0, QColor(shades[0]))
            gradient.setColorAt(0.48, QColor(shades[min(segment + 1, 3)]))
            gradient.setColorAt(1, QColor("#AC634B"))
            painter.setPen(QPen(gradient, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start, end)

            if segment < 2:
                joint = end
                radius = width * 0.39
                painter.setPen(QPen(outline, 1.0))
                painter.setBrush(QColor(shades[segment + 1]))
                painter.drawEllipse(joint, radius, radius)
                crease = QPainterPath()
                crease.moveTo(joint + QPointF(-width * 0.19, width * 0.08))
                crease.quadTo(
                    joint + QPointF(0, width * 0.18),
                    joint + QPointF(width * 0.19, width * 0.08),
                )
                painter.setPen(QPen(QColor(115, 55, 45, 125), max(0.8, width * 0.045)))
                painter.drawPath(crease)

            direction = end - start
            length = max(math.hypot(direction.x(), direction.y()), 1e-6)
            normal = QPointF(-direction.y() / length, direction.x() / length)
            highlight = QPainterPath()
            highlight.moveTo(start + normal * width * 0.22)
            highlight.lineTo(end + normal * width * 0.22)
            painter.setPen(QPen(QColor(255, 231, 202, 95), max(0.8, width * 0.07), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(highlight)

    @staticmethod
    def _draw_nail(
        painter: QPainter,
        dip: QPointF,
        tip: QPointF,
        width: float,
    ) -> None:
        direction = tip - dip
        angle = math.degrees(math.atan2(direction.y(), direction.x()))
        center = dip + direction * 0.72
        painter.save()
        painter.translate(center)
        painter.rotate(angle + 90)
        nail = QRectF(-width * 0.47, -width * 0.42, width * 0.94, width * 0.78)
        gradient = QLinearGradient(nail.topLeft(), nail.bottomRight())
        gradient.setColorAt(0, QColor("#F8D9C8"))
        gradient.setColorAt(1, QColor("#D89A80"))
        painter.setPen(QPen(QColor("#B77562"), 0.8))
        painter.setBrush(gradient)
        painter.drawRoundedRect(nail, width * 0.35, width * 0.35)
        painter.setPen(QPen(QColor(255, 244, 230, 170), max(0.6, width * 0.045)))
        painter.drawLine(QPointF(-width * 0.25, -width * 0.23), QPointF(width * 0.23, -width * 0.23))
        painter.restore()

    @staticmethod
    def _draw_palm_creases(
        painter: QPainter,
        points: tuple[QPointF, ...],
        palm_width: float,
        highlight: QColor,
        shadow: QColor,
    ) -> None:
        towards_wrist = points[0] - points[9]
        crease_start = points[5] + (points[0] - points[5]) * 0.32
        crease_control = points[9] + towards_wrist * 0.3
        crease_end = points[17] + (points[0] - points[17]) * 0.32
        thumb_crease_start = points[2] + (points[0] - points[2]) * 0.3
        thumb_crease_control = points[9] + (points[0] - points[9]) * 0.52
        thumb_crease_end = points[17] + (points[0] - points[17]) * 0.5
        creases = (
            (crease_start, crease_control, crease_end),
            (thumb_crease_start, thumb_crease_control, thumb_crease_end),
        )
        for index, (start, control, end) in enumerate(creases):
            path = QPainterPath(start)
            path.quadTo(control, end)
            painter.setPen(QPen(QColor(shadow.red(), shadow.green(), shadow.blue(), 105), max(1.0, palm_width * 0.018), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)
            if index == 0:
                path.translate(0, -1.6)
                painter.setPen(QPen(QColor(highlight.red(), highlight.green(), highlight.blue(), 115), max(0.8, palm_width * 0.012), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawPath(path)


class CameraPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cameraPanel")
        self._pixmap: Optional[QPixmap] = None
        self.image = QLabel("CAMERA INITIALIZING")
        self.image.setObjectName("cameraImage")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(300, 190)
        self.image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.status = QLabel("●  STARTING")
        self.status.setObjectName("cameraStatus")
        self.rate = QLabel("-- FPS")
        self.rate.setObjectName("mutedLabel")
        header = QHBoxLayout()
        header.addWidget(_eyebrow("LIVE HAND TRACKER"))
        header.addStretch(1)
        header.addWidget(self.rate)
        header.addSpacing(14)
        header.addWidget(self.status)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self.image, 1)

    def set_status(self, text: str, online: bool = False, error: bool = False) -> None:
        self.status.setText(f"●  {text.upper()}")
        self.status.setStyleSheet(
            f"color: {_COLORS['red'] if error else _COLORS['green'] if online else _COLORS['orange']};"
        )

    def set_frame(self, image: QImage, fps: float) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self.rate.setText(f"{fps:.0f} FPS")
        self._show_pixmap()

    def set_disabled(self, text: str) -> None:
        self._pixmap = None
        self.image.setPixmap(QPixmap())
        self.image.setText(text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show_pixmap()

    def _show_pixmap(self) -> None:
        if self._pixmap is not None and not self.image.size().isEmpty():
            self.image.setPixmap(
                self._pixmap.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


def _eyebrow(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("eyebrow")
    return label


class ProjectCard(QGraphicsObject):
    activated = pyqtSignal(int, bool)

    def __init__(self, index: int, node: ProjectNode, child_counts: dict[str, int], parent=None):
        super().__init__(parent)
        self.index = index
        self.node = node
        self.child_counts = child_counts
        self.selected = False
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, 258, 126)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.boundingRect().adjusted(1, 1, -1, -1)
        accent = _kind_color(self.node.kind)
        background = QColor("#1A2940" if self.selected else "#131E2F" if self.hovered else "#101827")
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, background.lighter(112 if self.selected else 100))
        gradient.setColorAt(1, QColor("#111A29"))
        painter.setBrush(gradient)
        painter.setPen(QPen(accent if self.selected else QColor(_COLORS["border"]), 1.8 if self.selected else 1))
        painter.drawRoundedRect(rect, 14, 14)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(1, 17, 4, 40), 2, 2)
        painter.setPen(accent)
        painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        painter.drawText(QRectF(17, 14, 130, 16), self.node.kind.upper())
        painter.setPen(QColor(_COLORS["text"]))
        painter.setFont(QFont("Inter", 11, QFont.Weight.DemiBold))
        title = painter.fontMetrics().elidedText(
            self.node.name,
            Qt.TextElideMode.ElideRight,
            222,
        )
        painter.drawText(QRectF(17, 39, 224, 22), title)
        painter.setPen(QColor(_COLORS["muted"]))
        painter.setFont(QFont("Inter", 8))
        path = self.node.path or self.node.name
        painter.drawText(
            QRectF(17, 66, 224, 17),
            painter.fontMetrics().elidedText(path, Qt.TextElideMode.ElideMiddle, 224),
        )
        child_summary = "  ·  ".join(
            f"{count} {kind}{'s' if count != 1 else ''}"
            for kind, count in self.child_counts.items()
            if count
        ) or "LEAF NODE"
        has_children = any(self.child_counts.values())
        painter.setPen(QColor(_COLORS["cyan"] if has_children else _COLORS["muted"]))
        painter.setFont(QFont("Inter", 8, QFont.Weight.DemiBold))
        painter.drawText(QRectF(17, 98, 224, 15), child_summary.upper())

    def hoverEnterEvent(self, event) -> None:
        self.hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.index, False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit(self.index, True)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ProjectMapView(QGraphicsView):
    item_activated = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("projectMapView")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(_COLORS["background"]))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.controller = ExplorerController(None)
        self.graph: Optional[ProjectGraph] = None
        self._rebuild_pending = False

    def set_project(self, graph: Optional[ProjectGraph], controller: ExplorerController) -> None:
        self.graph = graph
        self.controller = controller
        self.rebuild()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._rebuild_pending:
            self._rebuild_pending = True
            QTimer.singleShot(0, self._rebuild_after_resize)

    def _rebuild_after_resize(self) -> None:
        self._rebuild_pending = False
        self.rebuild()

    def rebuild(self) -> None:
        self.scene.clear()
        width = max(540, self.viewport().width())
        height = max(270, self.viewport().height())
        nodes = self.controller.nodes
        columns = max(1, (width - 40) // 278)
        rows = max(1, (height - 34) // 144)
        self.controller.page_size = max(1, columns * rows)
        page_count = max(1, math.ceil(len(nodes) / self.controller.page_size))
        if nodes:
            self.controller.page = min(self.controller.selected_index // self.controller.page_size, page_count - 1)
        else:
            self.controller.page = 0
        first = self.controller.page * self.controller.page_size
        visible = nodes[first:first + self.controller.page_size]
        positions: dict[str, QPointF] = {}
        for local_index, node in enumerate(visible):
            row, column = divmod(local_index, columns)
            positions[node.node_id] = QPointF(24 + column * 278, 20 + row * 144)

        if self.graph is not None:
            for edge in self.graph.edges:
                if edge.source_id not in positions or edge.target_id not in positions:
                    continue
                start = positions[edge.source_id] + QPointF(258, 63)
                end = positions[edge.target_id] + QPointF(0, 63)
                path = QPainterPath(start)
                path.cubicTo(
                    QPointF(start.x() + 44, start.y()),
                    QPointF(end.x() - 44, end.y()),
                    end,
                )
                color = QColor(_COLORS["violet"] if edge.kind == "calls" else _COLORS["blue"])
                color.setAlpha(125)
                item = QGraphicsPathItem(path)
                item.setPen(QPen(color, 1.6, Qt.PenStyle.DashLine))
                item.setZValue(0)
                self.scene.addItem(item)

        for local_index, node in enumerate(visible):
            global_index = first + local_index
            child_counts: dict[str, int] = {}
            if self.controller.graph is not None:
                child_counts = {
                    kind: len(
                        tuple(child for child in self.controller.children(node) if child.kind == kind)
                    )
                    for kind in ("package", "file", "class", "function")
                }
            card = ProjectCard(global_index, node, child_counts)
            card.setPos(positions[node.node_id])
            card.selected = global_index == self.controller.selected_index
            card.activated.connect(self.item_activated)
            card.setZValue(1)
            self.scene.addItem(card)

        scene_width = max(width, columns * 278 + 36)
        scene_height = max(height, math.ceil(len(visible) / columns) * 144 + 34, 270)
        self.scene.setSceneRect(0, 0, scene_width, scene_height)


class MainWindow(QMainWindow):
    def __init__(
        self,
        project_path: Optional[str | Path] = None,
        camera_index: int = 0,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        demo: bool = False,
        camera_view: bool = True,
        prompt_for_project: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("VisiGUI · VisiGUI")
        self.setMinimumSize(1024, 700)
        self._camera_index = camera_index
        self._model_path = str(model_path)
        self._demo = demo
        self._camera_view = camera_view
        self._controller = ExplorerController(None)
        self._graph: Optional[ProjectGraph] = None
        self._camera_worker: Optional[CameraWorker] = None
        self._scan_worker: Optional[ProjectScanWorker] = None
        self._closing = False
        self._camera_error = False
        self._project_error: Optional[str] = None
        self._build_ui()
        self._apply_theme()
        self._connect_ui()
        if not camera_view:
            self.camera_panel.set_disabled("CAMERA TRACKING ACTIVE · PREVIEW HIDDEN")
        if demo:
            self.camera_panel.set_status("Keyboard demo", online=True)
            self.camera_panel.set_disabled("PRESS 0–5 TO SIMULATE HAND POSES")
        elif camera_view:
            self._start_camera()
        else:
            self.camera_panel.set_status("Tracker active · preview hidden", online=True)
            self._start_camera()
        if project_path is not None:
            self.load_project(project_path)
        elif prompt_for_project:
            QTimer.singleShot(180, self.open_project_dialog)
        self._refresh_map()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 18, 28, 16)
        outer.setSpacing(16)
        header = self._build_header()
        outer.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        self._splitter = splitter
        upper = QWidget()
        upper_layout = QHBoxLayout(upper)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(14)
        self.hand_panel = self._build_hand_panel()
        self.camera_panel = CameraPanel()
        upper_layout.addWidget(self.hand_panel, 8)
        upper_layout.addWidget(self.camera_panel, 12)
        splitter.addWidget(upper)

        lower = self._build_project_panel()
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        outer.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.footer_status = QLabel("SYSTEM READY")
        self.footer_status.setObjectName("mutedLabel")
        self.footer_status.setFont(QFont("Inter", 8, QFont.Weight.DemiBold))
        footer.addWidget(self.footer_status)
        footer.addStretch(1)
        footer.addWidget(_eyebrow("VisiGUI  /  VISION SYSTEMS"))
        outer.addLayout(footer)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(14)
        layout.addWidget(BrandMark())
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("VisiGUI")
        title.setObjectName("brandTitle")
        subtitle = QLabel("VISION-DRIVEN CODE EXPLORATION")
        subtitle.setObjectName("brandSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        layout.addLayout(titles)
        layout.addSpacing(24)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setObjectName("headerDivider")
        layout.addWidget(divider)
        product = QVBoxLayout()
        product.setSpacing(3)
        app_name = QLabel("VISIGUI")
        app_name.setObjectName("productTag")
        product_desc = QLabel("PYTHON PROJECT VISION")
        product_desc.setObjectName("mutedLabel")
        product.addWidget(app_name)
        product.addWidget(product_desc)
        layout.addLayout(product)
        layout.addStretch(1)
        developer = QLabel("dev : ManiKamran")
        developer.setObjectName("developerCredit")
        developer.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(developer)
        return header

    def _build_hand_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("surfacePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(6)
        heading = QHBoxLayout()
        heading.addWidget(_eyebrow("LIVE POSE / HAND MODEL"))
        heading.addStretch(1)
        self.hand_state = QLabel("WAITING")
        self.hand_state.setObjectName("statePill")
        heading.addWidget(self.hand_state)
        layout.addLayout(heading)
        self.hand_graphic = HandPoseWidget()
        layout.addWidget(self.hand_graphic, 1)
        return panel

    def _build_project_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("surfacePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        heading.addWidget(_eyebrow("PROJECT VISION / INTERACTIVE MAP"))
        heading.addStretch(1)
        self.project_stats = QLabel("OPEN A PYTHON PROJECT TO BEGIN")
        self.project_stats.setObjectName("mutedLabel")
        heading.addWidget(self.project_stats)
        self.open_button = QPushButton("＋  OPEN PROJECT")
        self.open_button.setObjectName("primaryButton")
        heading.addWidget(self.open_button)
        layout.addLayout(heading)

        controls = QHBoxLayout()
        self.breadcrumb = QLabel("NO PROJECT")
        self.breadcrumb.setObjectName("breadcrumb")
        controls.addWidget(self.breadcrumb, 1)
        self.back_button = QPushButton("←  BACK")
        self.next_button = QPushButton("NEXT  →")
        self.enter_button = QPushButton("ENTER  ↵")
        self.functions_button = QPushButton("ƒ  FUNCTIONS")
        self.page_button = QPushButton("NEXT PAGE  ›")
        for button in (
            self.back_button,
            self.next_button,
            self.enter_button,
            self.functions_button,
            self.page_button,
        ):
            button.setObjectName("navButton")
            controls.addWidget(button)
        layout.addLayout(controls)
        self.map_view = ProjectMapView()
        layout.addWidget(self.map_view, 1)
        return panel

    def _connect_ui(self) -> None:
        self.open_button.clicked.connect(self.open_project_dialog)
        self.back_button.clicked.connect(lambda: self._apply_action(ProjectAction.BACK))
        self.next_button.clicked.connect(lambda: self._apply_action(ProjectAction.NEXT))
        self.enter_button.clicked.connect(lambda: self._apply_action(ProjectAction.OPEN))
        self.functions_button.clicked.connect(lambda: self._apply_action(ProjectAction.SHOW_FUNCTIONS))
        self.page_button.clicked.connect(lambda: self._apply_action(ProjectAction.NEXT_PAGE))
        self.map_view.item_activated.connect(self._on_card_activated)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#root {{ background: {_COLORS['background']}; color: {_COLORS['text']}; }}
            QMainWindow {{ background: {_COLORS['background']}; }}
            QFrame#surfacePanel, QFrame#cameraPanel {{
                background: {_COLORS['surface']};
                border: 1px solid {_COLORS['border']};
                border-radius: 18px;
            }}
            QLabel {{ color: {_COLORS['text']}; background: transparent; }}
            QLabel#brandTitle {{ color: {_COLORS['text']}; font-size: 24px; font-weight: 800; letter-spacing: 3px; }}
            QLabel#brandSubtitle {{ color: {_COLORS['cyan']}; font-size: 9px; font-weight: 700; letter-spacing: 2px; }}
            QLabel#productTag {{ color: {_COLORS['violet']}; font-size: 11px; font-weight: 800; letter-spacing: 2px; }}
            QLabel#developerCredit {{ color: {_COLORS['muted']}; font-size: 11px; padding-right: 8px; }}
            QLabel#eyebrow {{ color: {_COLORS['muted']}; font-size: 9px; font-weight: 800; letter-spacing: 1.5px; }}
            QLabel#mutedLabel {{ color: {_COLORS['muted']}; font-size: 9px; }}
            QLabel#cameraStatus {{ color: {_COLORS['orange']}; font-size: 9px; font-weight: 800; letter-spacing: 1px; }}
            QLabel#cameraImage {{ color: {_COLORS['muted']}; background: #080D17; border-radius: 12px; font-size: 11px; letter-spacing: 2px; }}
            QLabel#statePill {{ color: {_COLORS['cyan']}; background: #142B32; border: 1px solid #23545B; border-radius: 9px; padding: 5px 9px; font-size: 8px; font-weight: 800; }}
            QLabel#breadcrumb {{ color: {_COLORS['text']}; font-size: 10px; font-weight: 700; }}
            QPushButton {{ color: {_COLORS['text']}; background: {_COLORS['surface_alt']}; border: 1px solid {_COLORS['border']}; border-radius: 9px; padding: 8px 12px; font-size: 9px; font-weight: 800; }}
            QPushButton:hover {{ color: {_COLORS['cyan']}; border-color: {_COLORS['cyan']}; background: #142536; }}
            QPushButton#primaryButton {{ color: #061416; background: {_COLORS['cyan']}; border: none; padding: 9px 14px; }}
            QPushButton#primaryButton:hover {{ background: #73F3E3; color: #061416; }}
            QSplitter::handle {{ background: transparent; height: 8px; }}
            QScrollBar:vertical {{ background: #0B111C; width: 10px; margin: 2px; }}
            QScrollBar::handle:vertical {{ background: #34425B; border-radius: 5px; min-height: 24px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{ background: #0B111C; height: 10px; margin: 2px; }}
            QScrollBar::handle:horizontal {{ background: #34425B; border-radius: 5px; min-width: 24px; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
            """
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_fullscreen_requested", False):
            self._fullscreen_requested = True
            self.showFullScreen()
            QTimer.singleShot(
                0,
                lambda: self._splitter.setSizes(
                    [int(self.height() * 0.42), int(self.height() * 0.58)]
                ),
            )

    def _start_camera(self) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            return
        self.camera_panel.set_status("Starting")
        worker = CameraWorker(self._camera_index, self._model_path, self)
        worker.frame_ready.connect(self._on_camera_frame)
        worker.action_ready.connect(self._on_gesture_action)
        worker.status_changed.connect(self._on_camera_status)
        worker.error_occurred.connect(self._on_camera_error)
        worker.finished.connect(self._on_worker_finished)
        self._camera_worker = worker
        worker.start()

    def _on_camera_frame(
        self,
        image: QImage,
        finger_state: Optional[FingerState],
        gesture_name: str,
        handedness: str,
        detected: bool,
        fps: float,
        landmark_positions: Optional[tuple[tuple[float, float], ...]] = None,
    ) -> None:
        if self._camera_view:
            self.camera_panel.set_frame(image, fps)
        self.hand_graphic.set_pose(
            finger_state,
            gesture_name,
            detected,
            landmark_positions,
        )
        self.hand_state.setText("TRACKING" if detected else "SEARCHING")
        self.hand_state.setStyleSheet(
            f"color: {_COLORS['green'] if detected else _COLORS['orange']};"
        )
        if detected:
            self.footer_status.setText(
                f"TRACKING {handedness.upper() or 'HAND'}  ·  {fps:.0f} FPS  ·  "
                f"{gesture_name.replace('_', ' ')}"
            )
        else:
            self.footer_status.setText("SEARCHING FOR HAND  ·  KEEP YOUR HAND INSIDE THE CAMERA FRAME")

    def _on_gesture_action(self, gesture_name: str) -> None:
        self._controller.choose_gesture(gesture_name)
        self._refresh_map()

    def _on_camera_status(self, text: str) -> None:
        if text == "Camera connected":
            self._camera_error = False
            self.camera_panel.set_status(text, online=True)
        elif text == "Camera stopped" and not self._closing and not self._camera_error:
            self.camera_panel.set_status(text)

    def _on_camera_error(self, text: str) -> None:
        self._camera_error = True
        self.camera_panel.set_status("Camera error", error=True)
        self.camera_panel.set_disabled(text)

    def _apply_action(self, action: ProjectAction) -> None:
        self._controller.apply(action)
        self._refresh_map()

    def _on_card_activated(self, index: int, open_item: bool) -> None:
        self._controller.select(index)
        if open_item:
            self._controller.confirm()
        self._refresh_map()

    def _refresh_map(self) -> None:
        self.map_view.set_project(self._graph, self._controller)
        if self._graph is None:
            self.breadcrumb.setText("NO PROJECT  /  OPEN A FOLDER TO EXPLORE")
            if self._project_error is None:
                self.project_stats.setText("STATIC PYTHON SOURCE GRAPH")
        else:
            crumbs = ["PROJECT"] + [node.name for node in self._controller.breadcrumb]
            self.breadcrumb.setText("  ›  ".join(crumbs))
            if self._project_error is None:
                summary = (
                    f"{len(self._graph.files)} FILES  ·  "
                    f"{sum(node.kind in ('class', 'function') for node in self._graph.nodes)} SYMBOLS  ·  "
                    f"PAGE {self._controller.page + 1}"
                )
                if self._graph.parse_errors:
                    summary += f"  ·  {len(self._graph.parse_errors)} SOURCE WARNING(S)"
                    self.project_stats.setStyleSheet(f"color: {_COLORS['orange']};")
                self.project_stats.setText(summary)
        self.functions_button.setText(
            "ƒ  FUNCTIONS ON" if self._controller.include_functions else "ƒ  FUNCTIONS OFF"
        )
        selected = self._controller.selected_node
        self.footer_status.setText(
            f"{self._controller.last_action.upper()}"
            + (f"  ·  SELECTED {selected.name}" if selected else "")
        )

    def open_project_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Python project",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.load_project(directory)

    def load_project(self, project_path: str | Path) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        worker = ProjectScanWorker(project_path, self)
        worker.project_ready.connect(self._on_project_loaded)
        worker.error_occurred.connect(self._on_project_error)
        worker.finished.connect(self._on_worker_finished)
        self._scan_worker = worker
        self._project_error = None
        self.open_button.setEnabled(False)
        self.project_stats.setText(f"SCANNING {Path(project_path).expanduser()}")
        self.project_stats.setStyleSheet(f"color: {_COLORS['blue']};")
        self.footer_status.setStyleSheet("")
        worker.start()

    def _on_project_loaded(self, graph: ProjectGraph) -> None:
        self._graph = graph
        self._project_error = None
        self.project_stats.setStyleSheet("")
        self._controller = ExplorerController(graph)
        self._refresh_map()

    def _on_project_error(self, text: str) -> None:
        self._project_error = text
        self.project_stats.setText(f"PROJECT SCAN ERROR  ·  {text}")
        self.project_stats.setStyleSheet(f"color: {_COLORS['red']};")
        self.footer_status.setText("PROJECT SCAN FAILED")
        self.open_button.setEnabled(True)

    def _on_worker_finished(self) -> None:
        self.open_button.setEnabled(True)
        if self._closing and not self._has_running_workers():
            QTimer.singleShot(0, self.close)

    def _has_running_workers(self) -> bool:
        return any(
            worker is not None and worker.isRunning()
            for worker in (self._camera_worker, self._scan_worker)
        )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._controller.breadcrumb:
                self._apply_action(ProjectAction.BACK)
            else:
                self.close()
            return
        if key in _DEMO_GESTURES:
            gesture, state = _DEMO_GESTURES[key]
            if self._demo:
                self.hand_graphic.set_pose(state, gesture.name, True)
                self.hand_state.setText("DEMO POSE")
            self._controller.choose_gesture(gesture.name)
            self._refresh_map()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_action(ProjectAction.OPEN)
            return
        if key == Qt.Key.Key_Backspace:
            self._apply_action(ProjectAction.BACK)
            return
        if key == Qt.Key.Key_Left:
            self._apply_action(ProjectAction.PREVIOUS)
            return
        if key == Qt.Key.Key_Right:
            self._apply_action(ProjectAction.NEXT)
            return
        if key == Qt.Key.Key_PageDown:
            self._apply_action(ProjectAction.NEXT_PAGE)
            return
        if key == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self._apply_action(ProjectAction.SHOW_FUNCTIONS)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        running = [worker for worker in (self._camera_worker, self._scan_worker) if worker and worker.isRunning()]
        if running:
            self._closing = True
            for worker in running:
                if isinstance(worker, CameraWorker):
                    worker.stop()
            self.footer_status.setText("CLOSING WORKERS…")
            event.ignore()
            return
        event.accept()


def launch_gui(
    project_path: Optional[str | Path] = None,
    camera_index: int = 0,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    demo: bool = False,
    camera_view: bool = True,
) -> int:
    with StderrLogCapture():
        app = QApplication.instance() or QApplication([])
        app.setApplicationName("VisiGUI")
        app.setOrganizationName("ManiKamran")
        window = MainWindow(
            project_path=project_path,
            camera_index=camera_index,
            model_path=model_path,
            demo=demo,
            camera_view=camera_view,
            prompt_for_project=project_path is None,
        )
        window.showFullScreen()
        return app.exec()
