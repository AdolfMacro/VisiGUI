from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from ..core.gesture import Gesture, GestureEvent
from ..core.intent import Intent
from ..vision.tracking import HandState, TwoHandState
from .controller import InteractionController


DEFAULT_MOTION_SPEED = 50.0
DEFAULT_ZOOM_SPEED = DEFAULT_MOTION_SPEED


@dataclass(frozen=True)
class WorldMotion:
    strafe: float = 0.0
    vertical: float = 0.0
    depth: float = 0.0


class TwoHandWorldController:
    """Convert stable logical hand states into analog camera velocity targets."""

    def __init__(
        self,
        navigation_dead_zone: float = 0.08,
        navigation_sensitivity: float = 2.2,
        zoom_sensitivity: float = 1.0,
        max_navigation_speed: float = DEFAULT_MOTION_SPEED,
        max_zoom_speed: float = DEFAULT_ZOOM_SPEED,
        pinch_activation_seconds: float = 0.12,
        pinch_activation_distance: float = 0.12,
        pinch_contact_exit_distance: float = 0.2,
        zoom_filter_time_constant: float = 0.035,
    ) -> None:
        if not 0 <= navigation_dead_zone < 1:
            raise ValueError("Navigation dead zone must be between 0 and 1")
        if min(navigation_sensitivity, zoom_sensitivity, max_navigation_speed, max_zoom_speed) <= 0:
            raise ValueError("Hand control sensitivities and speeds must be positive")
        if pinch_activation_seconds < 0:
            raise ValueError("Pinch activation time cannot be negative")
        if not 0 < pinch_activation_distance < pinch_contact_exit_distance:
            raise ValueError("Pinch contact exit must exceed its positive entry threshold")
        if zoom_filter_time_constant <= 0:
            raise ValueError("Zoom filter time constant must be positive")
        self.navigation_dead_zone = navigation_dead_zone
        self.navigation_sensitivity = navigation_sensitivity
        self.zoom_sensitivity = zoom_sensitivity
        self.max_navigation_speed = max_navigation_speed
        self.max_zoom_speed = max_zoom_speed
        self.pinch_activation_seconds = pinch_activation_seconds
        self.pinch_activation_distance = pinch_activation_distance
        self.pinch_contact_exit_distance = pinch_contact_exit_distance
        self.zoom_filter_time_constant = zoom_filter_time_constant
        self._left_reference: Optional[tuple[float, float]] = None
        self._filtered_pinch_distance: Optional[float] = None
        self._zoom_candidate: Optional[Gesture] = None
        self._zoom_candidate_duration = 0.0
        self._zoom_mode: Optional[Gesture] = None
        self._interaction = InteractionController()
        self._interaction.set_context("WORLD")

    def update(self, hands: TwoHandState, delta_time: float) -> WorldMotion:
        dt = min(max(delta_time, 0.0), 0.1)
        left = hands.left_hand
        if left is None or left.gesture is not Gesture.FIST:
            self._left_reference = None
            strafe = vertical = 0.0
        else:
            if self._left_reference is None:
                self._left_reference = left.normalized_position[:2]
                strafe = vertical = 0.0
            else:
                dx = left.normalized_position[0] - self._left_reference[0]
                dy = left.normalized_position[1] - self._left_reference[1]
                strafe = self._axis_velocity(
                    dx,
                    self.navigation_dead_zone,
                    self.navigation_sensitivity,
                )
                vertical = self._axis_velocity(
                    -dy,
                    self.navigation_dead_zone,
                    self.navigation_sensitivity,
                )

        right = hands.right_hand
        if right is None:
            self._reset_zoom()
            return WorldMotion(strafe, vertical, 0.0)

        if self._filtered_pinch_distance is None:
            self._filtered_pinch_distance = right.pinch_distance
        else:
            alpha = 1.0 - math.exp(-dt / self.zoom_filter_time_constant)
            self._filtered_pinch_distance += alpha * (
                right.pinch_distance - self._filtered_pinch_distance
            )
        gap = self._filtered_pinch_distance

        if self._zoom_mode is Gesture.PINCH:
            candidate = (
                Gesture.OPEN_PALM
                if gap > self.pinch_contact_exit_distance
                else Gesture.PINCH
            )
        elif self._zoom_mode is Gesture.OPEN_PALM:
            candidate = (
                Gesture.PINCH
                if gap <= self.pinch_activation_distance
                else Gesture.OPEN_PALM
            )
        else:
            candidate = (
                Gesture.PINCH
                if gap <= self.pinch_activation_distance
                else Gesture.OPEN_PALM
            )

        if candidate is None or candidate is self._zoom_mode:
            self._zoom_candidate = None
            self._zoom_candidate_duration = 0.0
        else:
            if candidate is not self._zoom_candidate:
                self._zoom_candidate = candidate
                self._zoom_candidate_duration = 0.0
            self._zoom_candidate_duration += dt
            if self._zoom_candidate_duration >= self.pinch_activation_seconds:
                self._zoom_mode = candidate
                self._zoom_candidate = None
                self._zoom_candidate_duration = 0.0

        if self._zoom_mode is None:
            return WorldMotion(strafe, vertical, 0.0)

        gesture_event = GestureEvent(
            gesture=self._zoom_mode,
            previous_gesture=right.gesture,
            timestamp=hands.timestamp,
            confidence=right.confidence,
        )
        intent = self._interaction.handle_gesture(gesture_event).intent
        if intent is Intent.ZOOM_IN:
            depth = self.zoom_sensitivity * self.max_zoom_speed
        elif intent is Intent.ZOOM_OUT:
            depth = -self.zoom_sensitivity * self.max_zoom_speed
        else:
            depth = 0.0
        return WorldMotion(strafe, vertical, depth)

    def rebase(self, hands: TwoHandState) -> None:
        self._left_reference = (
            hands.left_hand.normalized_position[:2]
            if hands.left_hand is not None else None
        )
        self._reset_zoom()

    def _reset_zoom(self) -> None:
        self._filtered_pinch_distance = None
        self._zoom_candidate = None
        self._zoom_candidate_duration = 0.0
        self._zoom_mode = None

    def _axis_velocity(
        self,
        value: float,
        dead_zone: float,
        sensitivity: float,
        maximum: Optional[float] = None,
    ) -> float:
        magnitude = abs(value)
        if magnitude <= dead_zone:
            return 0.0
        normalized = min(1.0, (magnitude - dead_zone) / (1.0 - dead_zone))
        limit = maximum or self.max_navigation_speed
        return math.copysign(min(limit, normalized * sensitivity * limit), value)
