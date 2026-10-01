from __future__ import annotations

import math

import pytest

from visigui.core.contracts import HandObservation
from visigui.core.gesture import Gesture
from visigui.core.hand import FingerState, HandLandmark
from visigui.gui.hand_tracking import HandTrackSelector, OneEuroLandmarkFilter
from visigui.vision.tracking import DualHandTracker


def _hand(center_x: float, handedness: str, size: float = 0.2) -> HandObservation:
    points = tuple(
        (
            center_x + ((index % 5) - 2) * size / 4,
            0.5 + ((index // 5) - 2) * size / 4,
            0.0,
        )
        for index in range(21)
    )
    return HandObservation(
        landmark_positions=points,
        handedness=handedness,
        handedness_score=0.9,
    )


def _landmarks(value: float) -> tuple[tuple[float, float], ...]:
    return tuple((value, value) for _ in range(21))


def test_selector_keeps_the_same_hand_when_detector_order_changes():
    left = _hand(0.28, "Left")
    right = _hand(0.72, "Right", size=0.3)
    selector = HandTrackSelector()

    assert selector.select((left, right)) is right
    assert selector.select((left, right)) is right
    assert selector.select((right, left)) is right


def test_selector_reacquires_same_hand_after_brief_detection_drop():
    left = _hand(0.28, "Left")
    right = _hand(0.72, "Right")
    selector = HandTrackSelector()
    selector.select((right,))

    assert selector.select(()) is None
    assert selector.select((right, left)) is right


def test_selector_resets_after_prolonged_loss_and_selects_best_new_track():
    left = _hand(0.28, "Left")
    right = _hand(0.72, "Right", size=0.3)
    selector = HandTrackSelector()
    selector.select((left, right))

    for _ in range(selector.MAX_MISSED_FRAMES + 1):
        selector.select(())

    assert selector.select((left, right)) is right


def test_one_euro_filter_reduces_small_frame_to_frame_landmark_jitter():
    smoother = OneEuroLandmarkFilter()
    smoother.update(_landmarks(0.5), 1.0)

    smoothed = smoother.update(_landmarks(0.51), 1.0 + 1 / 30)

    assert smoothed[0][0] == pytest.approx(smoothed[0][1])
    assert 0.5 < smoothed[0][0] < 0.51


def test_one_euro_filter_responds_more_quickly_to_large_intentional_movement():
    slow = OneEuroLandmarkFilter()
    fast = OneEuroLandmarkFilter()
    for smoother in (slow, fast):
        smoother.update(_landmarks(0.5), 1.0)

    slow_output = slow.update(_landmarks(0.51), 1.0 + 1 / 30)
    fast_output = fast.update(_landmarks(0.9), 1.0 + 1 / 30)

    slow_fraction = (slow_output[0][0] - 0.5) / 0.01
    fast_fraction = (fast_output[0][0] - 0.5) / 0.4
    assert fast_fraction > slow_fraction


def test_one_euro_filter_rejects_invalid_landmark_frames():
    smoother = OneEuroLandmarkFilter()

    with pytest.raises(ValueError, match="21 finite 2D points"):
        smoother.update(((0.5, 0.5),), 1.0)

    with pytest.raises(ValueError, match="finite"):
        smoother.update(((float("nan"), 0.5),) * 21, 1.0)


def test_dual_hand_tracker_assigns_by_handedness_not_detector_index():
    left = _hand(0.3, "Left")
    right = _hand(0.7, "Right")
    tracker = DualHandTracker()

    first = tracker.update((right, left), 1.0)
    second = tracker.update((left, right), 1.03)

    assert first.left_hand is not None and first.left_hand.handedness == "Left"
    assert first.right_hand is not None and first.right_hand.handedness == "Right"
    assert second.left_hand is not None and second.left_hand.handedness == "Left"
    assert second.right_hand is not None and second.right_hand.handedness == "Right"
    assert second.hands_detected == 2


def test_dual_hand_tracker_handles_one_or_zero_hands_without_resetting_other_track():
    left = _hand(0.3, "Left")
    right = _hand(0.7, "Right")
    tracker = DualHandTracker()
    tracker.update((left, right), 1.0)

    one = tracker.update((right,), 1.03)
    none = tracker.update((), 1.06)

    assert one.left_hand is None and one.right_hand is not None
    assert none.hands_detected == 0


def test_hand_state_exposes_stable_logical_navigation_fields():
    observation = _hand(0.68, "Right")
    state = DualHandTracker().update((observation,), 2.0).right_hand

    assert state is not None
    assert state.handedness == "Right"
    assert len(state.landmarks) == 21
    assert len(state.palm_position) == 3
    assert len(state.normalized_position) == 3
    assert len(state.velocity) == 2
    assert isinstance(state.gesture, Gesture)
    assert isinstance(state.pinch_distance, float)
    assert isinstance(state.finger_state, FingerState)


def test_pinch_measurement_uses_world_landmarks_to_reduce_hand_rotation_error():
    landmarks = [(0.0, 0.0, 0.0) for _ in range(21)]
    landmarks[HandLandmark.THUMB_TIP] = (0.2, 0.0, 0.0)
    landmarks[HandLandmark.INDEX_TIP] = (0.0, 0.0, 0.0)
    landmarks[HandLandmark.INDEX_MCP] = (0.0, 0.0, 0.0)
    landmarks[HandLandmark.PINKY_MCP] = (0.0, 1.0, 0.0)

    def observation_after_rotation(angle: float, with_world_landmarks: bool) -> HandObservation:
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = tuple(
            (cosine * x + sine * z, y, -sine * x + cosine * z)
            for x, y, z in landmarks
        )
        image_points = tuple((0.5 + x, 0.5 + y, z) for x, y, z in rotated)
        return HandObservation(
            landmark_positions=image_points,
            handedness="Right",
            world_landmark_positions=rotated if with_world_landmarks else None,
        )

    tracker = DualHandTracker()
    flat_hand = tracker.update(
        (observation_after_rotation(0.0, True),),
        1.0,
    ).right_hand
    rotated_hand = tracker.update(
        (observation_after_rotation(1.0, True),),
        1.03,
    ).right_hand

    assert flat_hand is not None and rotated_hand is not None
    assert flat_hand.pinch_distance == pytest.approx(0.2)
    assert rotated_hand.pinch_distance == pytest.approx(0.2)

    image_only = DualHandTracker().update(
        (observation_after_rotation(1.0, False),),
        1.0,
    ).right_hand
    assert image_only is not None
    assert image_only.pinch_distance < rotated_hand.pinch_distance
