from types import SimpleNamespace
import math

import numpy as np
import pytest

import visigui.vision.detector as detector_module
from visigui.core.contracts import CameraFrame, HandObservation
from visigui.core.gesture import Gesture
from visigui.core.hand import FingerState, HandLandmark, HandLandmarks
from visigui.gesture.recognizer import GestureRecognizer
from visigui.vision.detector import FingerAnalyzer, HandDetector, resolve_model_path


def make_hand(extended=(), thumb=False, transform=None):
    positions = [(0.0, 0.0, 0.0)] * 21
    positions[HandLandmark.WRIST] = (0.0, 0.0, 0.0)
    thumb_points = (
        ((-0.35, 0.3, 0.0), (-0.55, 0.42, 0.0), (-0.75, 0.54, 0.0), (-0.95, 0.66, 0.0))
        if thumb
        else ((-0.35, 0.3, 0.0), (-0.5, 0.3, 0.0), (-0.35, 0.15, 0.0), (-0.2, 0.25, 0.0))
    )
    for landmark, point in zip(
        (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
        thumb_points,
    ):
        positions[landmark] = point

    finger_indices = (
        (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP),
        (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP),
        (HandLandmark.RING_MCP, HandLandmark.RING_PIP, HandLandmark.RING_DIP, HandLandmark.RING_TIP),
        (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    )
    for finger, (mcp, pip, dip, tip) in enumerate(finger_indices):
        x = (finger - 1.5) * 0.2
        points = (
            ((x, 0.5, 0.0), (x, 0.8, 0.0), (x, 1.1, 0.0), (x, 1.4, 0.0))
            if finger in extended
            else ((x, 0.5, 0.0), (x, 0.7, 0.0), (x, 0.48, 0.0), (x, 0.28, 0.0))
        )
        for landmark, point in zip((mcp, pip, dip, tip), points):
            positions[landmark] = point

    if transform is not None:
        positions = [transform(point) for point in positions]
    return tuple(positions)


def test_model_resolution_prefers_new_cache_and_falls_back_to_existing_legacy(
    tmp_path,
    monkeypatch,
):
    new_model = tmp_path / "visigui" / "hand_landmarker.task"
    legacy_model = tmp_path / "eyehand" / "hand_landmarker.task"
    legacy_model.parent.mkdir()
    legacy_model.write_bytes(b"existing model")
    monkeypatch.setattr(detector_module, "DEFAULT_MODEL_PATH", new_model)
    monkeypatch.setattr(detector_module, "LEGACY_MODEL_PATH", legacy_model)

    assert resolve_model_path(new_model) == legacy_model

    new_model.parent.mkdir()
    new_model.write_bytes(b"new model")
    assert resolve_model_path(new_model) == new_model
    custom_model = tmp_path / "custom.task"
    assert resolve_model_path(custom_model) == custom_model


@pytest.mark.parametrize(
    ("fingers", "thumb", "expected"),
    (
        ((), False, FingerState()),
        ((0,), False, FingerState(index=True)),
        ((0, 1), False, FingerState(index=True, middle=True)),
        ((0, 1, 2), False, FingerState(index=True, middle=True, ring=True)),
        ((0, 1, 2, 3), False, FingerState(index=True, middle=True, ring=True, pinky=True)),
        ((0, 1, 2, 3), True, FingerState(True, True, True, True, True)),
    ),
)
def test_finger_analysis_returns_five_states_and_count(fingers, thumb, expected):
    observation = HandObservation(landmark_positions=make_hand(fingers, thumb))
    state = FingerAnalyzer().analyze(observation)
    assert state == expected
    assert state.count == sum((expected.thumb, expected.index, expected.middle, expected.ring, expected.pinky))


@pytest.mark.parametrize(
    "transform",
    (
        lambda point: (point[0] + 4.0, point[1] - 3.0, point[2] + 2.0),
        lambda point: (point[0] * 2.5, point[1] * 2.5, point[2] * 2.5),
        lambda point: (-point[1], point[0], point[2]),
    ),
)
def test_finger_analysis_is_translation_scale_and_rotation_invariant(transform):
    observation = HandObservation(landmark_positions=make_hand((0, 1, 2), False, transform))
    assert FingerAnalyzer().analyze(observation) == FingerState(index=True, middle=True, ring=True)


@pytest.mark.parametrize("mirrored", (False, True))
def test_finger_analysis_is_invariant_to_palm_back_and_out_of_plane_rotation(mirrored):
    def transform(point):
        x, y, z = point
        angle_x, angle_y, angle_z = 1.1, -0.7, 0.4
        y, z = y * math.cos(angle_x) - z * math.sin(angle_x), y * math.sin(angle_x) + z * math.cos(angle_x)
        x, z = x * math.cos(angle_y) + z * math.sin(angle_y), -x * math.sin(angle_y) + z * math.cos(angle_y)
        x, y = x * math.cos(angle_z) - y * math.sin(angle_z), x * math.sin(angle_z) + y * math.cos(angle_z)
        if mirrored:
            x = -x
        return (x + 2.0, y - 3.0, z + 1.0)

    observation = HandObservation(
        landmark_positions=make_hand((0, 1, 2, 3), True, transform),
        handedness="Left" if mirrored else "Right",
    )

    assert FingerAnalyzer().analyze(observation) == FingerState(True, True, True, True, True)


def test_finger_analysis_rejects_degenerate_hand_geometry_instead_of_emitting_a_fist():
    observation = HandObservation(landmark_positions=((0.0, 0.0, 0.0),) * 21)

    assert FingerAnalyzer().analyze(observation) is None


def test_world_landmarks_are_preferred_to_image_projection():
    world_hand = make_hand((0, 1, 2, 3), True)
    observation = HandObservation(
        landmark_positions=((0.0, 0.0, 0.0),) * 21,
        image_width=640,
        image_height=480,
        world_landmark_positions=tuple(
            (x * 0.01, y * 0.01, z * 0.01)
            for x, y, z in world_hand
        ),
    )

    assert FingerAnalyzer().analyze(observation) == FingerState(True, True, True, True, True)


def test_world_landmarks_require_21_finite_points():
    with pytest.raises(ValueError, match="21 finite 3D points"):
        HandObservation(
            landmark_positions=make_hand(),
            world_landmark_positions=((0.0, 0.0, 0.0),),
        )


def test_finger_analysis_corrects_normalized_landmarks_for_image_aspect_ratio():
    width, height = 640, 480
    pixels = make_hand((0, 1, 2), False)
    normalized = tuple(
        (x / width, y / height, z / width)
        for x, y, z in pixels
    )
    observation = HandObservation(
        landmark_positions=normalized,
        image_width=width,
        image_height=height,
    )

    assert FingerAnalyzer().analyze(observation) == FingerState(
        index=True,
        middle=True,
        ring=True,
    )


def test_finger_analysis_handles_absent_and_invalid_landmarks():
    analyzer = FingerAnalyzer()
    assert analyzer.analyze(None) is None
    with pytest.raises(ValueError, match="21 landmarks"):
        analyzer.analyze(HandObservation(landmark_positions=((0.0, 0.0, 0.0),)))


def test_hand_observation_requires_both_positive_image_dimensions():
    points = make_hand()
    with pytest.raises(ValueError, match="provided together"):
        HandObservation(points, image_width=640)
    with pytest.raises(ValueError, match="positive integers"):
        HandObservation(points, image_width=640, image_height=0)


def test_gesture_recognizer_maps_supported_mvp_shapes():
    recognizer = GestureRecognizer()
    cases = (
        (FingerState(), Gesture.FIST),
        (FingerState(index=True), Gesture.INDEX),
        (FingerState(thumb=True, index=True), Gesture.THUMB_INDEX),
        (FingerState(index=True, middle=True), Gesture.INDEX_MIDDLE),
        (FingerState(index=True, middle=True, ring=True), Gesture.INDEX_MIDDLE_RING),
        (FingerState(index=True, middle=True, ring=True, pinky=True), Gesture.INDEX_MIDDLE_RING_PINKY),
        (FingerState(True, True, True, True, True), Gesture.OPEN_PALM),
    )
    for state, expected in cases:
        observation = recognizer.recognize(state, timestamp=12.5)
        assert observation.gesture_name == expected.name
        assert observation.timestamp == 12.5
        assert observation.confidence is None


def test_hand_landmarker_conversion_preserves_21_positions_and_metadata():
    points = [SimpleNamespace(x=i / 21, y=(i + 1) / 21, z=-i / 100) for i in range(21)]
    category = SimpleNamespace(category_name="Right", score=0.93)

    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            self.image = image
            self.timestamp_ms = timestamp_ms
            return SimpleNamespace(hand_landmarks=[points], handedness=[[category]])

        def close(self):
            pass

    landmarker = FakeLandmarker()
    detector = HandDetector(landmarker=landmarker)
    frame = CameraFrame(32, 24, 2.5, image=np.zeros((24, 32, 3), dtype=np.uint8), frame_id=7)
    observation = detector.detect(frame)

    assert observation is not None
    assert len(observation.landmark_positions) == 21
    assert observation.landmark_positions[20] == pytest.approx((20 / 21, 1.0, -0.2))
    assert observation.handedness == "Right"
    assert observation.handedness_score == pytest.approx(0.93)
    assert observation.confidence is None
    assert observation.timestamp == 2.5
    assert observation.frame_id == 7
    assert observation.image_width == 32
    assert observation.image_height == 24
    detector.close()


def test_hand_landmarker_conversion_preserves_world_landmarks():
    normalized = [
        SimpleNamespace(x=x, y=y, z=z)
        for x, y, z in make_hand((0, 1, 2, 3), True)
    ]
    world = [
        SimpleNamespace(x=x * 0.01, y=y * 0.01, z=z * 0.01)
        for x, y, z in make_hand((0, 1, 2, 3), True)
    ]

    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            return SimpleNamespace(
                hand_landmarks=[normalized],
                hand_world_landmarks=[world],
                handedness=[],
            )

        def close(self):
            pass

    detector = HandDetector(landmarker=FakeLandmarker())
    frame = CameraFrame(32, 24, 2.5, image=np.zeros((24, 32, 3), dtype=np.uint8))

    observation = detector.detect(frame)

    assert observation is not None
    assert observation.world_landmark_positions is not None
    assert len(observation.world_landmark_positions) == 21
    assert FingerAnalyzer().analyze(observation) == FingerState(True, True, True, True, True)
    detector.close()


def test_detector_returns_none_without_a_hand():
    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            return SimpleNamespace(hand_landmarks=[], handedness=[])

        def close(self):
            pass

    detector = HandDetector(landmarker=FakeLandmarker())
    frame = CameraFrame(8, 8, 1.0, image=np.zeros((8, 8, 3), dtype=np.uint8))
    assert detector.detect(frame) is None
    detector.close()


def test_detector_returns_two_hands_with_independent_handedness():
    points = [
        SimpleNamespace(x=index / 21, y=(index + 1) / 21, z=0.0)
        for index in range(21)
    ]
    categories = [
        [SimpleNamespace(category_name="Right", score=0.91)],
        [SimpleNamespace(category_name="Left", score=0.87)],
    ]

    class FakeLandmarker:
        def detect_for_video(self, image, timestamp_ms):
            return SimpleNamespace(hand_landmarks=[points, points], handedness=categories)

        def close(self):
            pass

    detector = HandDetector(landmarker=FakeLandmarker())
    frame = CameraFrame(20, 16, 1.5, image=np.zeros((16, 20, 3), dtype=np.uint8))

    observations = detector.detect_all(frame)

    assert len(observations) == 2
    assert [hand.handedness for hand in observations] == ["Right", "Left"]
    assert [hand.handedness_score for hand in observations] == pytest.approx([0.91, 0.87])
