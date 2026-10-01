import pytest
from visigui.gesture.stabilizer import GestureStabilizer
from visigui.core.contracts import GestureObservation
from visigui.core.gesture import Gesture, GestureEvent


@pytest.fixture
def obs_index():
    return GestureObservation(gesture_name='INDEX', confidence=1.0, timestamp=0.0)


@pytest.fixture
def obs_fist():
    return GestureObservation(gesture_name='FIST', confidence=1.0, timestamp=0.0)


def test_short_stable_duration_no_event(obs_index):
    stab = GestureStabilizer(min_stable_duration=0.5, buffer_size=5)
    for i in range(5):
        result = stab.update(obs_index, 0.01)
    assert result is None


def test_sufficient_stable_duration_emits_event(obs_index):
    stab = GestureStabilizer(min_stable_duration=0.1, buffer_size=5)
    event = None
    for i in range(20):
        result = stab.update(obs_index, 0.02)
        if result is not None:
            event = result
    assert event is not None
    assert event.gesture == Gesture.INDEX


def test_noisy_sequence_no_events(obs_index, obs_fist):
    stab = GestureStabilizer(min_stable_duration=0.2, buffer_size=5)
    events = []
    for i in range(20):
        o = obs_index if i % 2 == 0 else obs_fist
        result = stab.update(o, 0.01)
        if result is not None:
            events.append(result)
    assert len(events) == 0


def test_transition_emits_event(obs_fist, obs_index):
    stab = GestureStabilizer(min_stable_duration=0.15, buffer_size=5)
    events = []
    dt = 0.01
    for i in range(25):
        result = stab.update(obs_fist, dt)
        if result is not None:
            events.append(result)
    for i in range(25):
        result = stab.update(obs_index, dt)
        if result is not None:
            events.append(result)
    assert len(events) == 2
    assert events[0].gesture == Gesture.FIST
    assert events[1].gesture == Gesture.INDEX
    assert events[1].previous_gesture == Gesture.FIST


def test_repeated_identical_no_duplicates(obs_index):
    stab = GestureStabilizer(min_stable_duration=0.1, buffer_size=5)
    events = []
    for i in range(30):
        result = stab.update(obs_index, 0.01)
        if result is not None:
            events.append(result)
    assert len(events) == 1


def test_buffer_not_full_no_event(obs_index):
    stab = GestureStabilizer(min_stable_duration=0.01, buffer_size=5)
    for i in range(4):
        result = stab.update(obs_index, 0.01)
    assert result is None


def test_unknown_gesture_allows_same_shape_to_stabilize_again(obs_index):
    stabilizer = GestureStabilizer(min_stable_duration=0.02, buffer_size=1)
    first_event = None
    for _ in range(4):
        first_event = stabilizer.update(obs_index, 0.01) or first_event
    assert first_event is not None

    unknown = GestureObservation("UNKNOWN", confidence=None, timestamp=1.0)
    assert stabilizer.update(unknown, 0.01) is None
    second_event = None
    for _ in range(4):
        second_event = stabilizer.update(obs_index, 0.01) or second_event

    assert second_event is not None
    assert second_event.gesture is Gesture.INDEX


def test_last_stable_gesture_is_retained_while_a_new_gesture_is_being_confirmed(obs_index, obs_fist):
    stabilizer = GestureStabilizer(min_stable_duration=0.02, buffer_size=1)
    for _ in range(4):
        stabilizer.update(obs_index, 0.01)
    assert stabilizer.stable_gesture is not None

    stabilizer.update(obs_fist, 0.01)

    assert stabilizer.stable_gesture is not None
    assert stabilizer.stable_gesture.gesture_name == Gesture.INDEX.name


def test_non_finite_observation_timestamp_is_rejected(obs_index):
    stabilizer = GestureStabilizer()
    invalid = GestureObservation("INDEX", confidence=None, timestamp=float("nan"))

    with pytest.raises(ValueError, match="timestamp must be finite"):
        stabilizer.update(invalid, 0.01)
