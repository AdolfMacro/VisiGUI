import pytest

from visigui.core.gesture import Gesture, GestureEvent
from visigui.core.intent import Intent
from visigui.interaction.controller import InteractionController


@pytest.fixture
def event_factory():
    def make_event(gesture):
        return GestureEvent(
            gesture=gesture,
            previous_gesture=Gesture.UNKNOWN,
            timestamp=1.0,
        )

    return make_event


@pytest.mark.parametrize(
    ("gesture", "expected"),
    (
        (Gesture.FIST, Intent.CLOSE),
        (Gesture.INDEX, Intent.SELECT),
        (Gesture.INDEX_MIDDLE, Intent.INSPECT),
        (Gesture.INDEX_MIDDLE_RING, Intent.EXPAND),
        (Gesture.INDEX_MIDDLE_RING_PINKY, Intent.EXPAND),
        (Gesture.OPEN_PALM, Intent.OPEN),
        (Gesture.PINCH, Intent.NONE),
        (Gesture.UNKNOWN, Intent.NONE),
    ),
)
def test_only_supported_shapes_resolve_to_intents(gesture, expected, event_factory):
    action = InteractionController().handle_gesture(event_factory(gesture))

    assert action.intent is expected


def test_two_fingers_resolve_by_interaction_context(event_factory):
    controller = InteractionController()
    event = event_factory(Gesture.INDEX_MIDDLE)

    assert controller.handle_gesture(event).intent is Intent.INSPECT
    controller.set_context("STRUCTURE")
    assert controller.handle_gesture(event).intent is Intent.EXPAND
    controller.set_context("UNKNOWN")
    assert controller.handle_gesture(event).intent is Intent.SELECT


def test_world_context_maps_pinched_and_spread_hand_states_to_zoom_intents(event_factory):
    controller = InteractionController()
    controller.set_context("WORLD")

    assert controller.handle_gesture(event_factory(Gesture.PINCH)).intent is Intent.ZOOM_OUT
    assert controller.handle_gesture(event_factory(Gesture.OPEN_PALM)).intent is Intent.ZOOM_IN


def test_unknown_gesture_does_not_manufacture_a_target_or_command(event_factory):
    action = InteractionController().handle_gesture(event_factory(Gesture.UNKNOWN))

    assert action.intent is Intent.NONE
    assert not hasattr(action, "command")
    assert not hasattr(action, "target")
