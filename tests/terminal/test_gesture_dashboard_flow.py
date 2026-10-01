from visigui.core.contracts import GestureObservation
from visigui.core.gesture import Gesture
from visigui.core.hand import FingerState
from visigui.gesture.stabilizer import GestureStabilizer
from visigui.interaction.controller import InteractionController
from visigui.terminal.dashboard import DashboardState, render_dashboard


def test_stable_gesture_flows_into_dashboard_intent():
    stabilizer = GestureStabilizer(min_stable_duration=0.05, buffer_size=1)
    observation = GestureObservation(
        gesture_name=Gesture.INDEX_MIDDLE.name,
        confidence=None,
        timestamp=4.0,
    )
    event = None
    for _ in range(6):
        event = stabilizer.update(observation, 0.02) or event

    assert event is not None
    action = InteractionController().handle_gesture(event)
    lines = render_dashboard(
        DashboardState(
            mode="CAMERA",
            camera_ready=True,
            hand_detected=True,
            handedness="Right",
            finger_state=FingerState(index=True, middle=True),
            gesture_name=event.gesture.name,
            stable=stabilizer.stable_gesture is not None,
            intent=action.intent.name,
        ),
        80,
        24,
    )
    output = "\n".join(lines)

    assert "INDEX + MIDDLE" in output
    assert "STABLE" in output
    assert "INSPECT" in output
    assert "GEAR" not in output
