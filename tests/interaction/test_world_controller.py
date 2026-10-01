from visigui.core.gesture import Gesture, GestureEvent
from visigui.core.hand import FingerState
from visigui.core.intent import Intent
from visigui.interaction.controller import InteractionController
from visigui.interaction.world_controller import DEFAULT_ZOOM_SPEED, TwoHandWorldController
from visigui.vision.tracking import HandState, TwoHandState


def _hand(
    side: str,
    x: float = 0.5,
    y: float = 0.5,
    gesture: Gesture = Gesture.OPEN_PALM,
    pinch_distance: float = 0.5,
    size: float = 0.3,
) -> HandState:
    return HandState(
        handedness=side,
        confidence=0.95,
        landmarks=((x, y, 0.0),) * 21,
        palm_position=(x, y, 0.0),
        normalized_position=(x, y, size),
        velocity=(0.0, 0.0),
        gesture=gesture,
        pinch_distance=pinch_distance,
        finger_state=FingerState(),
    )


def test_world_intents_map_contact_to_zoom_out_and_separation_to_zoom_in():
    interaction = InteractionController()
    interaction.set_context("WORLD")

    zoom_in = interaction.handle_gesture(
        GestureEvent(Gesture.PINCH, Gesture.UNKNOWN, 1.0)
    )
    zoom_out = interaction.handle_gesture(
        GestureEvent(Gesture.OPEN_PALM, Gesture.UNKNOWN, 1.1)
    )

    assert zoom_in.intent is Intent.ZOOM_OUT
    assert zoom_out.intent is Intent.ZOOM_IN


def test_left_hand_fist_activates_relative_steering_and_open_hand_stops_target():
    controller = TwoHandWorldController()
    fist = lambda x=0.5, y=0.5: _hand("Left", x, y, Gesture.FIST)

    initial = controller.update(TwoHandState(fist(), None, 1.0), 1 / 30)
    moved = controller.update(
        TwoHandState(fist(x=0.8, y=0.3), None, 1.03),
        1 / 30,
    )
    released = controller.update(
        TwoHandState(_hand("Left", 0.8, 0.3, Gesture.OPEN_PALM), None, 1.06),
        1 / 30,
    )

    assert initial.strafe == 0.0
    assert initial.vertical == 0.0
    assert moved.strafe > 0.0
    assert moved.vertical > 0.0
    assert released.strafe == 0.0
    assert released.vertical == 0.0


def test_left_hand_open_pose_does_not_move_world():
    controller = TwoHandWorldController()
    controller.update(
        TwoHandState(_hand("Left", gesture=Gesture.OPEN_PALM), None, 1.0),
        1 / 30,
    )

    motion = controller.update(
        TwoHandState(
            _hand("Left", x=0.9, y=0.2, gesture=Gesture.OPEN_PALM),
            None,
            1.03,
        ),
        1 / 30,
    )

    assert motion.strafe == 0.0
    assert motion.vertical == 0.0


def test_touching_thumb_and_index_activates_continuous_zoom_out():
    controller = TwoHandWorldController(pinch_activation_seconds=0.06)
    assert controller.max_zoom_speed == DEFAULT_ZOOM_SPEED
    outputs = []
    for index in range(5):
        outputs.append(controller.update(
            TwoHandState(
                None,
                _hand("Right", gesture=Gesture.PINCH, pinch_distance=0.06),
                1.0 + index / 30,
            ),
            1 / 30,
        ).depth)

    assert outputs[0] == 0.0
    assert outputs[-1] == -DEFAULT_ZOOM_SPEED


def test_any_separated_thumb_and_index_gap_continuously_zooms_in():
    controller = TwoHandWorldController(pinch_activation_seconds=0.06)
    outputs = []
    for index in range(5):
        outputs.append(controller.update(
            TwoHandState(
                None,
                _hand("Right", gesture=Gesture.OPEN_PALM, pinch_distance=0.3),
                1.0 + index / 30,
            ),
            1 / 30,
        ).depth)

    assert outputs[0] == 0.0
    assert outputs[-1] == DEFAULT_ZOOM_SPEED


def test_midrange_gap_continues_zoom_in_instead_of_stopping():
    controller = TwoHandWorldController(pinch_activation_seconds=0.0)
    for index in range(10):
        motion = controller.update(
            TwoHandState(
                None,
                _hand("Right", gesture=Gesture.OPEN_PALM, pinch_distance=0.55),
                1.0 + index / 30,
            ),
            1 / 30,
        )

    assert motion.depth == DEFAULT_ZOOM_SPEED


def test_zoom_threshold_hysteresis_prevents_jittering_at_contact_and_open_edges():
    controller = TwoHandWorldController(pinch_activation_seconds=0.0)
    controller.update(
        TwoHandState(
            None,
            _hand("Right", gesture=Gesture.PINCH, pinch_distance=0.10),
            1.0,
        ),
        1 / 30,
    )
    contact_release = controller.update(
        TwoHandState(
            None,
            _hand("Right", gesture=Gesture.PINCH, pinch_distance=0.16),
            1.03,
        ),
        1 / 30,
    )
    still_released = None
    for index in range(8):
        still_released = controller.update(
            TwoHandState(
                None,
                _hand("Right", gesture=Gesture.OPEN_PALM, pinch_distance=0.55),
                1.06 + index / 30,
            ),
            1 / 30,
        )
    spread = None
    for index in range(8):
        spread = controller.update(
            TwoHandState(
                None,
                _hand("Right", gesture=Gesture.OPEN_PALM, pinch_distance=1.1),
                1.33 + index / 30,
            ),
            1 / 30,
        )
    spread_release = controller.update(
        TwoHandState(
            None,
            _hand("Right", gesture=Gesture.OPEN_PALM, pinch_distance=0.9),
            1.6,
        ),
        1 / 30,
    )

    assert contact_release.depth < 0
    assert still_released is not None
    assert still_released.depth > 0
    assert spread is not None
    assert spread.depth > 0
    assert spread_release.depth > 0


def test_hand_loss_stops_zoom_and_clears_activation():
    controller = TwoHandWorldController(pinch_activation_seconds=0.0)
    zoom_in = controller.update(
        TwoHandState(
            None,
            _hand("Right", gesture=Gesture.PINCH, pinch_distance=0.05),
            1.0,
        ),
        1 / 30,
    )
    missing = controller.update(TwoHandState(None, None, 1.03), 1 / 30)
    midrange = controller.update(
        TwoHandState(
            None,
            _hand("Right", pinch_distance=0.5),
            1.06,
        ),
        1 / 30,
    )

    assert zoom_in.depth < 0
    assert missing.depth == 0.0
    assert midrange.depth > 0.0


def test_hand_apparent_size_does_not_change_zoom_endpoint_behavior():
    controller = TwoHandWorldController(pinch_activation_seconds=0.0)
    for index in range(3):
        motion = controller.update(
            TwoHandState(
                None,
                _hand(
                    "Right",
                    gesture=Gesture.PINCH,
                    pinch_distance=0.05,
                    size=0.1 + index,
                ),
                1.0 + index / 30,
            ),
            1 / 30,
        )

    assert motion.depth < 0
