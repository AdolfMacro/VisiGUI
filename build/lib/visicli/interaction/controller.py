from dataclasses import dataclass

from ..core.gesture import Gesture, GestureEvent
from ..core.intent import Intent


class IntentResolver:
    def resolve(self, gesture_event: GestureEvent, context: str = "PROJECT") -> Intent:
        mapping = {
            Gesture.INDEX: Intent.SELECT,
            Gesture.OPEN_PALM: Intent.OPEN,
            Gesture.FIST: Intent.CLOSE,
            Gesture.INDEX_MIDDLE_RING: Intent.EXPAND,
            Gesture.INDEX_MIDDLE_RING_PINKY: Intent.EXPAND,
        }
        if gesture_event.gesture == Gesture.INDEX_MIDDLE:
            if context == "PROJECT":
                return Intent.INSPECT
            if context == "STRUCTURE":
                return Intent.EXPAND
            return Intent.SELECT
        return mapping.get(gesture_event.gesture, Intent.NONE)


@dataclass(frozen=True)
class Action:
    intent: Intent


class InteractionController:
    def __init__(self):
        self.intent_resolver = IntentResolver()
        self.context = "PROJECT"

    def handle_gesture(self, event: GestureEvent) -> Action:
        intent = self.intent_resolver.resolve(event, self.context)
        return Action(intent=intent)

    def set_context(self, context: str) -> None:
        self.context = context
