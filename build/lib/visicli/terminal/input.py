from __future__ import annotations

import sys


class InputHandler:
    def __init__(self) -> None:
        self._old_settings = None

    def __enter__(self) -> InputHandler:
        self.enable_raw_mode()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.disable_raw_mode()

    def enable_raw_mode(self) -> None:
        if self._old_settings is not None or not sys.stdin.isatty():
            return
        import termios
        import tty

        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

    def disable_raw_mode(self) -> None:
        if self._old_settings is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None
