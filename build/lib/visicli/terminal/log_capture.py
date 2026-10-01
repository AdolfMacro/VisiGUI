from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import threading


def default_log_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_root / "visicli" / "runtime.log"


class StderrLogCapture:
    """Keep native-library diagnostics out of the live TUI while retaining them on disk."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_log_path()
        self._saved_stderr: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._reader_error: OSError | None = None

    def start(self) -> None:
        if self._saved_stderr is not None:
            raise RuntimeError("Stderr log capture is already active")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.path.open("ab")
        read_fd: int | None = None
        write_fd: int | None = None
        saved_stderr: int | None = None
        redirected = False
        ready = False
        try:
            read_fd, write_fd = os.pipe()
            saved_stderr = os.dup(2)
            os.dup2(write_fd, 2)
            redirected = True
            os.close(write_fd)
            write_fd = None
            self._saved_stderr = saved_stderr
            self._read_fd = read_fd
            log_file.write(
                f"\n--- VisiCLI diagnostics {datetime.now(timezone.utc).isoformat()} ---\n".encode()
            )
            log_file.flush()
            self._thread = threading.Thread(
                target=self._drain,
                args=(read_fd, log_file),
                name="visicli-stderr-log",
                daemon=True,
            )
            self._thread.start()
            ready = True
        finally:
            if not ready:
                if redirected and saved_stderr is not None:
                    os.dup2(saved_stderr, 2)
                if saved_stderr is not None:
                    os.close(saved_stderr)
                if read_fd is not None:
                    os.close(read_fd)
                if write_fd is not None:
                    os.close(write_fd)
                if self._thread is not None and self._thread.is_alive():
                    self._thread.join()
                log_file.close()
                self._saved_stderr = None
                self._read_fd = None
                self._thread = None

    def _drain(self, read_fd: int, log_file) -> None:
        write_enabled = True
        while chunk := os.read(read_fd, 4096):
            if write_enabled:
                try:
                    log_file.write(chunk)
                    log_file.flush()
                except OSError as error:
                    self._reader_error = error
                    write_enabled = False
        try:
            log_file.close()
        except OSError as error:
            if self._reader_error is None:
                self._reader_error = error

    def close(self) -> None:
        if self._saved_stderr is None:
            return
        saved_stderr, self._saved_stderr = self._saved_stderr, None
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._read_fd is not None:
            os.close(self._read_fd)
            self._read_fd = None
        if self._reader_error is not None:
            error, self._reader_error = self._reader_error, None
            raise OSError(f"Could not write diagnostics to {self.path}: {error}") from error

    def __enter__(self) -> StderrLogCapture:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
