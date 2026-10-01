import os

from visigui.terminal.log_capture import StderrLogCapture


def test_stderr_capture_writes_native_diagnostics_to_log_not_terminal(tmp_path, capfd):
    log_path = tmp_path / "runtime.log"
    capture = StderrLogCapture(log_path)

    with capture:
        os.write(2, b"native warning for test\n")

    assert "native warning for test" in log_path.read_text()
    captured = capfd.readouterr()
    assert "native warning for test" not in captured.err
