import os
from io import StringIO
from types import SimpleNamespace

import pytest

import visigui.app as app
from visigui.app import _parse_args, main, run


def test_cli_rejects_invalid_frame_count():
    with pytest.raises(SystemExit) as error:
        _parse_args(["--demo", "--frames", "0"])
    assert error.value.code == 2


def test_cli_rejects_negative_camera_index():
    with pytest.raises(SystemExit) as error:
        _parse_args(["--camera-index", "-1"])
    assert error.value.code == 2


def test_cli_enables_ascii_art_option():
    args = _parse_args(["--demo", "--ascii-art"])

    assert args.ascii_art is True


def test_cli_can_disable_live_camera_popup():
    args = _parse_args(["--no-camera-view"])

    assert args.no_camera_view is True


def test_cli_switch_is_available():
    args = _parse_args(["--cli", "--camera", "--camera-index", "2"])

    assert args.cli is True
    assert args.camera is True
    assert args.camera_index == 2


def test_application_defaults_to_two_hand_world_gui(monkeypatch):
    from visigui.gui import world_window

    calls = []
    monkeypatch.setattr(
        world_window,
        "launch_world_gui",
        lambda **kwargs: calls.append(kwargs) or 17,
    )

    result = main(["--camera-index", "3"])

    assert result == 17
    assert calls == [
        {
            "camera_index": 3,
            "model_path": str(app.DEFAULT_MODEL_PATH),
            "camera_view": True,
            "demo": False,
        }
    ]


def test_cli_switch_preserves_terminal_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "run", lambda **kwargs: calls.append(kwargs) or 19)

    result = main(["--cli", "--camera", "--camera-index", "4", "--no-camera-view"])

    assert result == 19
    assert calls[0]["demo"] is False
    assert calls[0]["camera_index"] == 4
    assert calls[0]["camera_view"] is False


def test_finite_frames_keep_the_headless_cli_smoke_path(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "run", lambda **kwargs: calls.append(kwargs) or 23)

    assert main(["--demo", "--frames", "1"]) == 23
    assert calls[0]["demo"] is True
    assert calls[0]["max_frames"] == 1


def test_direct_run_rejects_invalid_frame_count():
    with pytest.raises(ValueError, match="max_frames must be positive"):
        run(demo=True, max_frames=0)


def test_finite_demo_prints_dashboard_not_a_3d_scene(capsys):
    assert run(demo=True, max_frames=1) == 0
    output = capsys.readouterr().out

    assert "PROJECT MAP" in output
    assert "KEYBOARD DEMO" in output
    assert "PROJECT MAP" in output
    assert "SET ACTION" not in output
    assert "GEAR" not in output
    assert "#" not in output


def test_legacy_ascii_art_option_keeps_the_input_panel_compact(capsys):
    assert run(demo=True, max_frames=1, ascii_art=True) == 0
    output = capsys.readouterr().out

    assert "HAND --" in output
    assert "NO HAND FOUND" in output
    assert "PROJECT MAP" in output
    assert "HOLD HAND IN VIEW" in output


def test_cli_validates_python_project_directory(tmp_path):
    args = _parse_args(["--demo", "-p", str(tmp_path)])
    assert args.project == tmp_path

    with pytest.raises(SystemExit) as error:
        _parse_args(["--demo", "-p", str(tmp_path / "missing")])
    assert error.value.code == 2


def test_finite_demo_loads_project_and_preserves_input_status(tmp_path, capsys):
    project = tmp_path / "sample"
    project.mkdir()
    (project / "main.py").write_text("def run():\n    return 1\n")

    assert run(demo=True, max_frames=1, project_path=project) == 0
    output = capsys.readouterr().out

    assert "sample" in output
    assert "1 files | 1 symbols" in output
    assert "NO HAND FOUND" in output
    assert "┌─" in output
    assert "RUN PROJECT" not in output


def test_interactive_keyboard_enters_selected_file_without_initial_menu(monkeypatch, tmp_path):
    project = tmp_path / "sample"
    project.mkdir()
    (project / "main.py").write_text("def run():\n    return 1\n")
    keys = iter((b"1", b"c"))
    ready_checks = iter((True, True))
    stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)

    class TtyOutput(StringIO):
        def isatty(self):
            return True

    class InputStub:
        def enable_raw_mode(self):
            pass

        def disable_raw_mode(self):
            pass

    stdout = TtyOutput()
    monkeypatch.setattr(app.sys, "stdin", stdin)
    monkeypatch.setattr(app.sys, "stdout", stdout)
    monkeypatch.setattr(app.os, "read", lambda _fd, _size: next(keys))
    monkeypatch.setattr(
        app.select,
        "select",
        lambda *_args: ([stdin], [], []) if next(ready_checks) else ([], [], []),
    )
    monkeypatch.setattr(app, "InputHandler", InputStub)
    monkeypatch.setattr(app, "_terminal_size", lambda: (100, 24))
    monkeypatch.setattr(app.time, "sleep", lambda _duration: None)

    assert run(demo=True, max_frames=2, project_path=project) == 0
    assert "Entered main.py" in stdout.getvalue()


def test_terminal_size_uses_small_reported_cell_grid(monkeypatch):
    monkeypatch.setattr(
        app.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((12, 6)),
    )

    assert app._terminal_size() == (12, 6)


def test_read_key_uses_single_unbuffered_byte(monkeypatch):
    monkeypatch.setattr(app.sys, "stdin", SimpleNamespace(fileno=lambda: 17))
    monkeypatch.setattr(app.os, "read", lambda descriptor, size: b"2")

    assert app._read_key() == "2"


def test_read_key_treats_eof_as_quit_signal(monkeypatch):
    monkeypatch.setattr(app.sys, "stdin", SimpleNamespace(fileno=lambda: 17))
    monkeypatch.setattr(app.os, "read", lambda descriptor, size: b"")

    assert app._read_key() is None
