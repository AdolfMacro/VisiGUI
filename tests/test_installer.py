from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "install.sh"


def test_linux_installer_passes_shell_syntax_check():
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_installer_help_is_available_without_installing():
    result = subprocess.run(
        ["bash", str(INSTALLER), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Install VisiGUI for the current Linux user" in result.stdout
    assert "--no-model" in result.stdout


def test_installer_rejects_unknown_arguments_without_side_effects(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["bash", str(INSTALLER), "--not-an-option"],
        check=False,
        capture_output=True,
        text=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
    )

    assert result.returncode != 0
    assert "unknown option" in result.stderr
    assert list(home.iterdir()) == []


def test_installer_creates_user_command_and_idempotent_shell_path(tmp_path):
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    install_dir = tmp_path / "data" / "visigui"
    bin_dir = home / ".local" / "bin"
    home.mkdir()
    fake_bin.mkdir()
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    python_stub = fake_bin / "python3"
    python_stub.write_text(
        """#!/usr/bin/env bash
set -e
if [[ "$1" == "-c" ]]; then
    if [[ "$2" == *"sys.version_info.major"* ]]; then
        printf '3.12\\n'
    fi
    exit 0
fi
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
    mkdir -p "$3/bin"
    cat > "$3/bin/python" <<'PYTHON'
#!/usr/bin/env bash
set -e
if [[ -n "${VISIGUI_TEST_LOG:-}" ]]; then
    printf '%s\\n' "$*" >> "$VISIGUI_TEST_LOG"
fi
if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "install" && "$4" != "--upgrade" ]]; then
    cat > "$(dirname "$0")/visigui" <<'COMMAND'
#!/usr/bin/env bash
exit 0
COMMAND
    chmod +x "$(dirname "$0")/visigui"
fi
exit 0
PYTHON
    chmod +x "$3/bin/python"
fi
"""
    )
    python_stub.chmod(0o755)
    environment = os.environ | {
        "HOME": str(home),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "VISIGUI_INSTALL_DIR": str(install_dir),
        "VISIGUI_BIN_DIR": str(bin_dir),
        "VISIGUI_TEST_LOG": str(tmp_path / "python-calls.log"),
        "TMPDIR": str(staging_root),
        "PYTHON": "python3",
    }

    first = subprocess.run(
        ["bash", str(INSTALLER), "--no-model"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert first.returncode == 0, first.stderr
    assert (bin_dir / "visigui").is_symlink()
    profile = (home / ".profile").read_text()
    assert str(bin_dir) in profile
    assert list(staging_root.iterdir()) == []

    second = subprocess.run(
        ["bash", str(INSTALLER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert second.returncode == 0, second.stderr
    assert (home / ".profile").read_text().count("# >>> VisiGUI user PATH >>>") == 1
    calls = (tmp_path / "python-calls.log").read_text().splitlines()
    assert "-m visigui.download_model" in calls
    staged_installs = [
        Path(call.split()[-1])
        for call in calls
        if call.startswith("-m pip install ") and "--upgrade" not in call
    ]
    assert len(staged_installs) == 2
    assert all(path.name == "project" and not path.exists() for path in staged_installs)
    assert list(staging_root.iterdir()) == []
