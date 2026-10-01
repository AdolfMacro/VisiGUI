import ast
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "visigui"


def _imports(package: str) -> set[str]:
    imported = set()
    for path in (SOURCE / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


def test_vision_does_not_depend_on_terminal():
    imports = _imports("vision")
    assert not any("terminal" in name for name in imports)


def test_gesture_does_not_depend_on_terminal():
    imports = _imports("gesture")
    assert not any("terminal" in name for name in imports)


def test_terminal_does_not_depend_on_gesture():
    imports = _imports("terminal")
    assert not any("gesture" in name for name in imports)


def test_dashboard_has_no_renderer_dependency():
    tree = ast.parse((SOURCE / "terminal" / "dashboard.py").read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name and ("rasterizer" in name or "renderer" in name) for name in imports)


def test_interaction_does_not_depend_on_terminal():
    imports = _imports("interaction")
    assert not any("terminal" in name for name in imports)


def test_application_has_no_retired_scene_dependency():
    tree = ast.parse((SOURCE / "app.py").read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name and ("simulation" in name or "gear" in name) for name in imports)
