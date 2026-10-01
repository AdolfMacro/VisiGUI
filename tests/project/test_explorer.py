from visigui.core.gesture import Gesture
from visigui.project.analyzer import ProjectAnalyzer
from visigui.project.explorer import ExplorerController, ExplorerMode, ProjectAction
from visigui.terminal.project_view import ProjectViewport


def _graph(tmp_path):
    (tmp_path / "app.py").write_text(
        "class App:\n"
        "    def run(self):\n"
        "        return helper()\n"
        "def helper():\n"
        "    return 1\n"
    )
    return ProjectAnalyzer().analyze(tmp_path)


def test_explorer_starts_in_run_mode_and_default_gestures_navigate(tmp_path):
    graph = _graph(tmp_path)
    controller = ExplorerController(graph, mode=ExplorerMode.RUN)

    assert controller.mode.name == "RUN"
    assert controller.include_functions

    controller.choose_gesture(Gesture.INDEX.name)
    assert controller.selected_index == 0
    assert controller.selected_node.name == "app.py"


def test_thumb_index_opens_selected_file_and_fist_returns_to_project_map(tmp_path):
    controller = ExplorerController(_graph(tmp_path))

    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    assert controller.focused_file_id == "file:app.py"
    assert controller.current_parent_id == "file:app.py"
    assert controller.last_action == "Entered app.py"

    controller.choose_gesture(Gesture.FIST.name)
    assert controller.focused_file_id is None
    assert controller.current_parent_id is None
    assert controller.last_action == "Back to project root"


def test_fist_returns_one_hierarchy_level_at_a_time(tmp_path):
    controller = ExplorerController(_graph(tmp_path))

    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    assert controller.current_parent_id == "file:app.py"
    assert controller.selected_node.name == "App"

    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    assert controller.selected_node.name == "run"

    controller.choose_gesture(Gesture.FIST.name)
    assert controller.current_parent_id == "file:app.py"
    assert controller.selected_node.name == "App"

    controller.choose_gesture(Gesture.FIST.name)
    assert controller.current_parent_id is None
    assert controller.selected_node.name == "app.py"
    assert controller.last_action == "Back to project root"


def test_package_file_class_and_method_navigation_preserves_parent_selection(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "service.py").write_text(
        "class Service:\n"
        "    def start(self):\n"
        "        return 1\n"
    )
    graph = ProjectAnalyzer().analyze(tmp_path)
    controller = ExplorerController(graph)

    assert [node.kind for node in controller.nodes] == ["package"]
    controller.confirm()
    assert [node.name for node in controller.nodes] == ["__init__.py", "service.py"]
    controller.choose_gesture(Gesture.INDEX.name)
    assert controller.selected_node.name == "service.py"
    controller.confirm()
    assert controller.selected_node.name == "Service"
    controller.confirm()
    assert controller.selected_node.name == "start"

    controller.back()
    assert controller.selected_node.name == "Service"
    controller.back()
    assert controller.selected_node.name == "service.py"
    controller.back()
    assert controller.selected_node.name == "pkg"
    assert controller.current_parent_id is None


def test_open_palm_is_a_visible_idle_pose_without_an_action(tmp_path):
    controller = ExplorerController(_graph(tmp_path))

    controller.choose_gesture(Gesture.OPEN_PALM.name)

    assert controller.current_parent_id is None
    assert controller.selected_index == 0
    assert Gesture.OPEN_PALM.name not in controller.bindings
    assert controller.last_action == "Idle pose | no action"


def test_file_page_is_restored_after_zoom_and_back(tmp_path):
    for index in range(5):
        (tmp_path / f"module_{index}.py").write_text("")
    graph = ProjectAnalyzer().analyze(tmp_path)
    controller = ExplorerController(graph)
    viewport = ProjectViewport()
    viewport.render(graph, controller, 45, 14)

    controller.choose_gesture(Gesture.INDEX_MIDDLE_RING_PINKY.name)
    assert controller.page == 1
    assert controller.selected_node.name == "module_2.py"

    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    assert controller.focused_file_id == "file:module_2.py"
    controller.choose_gesture(Gesture.FIST.name)

    assert controller.page == 1
    assert controller.selected_node.name == "module_2.py"


def test_resize_keeps_selected_item_visible_on_its_recomputed_page(tmp_path):
    for index in range(12):
        (tmp_path / f"module_{index:02}.py").write_text("")
    graph = ProjectAnalyzer().analyze(tmp_path)
    controller = ExplorerController(graph)
    viewport = ProjectViewport()

    viewport.render(graph, controller, 45, 14)
    for _ in range(7):
        controller.choose_gesture(Gesture.INDEX.name)
    selected_name = controller.selected_node.name

    resized = "\n".join(viewport.render(graph, controller, 24, 8))

    assert selected_name in resized


def test_repeated_file_zoom_and_back_rebuilds_the_visible_project_map(tmp_path):
    (tmp_path / "alpha.py").write_text(
        "class Alpha:\n"
        "    def open(self):\n"
        "        return 1\n"
    )
    (tmp_path / "beta.py").write_text(
        "class Beta:\n"
        "    def close(self):\n"
        "        return 2\n"
    )
    graph = ProjectAnalyzer().analyze(tmp_path)
    controller = ExplorerController(graph)
    viewport = ProjectViewport()

    initial = "\n".join(viewport.render(graph, controller, 90, 24))
    assert "alpha.py" in initial
    assert "beta.py" in initial

    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    focused = "\n".join(viewport.render(graph, controller, 90, 24))
    assert "alpha.py" in focused
    assert "beta.py" not in focused
    assert "open" in focused

    controller.choose_gesture(Gesture.FIST.name)
    restored = "\n".join(viewport.render(graph, controller, 45, 14))
    assert "alpha.py" in restored
    assert "beta.py" in restored
    assert controller.focused_file_id is None

    controller.choose_gesture(Gesture.INDEX.name)
    controller.choose_gesture(Gesture.INDEX.name)
    controller.choose_gesture(Gesture.INDEX.name)
    assert controller.selected_node.name == "beta.py"
    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    zoomed_again = "\n".join(viewport.render(graph, controller, 90, 24))
    assert "beta.py" in zoomed_again
    assert "alpha.py" not in zoomed_again


def test_project_viewport_draws_file_cells_with_nested_symbols(tmp_path):
    graph = _graph(tmp_path)
    controller = ExplorerController(graph)
    rows = ProjectViewport().render(graph, controller, 54, 20)
    output = "\n".join(rows)

    assert len(rows) <= 20
    assert all(len(row) <= 54 for row in rows)
    assert "┌─" in output
    assert "└" in output
    assert "#" not in output
    assert "C App" in output
    assert "helper" in output
    assert "3D" not in output

    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    controller.choose_gesture(Gesture.THUMB_INDEX.name)
    class_view = "\n".join(ProjectViewport().render(graph, controller, 54, 20))
    assert "run" in class_view


def test_viewport_adapts_to_short_and_narrow_dimensions(tmp_path):
    graph = _graph(tmp_path)
    controller = ExplorerController(graph, mode=ExplorerMode.RUN)
    view = ProjectViewport()

    for width, height in ((1, 1), (9, 5), (20, 8), (45, 14), (100, 30)):
        lines = view.render(graph, controller, width, height)
        assert len(lines) <= height
        assert all(len(line) <= width for line in lines)
