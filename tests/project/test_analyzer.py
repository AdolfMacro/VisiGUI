from visigui.project import ProjectAnalyzer


def test_analyzer_builds_file_definition_import_and_call_edges_without_execution(tmp_path):
    root = tmp_path / "sample"
    package = root / "pkg"
    package.mkdir(parents=True)
    marker = root / "executed.txt"
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text(
        "def helper():\n    return 1\n"
    )
    (package / "main.py").write_text(
        "from .helpers import helper\n"
        "class Service:\n"
        "    def run(self):\n"
        "        return helper()\n"
        f"open({str(marker)!r}, 'w').write('bad')\n"
    )
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.py").write_text("def hidden(): pass\n")

    graph = ProjectAnalyzer().analyze(root)

    assert len(graph.files) == 3
    assert {node.name for node in graph.nodes_of_kind("class")} == {"Service"}
    assert {node.name for node in graph.functions} == {"run", "helper"}
    assert any(edge.kind == "imports" for edge in graph.edges)
    assert any(edge.kind == "calls" for edge in graph.edges)
    assert not marker.exists()
    assert all(".venv" not in node.path for node in graph.nodes)


def test_analyzer_reports_parse_errors_and_skips_oversized_files(tmp_path):
    (tmp_path / "broken.py").write_text("def broken(:\n")
    (tmp_path / "large.py").write_text("x = 1\n" * 20)
    graph = ProjectAnalyzer(max_source_bytes=16).analyze(tmp_path)

    assert len(graph.parse_errors) == 2
    assert graph.skipped_files == 2


def test_analyzer_rejects_non_directories(tmp_path):
    source = tmp_path / "file.py"
    source.write_text("")

    try:
        ProjectAnalyzer().analyze(source)
    except ValueError as error:
        assert "not a directory" in str(error)
    else:
        raise AssertionError("Expected a non-directory path to be rejected")


def test_analyzer_resolves_parent_relative_imports(tmp_path):
    package = tmp_path / "pkg"
    child = package / "child"
    child.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "base.py").write_text("def shared(): pass\n")
    (child / "__init__.py").write_text("from ..base import shared\n")

    graph = ProjectAnalyzer().analyze(tmp_path)
    imported_edges = [edge for edge in graph.edges if edge.kind == "imports"]

    assert len(imported_edges) == 1
    assert imported_edges[0].target_id == "file:pkg/base.py"


def test_analyzer_links_package_nodes_to_their_files(tmp_path):
    package = tmp_path / "pkg"
    child = package / "child"
    child.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "module.py").write_text("")
    (child / "__init__.py").write_text("")

    graph = ProjectAnalyzer().analyze(tmp_path)

    assert graph.children("package:pkg") == (
        graph.node("file:pkg/__init__.py"),
        graph.node("file:pkg/module.py"),
        graph.node("package:pkg.child"),
    )
    assert graph.children("package:pkg.child") == (
        graph.node("file:pkg/child/__init__.py"),
    )


def test_analyzer_resolves_import_aliases_without_false_attribute_edges(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "helpers.py").write_text("def helper():\n    return 1\n")
    (package / "main.py").write_text(
        "from .helpers import helper as alias\n"
        "alias()\n"
        "module = __import__('pkg.helpers')\n"
        "module.helper()\n"
    )

    graph = ProjectAnalyzer().analyze(tmp_path)
    call_edges = [edge for edge in graph.edges if edge.kind == "calls"]

    assert any(
        edge.source_id == "file:pkg/main.py"
        and edge.target_id == "function:pkg/helpers.py:1:helper"
        for edge in call_edges
    )
    assert sum(
        edge.source_id == "file:pkg/main.py"
        and edge.target_id == "function:pkg/helpers.py:1:helper"
        for edge in call_edges
    ) == 1
