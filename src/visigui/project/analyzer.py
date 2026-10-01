from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path

from .model import ProjectEdge, ProjectGraph, ProjectNode

_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}
_MAX_FILES = 10_000
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_SYMBOLS_PER_FILE = 20_000
_MAX_TOTAL_SYMBOLS = 100_000


class ProjectAnalysisError(ValueError):
    pass


@dataclass
class _FileInfo:
    path: Path
    module_name: str
    package_name: str
    file_node_id: str
    imports: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    definitions: list[ProjectNode] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)


class ProjectAnalyzer:
    """Build a static Python structure graph; project code is never imported or executed."""

    def __init__(
        self,
        max_files: int = _MAX_FILES,
        max_source_bytes: int = _MAX_SOURCE_BYTES,
        max_total_source_bytes: int = _MAX_TOTAL_SOURCE_BYTES,
        max_symbols_per_file: int = _MAX_SYMBOLS_PER_FILE,
        max_total_symbols: int = _MAX_TOTAL_SYMBOLS,
    ):
        if min(
            max_files,
            max_source_bytes,
            max_total_source_bytes,
            max_symbols_per_file,
            max_total_symbols,
        ) <= 0:
            raise ValueError("Project analysis limits must be positive")
        self.max_files = max_files
        self.max_source_bytes = max_source_bytes
        self.max_total_source_bytes = max_total_source_bytes
        self.max_symbols_per_file = max_symbols_per_file
        self.max_total_symbols = max_total_symbols

    def analyze(self, project_path: str | Path) -> ProjectGraph:
        root = Path(project_path).expanduser().resolve()
        if not root.exists():
            raise ProjectAnalysisError(f"Project directory does not exist: {root}")
        if not root.is_dir():
            raise ProjectAnalysisError(f"Project path is not a directory: {root}")

        paths = self._python_files(root)
        skipped_files = max(0, len(paths) - self.max_files)
        paths = paths[: self.max_files]
        nodes: list[ProjectNode] = []
        edges: list[ProjectEdge] = []
        errors: list[str] = []
        infos: list[_FileInfo] = []
        package_ids: dict[str, str] = {}
        module_to_file: dict[str, str] = {}
        total_source_bytes = 0
        total_symbols = 0

        directories = sorted(
            {
                parent
                for path in paths
                for parent in path.relative_to(root).parents
                if str(parent) != "."
            },
            key=lambda item: (len(item.parts), item.as_posix()),
        )
        package_dirs = {
            path.parent.relative_to(root)
            for path in paths
            if path.name == "__init__.py"
        }
        for relative_dir in directories:
            if relative_dir not in package_dirs:
                continue
            package_name = ".".join(relative_dir.parts)
            package_id = f"package:{package_name}"
            parent_dir = relative_dir.parent
            parent_id = package_ids.get(str(parent_dir)) if str(parent_dir) != "." else None
            nodes.append(ProjectNode(
                node_id=package_id,
                name=relative_dir.name,
                kind="package",
                path=relative_dir.as_posix(),
                parent_id=parent_id,
            ))
            package_ids[str(relative_dir)] = package_id
            if parent_id:
                edges.append(ProjectEdge(parent_id, package_id, "contains"))

        for path in paths:
            relative = path.relative_to(root)
            module_name = self._module_name(relative)
            file_id = f"file:{relative.as_posix()}"
            parent_id = package_ids.get(str(relative.parent))
            file_node = ProjectNode(
                node_id=file_id,
                name=path.name,
                kind="file",
                path=relative.as_posix(),
                parent_id=parent_id,
            )
            nodes.append(file_node)
            if parent_id:
                edges.append(ProjectEdge(parent_id, file_id, "contains"))
            package_name = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
            info = _FileInfo(
                path=path,
                module_name=module_name,
                package_name=package_name,
                file_node_id=file_id,
            )
            infos.append(info)
            module_to_file[module_name] = file_id

            try:
                source = path.read_bytes()
                if len(source) > self.max_source_bytes:
                    errors.append(
                        f"{relative.as_posix()}: skipped (larger than "
                        f"{self.max_source_bytes} bytes)"
                    )
                    skipped_files += 1
                    continue
                if total_source_bytes + len(source) > self.max_total_source_bytes:
                    errors.append(
                        f"{relative.as_posix()}: scan byte limit reached "
                        f"({self.max_total_source_bytes} bytes)"
                    )
                    skipped_files += 1
                    continue
                total_source_bytes += len(source)
                tree = ast.parse(source, filename=relative.as_posix())
                symbol_count = sum(
                    isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    for node in ast.walk(tree)
                )
                if symbol_count > self.max_symbols_per_file:
                    errors.append(
                        f"{relative.as_posix()}: skipped ({symbol_count} definitions; "
                        f"limit is {self.max_symbols_per_file})"
                    )
                    skipped_files += 1
                    continue
                if total_symbols + symbol_count > self.max_total_symbols:
                    errors.append(
                        f"{relative.as_posix()}: scan definition limit reached "
                        f"({self.max_total_symbols})"
                    )
                    skipped_files += 1
                    continue
                total_symbols += symbol_count
            except (OSError, UnicodeError, SyntaxError, RecursionError) as error:
                line = f":{error.lineno}" if isinstance(error, SyntaxError) and error.lineno else ""
                errors.append(f"{relative.as_posix()}{line}: {error}")
                skipped_files += 1
                continue

            self._collect_tree(tree, relative.as_posix(), file_id, info, nodes, edges)

        for info in infos:
            for imported_module in info.imports:
                target = self._resolve_import(imported_module, info.package_name, module_to_file)
                if target is not None and target != info.file_node_id:
                    edges.append(ProjectEdge(info.file_node_id, target, "imports"))

        definition_by_name: dict[str, list[ProjectNode]] = {}
        definitions_by_file: dict[str, list[ProjectNode]] = {}
        for info in infos:
            for node in info.definitions:
                definition_by_name.setdefault(node.name, []).append(node)
                definitions_by_file.setdefault(info.file_node_id, []).append(node)

        for info in infos:
            for caller_id, call_name in info.calls:
                target = self._resolve_call_target(
                    call_name,
                    info.package_name,
                    module_to_file,
                    definition_by_name,
                    definitions_by_file,
                )
                if target is not None:
                    edges.append(ProjectEdge(caller_id, target.node_id, "calls"))

        unique_edges = {
            (edge.source_id, edge.target_id, edge.kind): edge
            for edge in edges
            if edge.source_id != edge.target_id
        }
        return ProjectGraph(
            root=root,
            nodes=tuple(nodes),
            edges=tuple(unique_edges.values()),
            parse_errors=tuple(errors),
            skipped_files=skipped_files,
        )

    def _python_files(self, root: Path) -> list[Path]:
        paths: list[Path] = []

        def raise_walk_error(error: OSError) -> None:
            raise ProjectAnalysisError(f"Cannot scan project directory {error.filename}: {error}") from error

        for directory, child_dirs, filenames in os.walk(
            root,
            followlinks=False,
            onerror=raise_walk_error,
        ):
            child_dirs[:] = sorted(
                name for name in child_dirs
                if name not in _IGNORED_DIRECTORIES and not name.startswith(".")
            )
            base = Path(directory)
            for name in sorted(filenames):
                if name.endswith(".py"):
                    path = base / name
                    if not path.is_symlink():
                        paths.append(path)
        return sorted(paths, key=lambda path: path.relative_to(root).as_posix())

    @staticmethod
    def _module_name(relative: Path) -> str:
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _collect_tree(
        self,
        tree: ast.Module,
        relative_path: str,
        file_id: str,
        info: _FileInfo,
        nodes: list[ProjectNode],
        edges: list[ProjectEdge],
    ) -> None:
        scope_stack = [file_id]

        class Collector(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                node_id = f"class:{relative_path}:{node.lineno}:{node.name}"
                model_node = ProjectNode(
                    node_id=node_id,
                    name=node.name,
                    kind="class",
                    path=relative_path,
                    line=node.lineno,
                    parent_id=scope_stack[-1],
                )
                nodes.append(model_node)
                info.definitions.append(model_node)
                edges.append(ProjectEdge(scope_stack[-1], node_id, "contains"))
                scope_stack.append(node_id)
                for child in node.body:
                    self.visit(child)
                scope_stack.pop()

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                node_id = f"function:{relative_path}:{node.lineno}:{node.name}"
                model_node = ProjectNode(
                    node_id=node_id,
                    name=node.name,
                    kind="function",
                    path=relative_path,
                    line=node.lineno,
                    parent_id=scope_stack[-1],
                )
                nodes.append(model_node)
                info.definitions.append(model_node)
                edges.append(ProjectEdge(scope_stack[-1], node_id, "contains"))
                scope_stack.append(node_id)
                for child in node.body:
                    self.visit(child)
                scope_stack.pop()

            visit_FunctionDef = _visit_function
            visit_AsyncFunctionDef = _visit_function

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    info.imports.add(alias.name)
                    local_name = alias.asname or alias.name.split(".", 1)[0]
                    info.aliases[local_name] = alias.name

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    base_module = "." * node.level + node.module
                    info.imports.add(base_module)
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        qualified_name = f"{base_module}.{alias.name}"
                        info.imports.add(qualified_name)
                        info.aliases[alias.asname or alias.name] = qualified_name
                elif node.level:
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        qualified_name = "." * node.level + alias.name
                        info.imports.add(qualified_name)
                        info.aliases[alias.asname or alias.name] = qualified_name

            def visit_Call(self, node: ast.Call) -> None:
                if isinstance(node.func, ast.Name):
                    call_name = info.aliases.get(node.func.id, node.func.id)
                    info.calls.append((scope_stack[-1], call_name))
                elif isinstance(node.func, ast.Attribute):
                    base_name = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                    if base_name is not None:
                        base_target = info.aliases.get(base_name, base_name)
                        info.calls.append((scope_stack[-1], f"{base_target}.{node.func.attr}"))
                self.generic_visit(node)

        Collector().visit(tree)

    def _resolve_call_target(
        self,
        call_name: str,
        current_package: str,
        module_to_file: dict[str, str],
        definition_by_name: dict[str, list[ProjectNode]],
        definitions_by_file: dict[str, list[ProjectNode]],
    ) -> ProjectNode | None:
        if "." in call_name:
            module_name, _, symbol_name = call_name.rpartition(".")
            if not module_name:
                candidates = definition_by_name.get(call_name, ())
                if len(candidates) == 1:
                    return candidates[0]
                return None
            file_id = self._resolve_import(module_name, current_package, module_to_file)
            if file_id is None:
                return None
            candidates = [
                node for node in definitions_by_file.get(file_id, ())
                if node.name == symbol_name
            ]
            if len(candidates) == 1:
                return candidates[0]
            return None

        candidates = definition_by_name.get(call_name, ())
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _resolve_import(
        imported: str,
        current_package: str,
        module_to_file: dict[str, str],
    ) -> str | None:
        if imported.startswith("."):
            level = len(imported) - len(imported.lstrip("."))
            suffix = imported[level:]
            package = current_package.split(".") if current_package else []
            if level > len(package):
                return None
            keep = len(package) - level + 1
            base = package[:keep]
            candidate = ".".join([*base, *([suffix] if suffix else [])])
        else:
            candidate = imported
        parts = candidate.split(".") if candidate else []
        while parts:
            match = module_to_file.get(".".join(parts))
            if match:
                return match
            parts.pop()
        return None
