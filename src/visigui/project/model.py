from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectNode:
    node_id: str
    name: str
    kind: str
    path: str
    line: int = 1
    parent_id: str | None = None


@dataclass(frozen=True)
class ProjectEdge:
    source_id: str
    target_id: str
    kind: str


@dataclass(frozen=True)
class ProjectGraph:
    root: Path
    nodes: tuple[ProjectNode, ...]
    edges: tuple[ProjectEdge, ...]
    parse_errors: tuple[str, ...] = ()
    skipped_files: int = 0

    def node(self, node_id: str) -> ProjectNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def children(self, node_id: str) -> tuple[ProjectNode, ...]:
        children = tuple(node for node in self.nodes if node.parent_id == node_id)
        parent = self.node(node_id)
        if parent is None or parent.kind != "package":
            return children
        return tuple(
            sorted(
                children,
                key=lambda node: (
                    0 if node.kind == "file" else 1 if node.kind == "package" else 2,
                    children.index(node),
                ),
            )
        )

    def nodes_of_kind(self, kind: str) -> tuple[ProjectNode, ...]:
        return tuple(node for node in self.nodes if node.kind == kind)

    @property
    def files(self) -> tuple[ProjectNode, ...]:
        return self.nodes_of_kind("file")

    @property
    def functions(self) -> tuple[ProjectNode, ...]:
        return self.nodes_of_kind("function")
