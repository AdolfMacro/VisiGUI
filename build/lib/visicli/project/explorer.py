from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from ..core.gesture import Gesture
from .model import ProjectGraph, ProjectNode


class ExplorerMode(Enum):
    RUN = auto()


class ProjectAction(Enum):
    BACK = "back to parent"
    PREVIOUS = "previous item"
    NEXT = "next item"
    OPEN = "enter selected item"
    SHOW_FUNCTIONS = "show functions"
    NEXT_PAGE = "next page"
    ZOOM_FILE = "enter selected file"


_DEFAULT_BINDINGS = {
    Gesture.FIST.name: ProjectAction.BACK,
    Gesture.INDEX.name: ProjectAction.NEXT,
    Gesture.THUMB_INDEX.name: ProjectAction.OPEN,
    Gesture.INDEX_MIDDLE_RING.name: ProjectAction.SHOW_FUNCTIONS,
    Gesture.INDEX_MIDDLE_RING_PINKY.name: ProjectAction.NEXT_PAGE,
}


@dataclass
class ExplorerController:
    graph: ProjectGraph | None
    mode: ExplorerMode = ExplorerMode.RUN
    selected_index: int = 0
    page: int = 0
    page_size: int = 1
    current_parent_id: str | None = None
    focused_file_id: str | None = None
    include_functions: bool = True
    last_action: str = "Ready"
    bindings: dict[str, ProjectAction] | None = None
    _navigation_stack: list[tuple[str | None, str | None, int, int, str | None]] = field(
        default_factory=list,
        repr=False,
    )
    _visible_node_cache: tuple[ProjectNode, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _children_cache: dict[str, tuple[ProjectNode, ...]] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.bindings is None:
            self.bindings = dict(_DEFAULT_BINDINGS)

    @property
    def nodes(self) -> tuple[ProjectNode, ...]:
        if self._visible_node_cache is None:
            if self.graph is None:
                source: tuple[ProjectNode, ...] = ()
            elif self.current_parent_id is None:
                source = tuple(node for node in self.graph.nodes if node.parent_id is None)
            else:
                source = self._children_by_parent().get(self.current_parent_id, ())
            if not self.include_functions:
                source = tuple(node for node in source if node.kind != "function")
            self._visible_node_cache = source
        return self._visible_node_cache

    def _children_by_parent(self) -> dict[str, tuple[ProjectNode, ...]]:
        if self._children_cache is None:
            grouped: dict[str, list[ProjectNode]] = {}
            if self.graph is not None:
                for node in self.graph.nodes:
                    if node.parent_id is not None:
                        grouped.setdefault(node.parent_id, []).append(node)
                for parent_id, children in grouped.items():
                    parent = self.graph.node(parent_id)
                    if parent is not None and parent.kind == "package":
                        children.sort(key=lambda child: 0 if child.kind == "file" else 1)
            self._children_cache = {
                parent_id: tuple(children)
                for parent_id, children in grouped.items()
            }
        return self._children_cache

    def _invalidate_nodes(self) -> None:
        self._visible_node_cache = None

    @property
    def selected_node(self) -> ProjectNode | None:
        nodes = self.nodes
        if not nodes:
            return None
        self.selected_index %= len(nodes)
        return nodes[self.selected_index]

    def children(self, node: ProjectNode) -> tuple[ProjectNode, ...]:
        children = self._children_by_parent().get(node.node_id, ())
        if not self.include_functions:
            children = tuple(child for child in children if child.kind != "function")
        return children

    @property
    def breadcrumb(self) -> tuple[ProjectNode, ...]:
        if self.graph is None:
            return ()
        node_ids = [state[0] for state in self._navigation_stack]
        if self.current_parent_id is not None:
            node_ids.append(self.current_parent_id)
        return tuple(
            node
            for node_id in node_ids
            if node_id is not None
            if (node := self.graph.node(node_id)) is not None
        )

    def choose_gesture(self, gesture_name: str) -> None:
        action = self.bindings.get(gesture_name) if self.bindings is not None else None
        if action is not None:
            self.apply(action)
        elif gesture_name == Gesture.OPEN_PALM.name:
            self.last_action = "Idle pose | no action"

    def confirm(self) -> None:
        self.apply(ProjectAction.OPEN)

    def back(self) -> None:
        self.apply(ProjectAction.BACK)

    def select(self, index: int) -> None:
        if not self.nodes:
            self.selected_index = 0
            self.page = 0
            self.last_action = "No items at this level"
            return
        if index < 0 or index >= len(self.nodes):
            raise IndexError("Selected project item index is out of range")
        self.selected_index = index
        self._update_page_from_selection()
        selected = self.selected_node
        self.last_action = f"Selected {selected.name}" if selected else "No item selected"

    def apply(self, action: ProjectAction) -> None:
        nodes = self.nodes
        if action is ProjectAction.BACK:
            if self._navigation_stack:
                parent_id, selected_node_id, selected_index, page, focused_file_id = self._navigation_stack.pop()
                self.current_parent_id = parent_id
                self.focused_file_id = focused_file_id
                self._invalidate_nodes()
                restored_index = next(
                    (
                        index for index, node in enumerate(self.nodes)
                        if node.node_id == selected_node_id
                    ),
                    None,
                )
                self.selected_index = (
                    restored_index if restored_index is not None
                    else min(selected_index, max(0, len(self.nodes) - 1))
                )
                self.page = page
                parent = self.graph.node(self.current_parent_id) if self.graph and self.current_parent_id else None
                self.last_action = f"Back to {parent.name}" if parent else "Back to project root"
            else:
                self.last_action = "Already at project root"
            return

        if action is ProjectAction.PREVIOUS:
            if nodes:
                self.selected_index = (self.selected_index - 1) % len(nodes)
                self._update_page_from_selection()
                selected = self.selected_node
                self.last_action = f"Selected {selected.name}" if selected else "No item selected"
            else:
                self.last_action = "No items at this level"
        elif action is ProjectAction.NEXT:
            if nodes:
                self.selected_index = (self.selected_index + 1) % len(nodes)
                self._update_page_from_selection()
                selected = self.selected_node
                self.last_action = f"Selected {selected.name}" if selected else "No item selected"
            else:
                self.last_action = "No items at this level"
        elif action in (ProjectAction.OPEN, ProjectAction.ZOOM_FILE):
            self._open_selected_item()
        elif action is ProjectAction.SHOW_FUNCTIONS:
            selected_node_id = self.selected_node.node_id if self.selected_node else None
            self.include_functions = not self.include_functions
            self._invalidate_nodes()
            restored_index = next(
                (
                    index for index, node in enumerate(self.nodes)
                    if node.node_id == selected_node_id
                ),
                None,
            )
            if restored_index is not None:
                self.selected_index = restored_index
            else:
                self.selected_index = min(self.selected_index, max(0, len(self.nodes) - 1))
            self._update_page_from_selection()
            self.last_action = "Functions visible" if self.include_functions else "Functions hidden"
        elif action is ProjectAction.NEXT_PAGE:
            self._next_page()

    def _open_selected_item(self) -> None:
        selected = self.selected_node
        if selected is None:
            self.last_action = "No item selected"
            return
        if not self.children(selected) and selected.kind != "file":
            self.last_action = f"No nested items in {selected.name}"
            return

        self._navigation_stack.append((
            self.current_parent_id,
            selected.node_id,
            self.selected_index,
            self.page,
            self.focused_file_id,
        ))
        self.current_parent_id = selected.node_id
        if selected.kind == "file":
            self.focused_file_id = selected.node_id
        self._invalidate_nodes()
        self.selected_index = 0
        self.page = 0
        self.last_action = f"Entered {selected.path}"

    def _update_page_from_selection(self) -> None:
        if not self.nodes:
            self.page = 0
            return
        self.page = self.selected_index // max(1, self.page_size)

    def _next_page(self) -> None:
        nodes = self.nodes
        if not nodes:
            self.last_action = "No items at this level"
            return
        page_count = max(1, (len(nodes) + max(1, self.page_size) - 1) // max(1, self.page_size))
        self.page = (self.page + 1) % page_count
        self.selected_index = min(self.page * max(1, self.page_size), len(nodes) - 1)
        self.last_action = f"Page {self.page + 1}/{page_count}"
