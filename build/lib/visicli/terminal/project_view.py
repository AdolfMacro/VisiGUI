from __future__ import annotations

from ..project.explorer import ExplorerController
from ..project.model import ProjectGraph, ProjectNode


class ProjectViewport:
    """Render a bounded grid of hierarchy cards with concise child summaries."""

    def render(
        self,
        graph: ProjectGraph | None,
        controller: ExplorerController,
        width: int,
        height: int,
        input_label: str = "CAMERA",
        debug_line: str | None = None,
    ) -> tuple[str, ...]:
        if width <= 0 or height <= 0:
            return ()

        nodes = controller.nodes
        if nodes:
            controller.selected_index %= len(nodes)
        selected = nodes[controller.selected_index] if nodes else None
        items = controller.nodes
        symbol_count = (
            sum(node.kind in ("class", "function") for node in graph.nodes)
            if graph else 0
        )
        location = " / ".join(
            node.name for node in controller.breadcrumb
        ) or (graph.root.name if graph else "NO PROJECT")
        header = [
            f"PROJECT MAP | {location} | {input_label}",
            f"{len(graph.files) if graph else 0} files | {symbol_count} symbols | FUNCTIONS {'ON' if controller.include_functions else 'OFF'}",
        ]
        if debug_line:
            header.append(f"DEBUG | {debug_line}")

        if not items:
            message = (
                f"No nested items in {controller.breadcrumb[-1].name}"
                if controller.breadcrumb else "No Python items to display."
            )
            rows = [*header, message, "0 BACK TO PARENT"]
            if graph and graph.parse_errors:
                rows.append(f"PARSE ISSUES | {len(graph.parse_errors)}")
            return self._fit(rows, width, height)

        columns = min(3, max(1, width // 24))
        gap_x = 2
        cell_width = max(1, (width - gap_x * (columns - 1)) // columns)
        card_height = min(4, max(0, height - len(header) - 2))
        if card_height < 4:
            return self._fit(
                [*header, *(f"{'>' if node.node_id == (selected.node_id if selected else None) else ' '} {node.kind.upper()} {node.path}" for node in items)],
                width,
                height,
            )

        gap_y = 1
        scene_height = max(0, height - len(header) - 2)
        rows_per_page = max(1, (scene_height + gap_y) // (card_height + gap_y))
        page_size = max(1, columns * rows_per_page)
        controller.page_size = page_size
        page_count = max(1, (len(items) + page_size - 1) // page_size)
        controller.page = min(
            controller.selected_index // page_size,
            page_count - 1,
        )
        page_start = controller.page * page_size
        page_items = items[page_start:page_start + page_size]

        canvas = [[" "] * width for _ in range(scene_height)]
        visible_rows = (len(page_items) + columns - 1) // columns
        grid_height = visible_rows * card_height + max(0, visible_rows - 1) * gap_y
        grid_top = max(0, (scene_height - grid_height) // 2)

        for index, item in enumerate(page_items):
            row, column = divmod(index, columns)
            x = column * (cell_width + gap_x)
            y = grid_top + row * (card_height + gap_y)
            self._draw_item_card(
                canvas,
                x,
                y,
                cell_width,
                card_height,
                item,
                controller.children(item),
                selected.node_id if selected else None,
            )

        selected_label = (
            f"{selected.path}:{selected.line}"
            if selected else "none"
        )
        footer = [
            f"LEVEL {controller.page + 1}/{page_count} | SELECTED {selected_label} | {controller.last_action}",
            "0 PARENT 1 NEXT 2 ENTER 3 FUNCS 4 PAGE 5 IDLE Q QUIT",
        ]
        scene = ["".join(line).rstrip() for line in canvas]
        return self._fit([*header, *scene, *footer], width, height)

    @staticmethod
    def _draw_item_card(
        canvas: list[list[str]],
        x: int,
        y: int,
        width: int,
        height: int,
        item: ProjectNode,
        children: tuple[ProjectNode, ...],
        selected_node_id: str | None,
    ) -> None:
        if width < 12 or height < 4:
            return

        marker = ">" if item.node_id == selected_node_id else " "
        kind_label = item.kind.upper()
        title = ProjectViewport._truncate(f" {marker} {kind_label} {item.name} ", width - 3)
        top = "┌─" + title + "─" * max(0, width - len(title) - 3) + "┐"
        ProjectViewport._put(canvas, x, y, top)
        categories = (
            ("P", tuple(child for child in children if child.kind == "package")),
            ("F", tuple(child for child in children if child.kind == "file")),
            ("C", tuple(child for child in children if child.kind == "class")),
            ("f", tuple(child for child in children if child.kind == "function")),
        )
        summaries = [
            (label, values)
            for label, values in categories
            if values
        ][:2]
        while len(summaries) < 2:
            summaries.append(("·", ()))
        ProjectViewport._put(
            canvas,
            x,
            y + 1,
            "│" + ProjectViewport._summary_line(summaries[0][0], tuple(child.name for child in summaries[0][1]), width - 4).ljust(width - 2) + "│",
        )
        ProjectViewport._put(
            canvas,
            x,
            y + 2,
            "│" + ProjectViewport._summary_line(summaries[1][0], tuple(child.name for child in summaries[1][1]), width - 4).ljust(width - 2) + "│",
        )
        ProjectViewport._put(canvas, x, y + 3, "└" + "─" * (width - 2) + "┘")

    @staticmethod
    def _summary_line(
        label: str,
        names: tuple[str, ...],
        width: int,
        marker: str = " ",
    ) -> str:
        prefix = f"{marker}{label} "
        available = max(0, width - len(prefix))
        if not names:
            return prefix + "—"
        shown = ""
        for index, name in enumerate(names):
            item = (", " if index else "") + name
            remaining = len(names) - index - 1
            suffix = f" +{remaining}" if remaining else ""
            if len(shown) + len(item) + len(suffix) > available:
                if not shown and available:
                    shown = ProjectViewport._truncate(name, max(0, available - len(f" +{len(names) - index}")))
                return prefix + shown + f" +{len(names) - index}"
            shown += item
        return prefix + shown

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[:width - 1] + "…"

    @staticmethod
    def _put(canvas: list[list[str]], x: int, y: int, text: str) -> None:
        if y < 0 or y >= len(canvas):
            return
        for offset, char in enumerate(text):
            target_x = x + offset
            if 0 <= target_x < len(canvas[y]):
                canvas[y][target_x] = char

    @staticmethod
    def _fit(rows: list[str], width: int, height: int) -> tuple[str, ...]:
        result = []
        for row in rows[:height]:
            if len(row) <= width:
                result.append(row)
            elif width > 1:
                result.append(row[:width - 1] + "…")
            else:
                result.append(row[:width])
        return tuple(result)
