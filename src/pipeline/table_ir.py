from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - covered by diagnostics fallback
    BeautifulSoup = None  # type: ignore[assignment]

from src.pipeline.table_utils import extract_latex_formulas, normalize_cell_text


@dataclass
class TableCell:
    id: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    tag: str
    text: str
    normalized_text: str
    formulas: list[str]
    rowspan: int
    colspan: int
    scope: str | None = None


@dataclass
class TableIR:
    parse_valid: bool
    has_table: bool
    row_count: int
    col_count: int
    cells: list[TableCell]
    grid: list[list[str | None]]
    caption: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def parse_html_table(html: str) -> TableIR:
    diagnostics: dict[str, Any] = {
        "parser_used": None,
        "table_count": 0,
        "selected_table_index": None,
        "invalid_spans": [],
        "overlaps": [],
        "warnings": [],
    }
    raw_html = str(html or "").strip()
    if not raw_html:
        diagnostics["warnings"].append("empty_html")
        return TableIR(
            parse_valid=True,
            has_table=False,
            row_count=0,
            col_count=0,
            cells=[],
            grid=[],
            caption=None,
            diagnostics=diagnostics,
        )

    if BeautifulSoup is None:
        diagnostics["warnings"].append("beautifulsoup4_unavailable")
        return TableIR(
            parse_valid=False,
            has_table=False,
            row_count=0,
            col_count=0,
            cells=[],
            grid=[],
            caption=None,
            diagnostics=diagnostics,
        )

    try:
        parser = "html5lib"
        soup = BeautifulSoup(raw_html, parser)
        diagnostics["parser_used"] = parser
    except Exception:
        try:
            parser = "html.parser"
            soup = BeautifulSoup(raw_html, parser)
            diagnostics["parser_used"] = parser
            diagnostics["warnings"].append("html5lib_parse_failed_fallback_to_html_parser")
        except Exception as exc:
            diagnostics["warnings"].append(f"html_parse_failed:{type(exc).__name__}")
            return TableIR(
                parse_valid=False,
                has_table=False,
                row_count=0,
                col_count=0,
                cells=[],
                grid=[],
                caption=None,
                diagnostics=diagnostics,
            )

    tables = soup.find_all("table")
    diagnostics["table_count"] = len(tables)
    if not tables:
        return TableIR(
            parse_valid=True,
            has_table=False,
            row_count=0,
            col_count=0,
            cells=[],
            grid=[],
            caption=None,
            diagnostics=diagnostics,
        )

    selected_table = None
    selected_table_index = 0
    selected_cell_count = -1
    for index, table in enumerate(tables):
        cell_count = _table_cell_count(table)
        if cell_count > selected_cell_count:
            selected_table = table
            selected_table_index = index
            selected_cell_count = cell_count

    if selected_table is None:
        diagnostics["warnings"].append("table_selection_failed")
        return TableIR(
            parse_valid=False,
            has_table=False,
            row_count=0,
            col_count=0,
            cells=[],
            grid=[],
            caption=None,
            diagnostics=diagnostics,
        )

    diagnostics["selected_table_index"] = selected_table_index
    rows = [
        row
        for row in selected_table.find_all("tr")
        if row.find_parent("table") is selected_table
    ]
    caption_tag = next(
        (
            caption
            for caption in selected_table.find_all("caption")
            if caption.find_parent("table") is selected_table
        ),
        None,
    )
    caption = None
    if caption_tag is not None:
        caption = normalize_cell_text(caption_tag.get_text(" ", strip=True)) or None

    cells: list[TableCell] = []
    grid: list[list[str | None]] = []
    for row_index, row in enumerate(rows):
        _ensure_grid_row(grid, row_index)
        row_cells = row.find_all(["th", "td"], recursive=False)
        for cell_tag in row_cells:
            rowspan = _normalize_span(
                cell_tag.get("rowspan"),
                default=1,
                diagnostics=diagnostics,
                axis="rowspan",
                row_index=row_index,
            )
            colspan = _normalize_span(
                cell_tag.get("colspan"),
                default=1,
                diagnostics=diagnostics,
                axis="colspan",
                row_index=row_index,
            )
            col_index = _find_next_fit(grid=grid, row_index=row_index, colspan=colspan)
            cell_id = f"cell_{len(cells) + 1:03d}"
            inner_html = "".join(str(child) for child in getattr(cell_tag, "contents", []))
            raw_text = cell_tag.get_text(" ", strip=True)
            formulas = extract_latex_formulas(inner_html or raw_text)

            for target_row in range(row_index, row_index + rowspan):
                _ensure_grid_row(grid, target_row)
                _ensure_grid_cols(grid[target_row], col_index + colspan)
                for target_col in range(col_index, col_index + colspan):
                    if grid[target_row][target_col] is not None:
                        diagnostics["overlaps"].append(
                            {
                                "row": target_row,
                                "col": target_col,
                                "existing": grid[target_row][target_col],
                                "incoming": cell_id,
                            }
                        )
                    grid[target_row][target_col] = cell_id

            cells.append(
                TableCell(
                    id=cell_id,
                    row_start=row_index,
                    row_end=row_index + rowspan - 1,
                    col_start=col_index,
                    col_end=col_index + colspan - 1,
                    tag=cell_tag.name.lower(),
                    text=raw_text.strip(),
                    normalized_text=normalize_cell_text(raw_text),
                    formulas=formulas,
                    rowspan=rowspan,
                    colspan=colspan,
                    scope=str(cell_tag.get("scope", "") or "").strip() or None,
                )
            )

    max_cols = max((len(row) for row in grid), default=0)
    for row in grid:
        _ensure_grid_cols(row, max_cols)

    return TableIR(
        parse_valid=True,
        has_table=True,
        row_count=len(grid),
        col_count=max_cols,
        cells=cells,
        grid=grid,
        caption=caption,
        diagnostics=diagnostics,
    )


def _table_cell_count(table: Any) -> int:
    count = 0
    for cell in table.find_all(["th", "td"]):
        if cell.find_parent("table") is table:
            count += 1
    return count


def _normalize_span(
    value: Any,
    default: int,
    diagnostics: dict[str, Any],
    axis: str,
    row_index: int,
) -> int:
    try:
        parsed = int(str(value or default).strip())
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        diagnostics["invalid_spans"].append(
            {"axis": axis, "row": row_index, "value": value, "corrected_to": 1}
        )
        return 1
    if str(value or "").strip() and parsed != int(str(value).strip()):
        diagnostics["invalid_spans"].append(
            {"axis": axis, "row": row_index, "value": value, "corrected_to": parsed}
        )
    return parsed


def _find_next_fit(grid: list[list[str | None]], row_index: int, colspan: int) -> int:
    row = grid[row_index]
    col_index = 0
    while True:
        _ensure_grid_cols(row, col_index + colspan)
        if all(row[target] is None for target in range(col_index, col_index + colspan)):
            return col_index
        col_index += 1


def _ensure_grid_row(grid: list[list[str | None]], row_index: int) -> None:
    while len(grid) <= row_index:
        grid.append([])


def _ensure_grid_cols(row: list[str | None], size: int) -> None:
    while len(row) < size:
        row.append(None)
