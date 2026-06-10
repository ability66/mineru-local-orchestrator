from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.schema import CanonicalBlock, CanonicalDocument, ParsedLabel

_DISPLAY_DOLLAR_RE = re.compile(r"(?s)\$\$(.+?)\$\$")
_DISPLAY_BRACKET_RE = re.compile(r"(?s)\\\[(.+?)\\\]")
_INLINE_PAREN_RE = re.compile(r"(?s)\\\((.+?)\\\)")
_INLINE_DOLLAR_RE = re.compile(r"(?s)(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^:?-{1,}:?$")
_THOUSAND_SEPARATOR_RE = re.compile(r"(?<=\d)[,\uFF0C](?=\d{3}(?:\D|$))")
_NUMERIC_RE = re.compile(
    r"(?P<number>[+-]?(?:\d{1,3}(?:[,\uFF0C]\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)"
    r"\s*"
    r"(?P<unit>%|‰|℃|°C|kg|g|mg|μg|ug|lb|lbs|ms|s|min|h|hz|khz|mhz|ghz|"
    r"ml|mL|l|L|mm|cm|m|km|m²|cm²|mm²|km²|㎡|亩|元|万元|亿元|美元|万美元|亿美元|"
    r"万亿|万|亿|人|次|倍|天|年|月|周|小时|分钟|秒)?",
    re.IGNORECASE,
)


@dataclass
class MarkdownTableCell:
    row_index: int
    col_index: int
    text: str
    normalized_text: str
    is_header: bool
    formulas: list[str] = field(default_factory=list)


@dataclass
class MarkdownTableIR:
    parse_valid: bool
    has_table: bool
    row_count: int
    col_count: int
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    cells: list[MarkdownTableCell] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def parse_markdown_table(markdown_text: str) -> MarkdownTableIR:
    lines = [line.rstrip() for line in str(markdown_text or "").splitlines()]
    start_index = -1
    for index, _line in enumerate(lines):
        if _is_markdown_table_start(lines, index):
            start_index = index
            break
    if start_index < 0:
        return MarkdownTableIR(
            parse_valid=False,
            has_table=False,
            row_count=0,
            col_count=0,
            diagnostics={"reason": "markdown_table_not_found"},
        )

    end_index = start_index + 2
    while end_index < len(lines):
        row = _split_markdown_table_row(lines[end_index])
        if not lines[end_index].strip() or row is None:
            break
        end_index += 1

    header = _split_markdown_table_row(lines[start_index]) or []
    if not header:
        return MarkdownTableIR(
            parse_valid=False,
            has_table=False,
            row_count=0,
            col_count=0,
            diagnostics={"reason": "markdown_table_header_missing"},
        )

    col_count = len(header)
    rows: list[list[str]] = []
    for line in lines[start_index + 2 : end_index]:
        row = _split_markdown_table_row(line)
        if row is None:
            continue
        normalized_row = row[:col_count] + [""] * max(0, col_count - len(row))
        rows.append(normalized_row)

    cells: list[MarkdownTableCell] = []
    for col_index, cell_text in enumerate(header):
        cells.append(
            MarkdownTableCell(
                row_index=0,
                col_index=col_index,
                text=cell_text,
                normalized_text=normalize_cell_text(cell_text),
                is_header=True,
                formulas=extract_latex_formulas(cell_text),
            )
        )
    for row_offset, row in enumerate(rows, start=1):
        for col_index, cell_text in enumerate(row):
            cells.append(
                MarkdownTableCell(
                    row_index=row_offset,
                    col_index=col_index,
                    text=cell_text,
                    normalized_text=normalize_cell_text(cell_text),
                    is_header=False,
                    formulas=extract_latex_formulas(cell_text),
                )
            )

    return MarkdownTableIR(
        parse_valid=True,
        has_table=True,
        row_count=len(rows) + 1,
        col_count=col_count,
        header=header,
        rows=rows,
        cells=cells,
        diagnostics={
            "start_index": start_index,
            "end_index": end_index,
            "body_row_count": len(rows),
        },
    )


def is_table_like(label_or_document: Any) -> bool:
    if isinstance(label_or_document, CanonicalDocument):
        return any(is_table_like(block) for block in label_or_document.blocks)

    if isinstance(label_or_document, CanonicalBlock):
        if _is_flowchart_block(label_or_document):
            return False
        if label_or_document.type == "table":
            return True
        if label_or_document.type != "chart":
            return False
        return bool(_first_parseable_table_candidate(_block_table_candidates(label_or_document)))

    if isinstance(label_or_document, ParsedLabel):
        image_type = str(label_or_document.image_type or "").strip().lower()
        if image_type == "flowchart":
            return False
        if image_type == "table":
            return True
        if image_type != "chart":
            return False
        if (
            label_or_document.structured_label.kind == "mermaid"
            or label_or_document.flowchart_graph
        ):
            return False
        candidates = [
            label_or_document.structured_label.content,
            *list(label_or_document.visible_text),
        ]
        return bool(_first_parseable_table_candidate(candidates))

    if isinstance(label_or_document, dict):
        if isinstance(label_or_document.get("blocks"), list):
            return is_table_like(
                CanonicalDocument(
                    document_id="dict",
                    source="dict",
                    blocks=[
                        CanonicalBlock(**item)
                        for item in label_or_document.get("blocks", [])
                        if isinstance(item, dict)
                    ],
                )
            )
        if "image_type" in label_or_document:
            structured = label_or_document.get("structured_label")
            return is_table_like(
                ParsedLabel(
                    image_type=str(label_or_document.get("image_type", "") or ""),
                    structured_label=structured if isinstance(structured, dict) else {},
                    visible_text=label_or_document.get("visible_text") or [],
                    flowchart_graph=label_or_document.get("flowchart_graph"),
                )
            )
        if "type" in label_or_document:
            try:
                return is_table_like(CanonicalBlock(**label_or_document))
            except Exception:
                block_type = str(label_or_document.get("type", "") or "").strip().lower()
                if block_type == "table":
                    return True
                if block_type != "chart":
                    return False
                if str(label_or_document.get("sub_type", "") or "").strip().lower() == "flowchart":
                    return False
                candidates: list[str] = []
                content = label_or_document.get("content")
                if isinstance(content, dict):
                    candidates.extend(
                        [
                            str(content.get("content", "") or ""),
                            str(content.get("table_body", "") or ""),
                        ]
                    )
                elif isinstance(content, str):
                    candidates.append(content)
                visible_text = label_or_document.get("visible_text")
                if isinstance(visible_text, list):
                    candidates.extend(str(item or "") for item in visible_text)
                return bool(_first_parseable_table_candidate(candidates))
    return False


def detect_table_format(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "none"
    return "markdown" if _looks_like_markdown_table(text) else "none"


def extract_best_table_candidate(
    document: CanonicalDocument | None,
    allow_non_table_chart_fallback: bool = False,
) -> dict[str, Any] | None:
    candidates = extract_table_candidates(
        document=document,
        allow_non_table_chart_fallback=allow_non_table_chart_fallback,
    )
    if not candidates:
        return None
    return _pick_best_table_candidate(candidates)


def extract_table_candidates(
    document: CanonicalDocument | None,
    allow_non_table_chart_fallback: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(document, CanonicalDocument):
        return []

    candidates: list[dict[str, Any]] = []
    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        parsed_candidate = None
        if is_table_like(block):
            parsed_candidate = _first_parseable_table_candidate(
                _block_table_candidates(block)
            )
        if parsed_candidate is not None:
            table_text, table_ir, table_format = parsed_candidate
            candidate = _build_table_candidate(
                block=block,
                table_text=table_text,
                table_ir=table_ir,
                table_format=table_format,
            )
            candidates.append(candidate)
            continue

        if not allow_non_table_chart_fallback:
            continue
        if not _is_non_flowchart_chart_origin_block(block):
            continue
        candidate = _build_chart_fallback_candidate(block)
        if candidate is None:
            continue
        candidates.append(candidate)
    return candidates


def normalize_cell_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\xa0", " ")
    value = _THOUSAND_SEPARATOR_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_latex_formulas(text: str) -> list[str]:
    value = str(text or "")
    formulas: list[str] = []
    seen: set[str] = set()
    for pattern in (
        _DISPLAY_DOLLAR_RE,
        _DISPLAY_BRACKET_RE,
        _INLINE_PAREN_RE,
        _INLINE_DOLLAR_RE,
    ):
        for match in pattern.finditer(value):
            formula = normalize_latex_formula(match.group(0))
            if not formula or formula in seen:
                continue
            seen.add(formula)
            formulas.append(formula)
    return formulas


def normalize_latex_formula(formula: str) -> str:
    value = unicodedata.normalize("NFKC", str(formula or "")).strip()
    if not value:
        return ""

    delimiters = [
        ("$$", "$$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$", "$"),
    ]
    for start, end in delimiters:
        if value.startswith(start) and value.endswith(end):
            value = value[len(start) : len(value) - len(end)].strip()
            break

    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_numeric_values_and_units(text: str) -> list[tuple[float, str | None]]:
    normalized = normalize_cell_text(text)
    results: list[tuple[float, str | None]] = []
    for match in _NUMERIC_RE.finditer(normalized):
        raw_number = str(match.group("number") or "").strip()
        if not raw_number:
            continue
        try:
            number = float(_THOUSAND_SEPARATOR_RE.sub("", raw_number))
        except ValueError:
            continue
        unit = _normalize_unit(match.group("unit"))
        results.append((number, unit))
    return results


def _normalize_unit(value: Any) -> str | None:
    unit = str(value or "").strip()
    if not unit:
        return None
    lowered = unit.lower()
    if lowered == "°c":
        return "℃"
    if lowered == "ml":
        return "ml"
    if lowered in {
        "l",
        "hz",
        "khz",
        "mhz",
        "ghz",
        "kg",
        "g",
        "mg",
        "ug",
        "lb",
        "lbs",
        "ms",
        "s",
        "min",
        "h",
        "mm",
        "cm",
        "m",
        "km",
    }:
        return lowered
    return unit


def _candidate_cell_count(candidate: dict[str, Any]) -> int:
    table_ir = candidate.get("table_ir")
    if not isinstance(table_ir, MarkdownTableIR):
        return 0
    return len(table_ir.cells)


def _pick_best_table_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    markdown_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("table_format", "") or "").strip().lower() == "markdown"
    ]
    if markdown_candidates:
        return max(markdown_candidates, key=_candidate_cell_count)
    return max(
        candidates,
        key=lambda candidate: len(str(candidate.get("table_text", "") or "")),
    )


def _is_flowchart_block(block: CanonicalBlock) -> bool:
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if block.structured_label.kind == "mermaid":
        return True
    return bool(block.flowchart_graph)


def _is_non_flowchart_chart_origin_block(block: CanonicalBlock) -> bool:
    if _is_flowchart_block(block):
        return False
    source_block_type = str(
        block.provenance.get("source_block_type", "") or ""
    ).strip().lower()
    source_sub_type = str(
        block.provenance.get("source_sub_type", "") or ""
    ).strip().lower()
    if source_sub_type == "flowchart":
        return False
    if block.type == "chart":
        return True
    return source_block_type == "chart"


def _block_table_candidates(block: CanonicalBlock) -> list[str]:
    content = block.content if isinstance(block.content, dict) else {}
    candidates: list[str] = []
    if block.type == "table":
        candidates.append(str(content.get("table_body", "") or ""))
    if block.type == "chart":
        candidates.append(str(content.get("content", "") or ""))
    if block.structured_label.content:
        candidates.append(block.structured_label.content)
    if block.text:
        candidates.append(block.text)
    candidates.extend(str(item or "") for item in block.visible_text)
    return candidates


def _first_parseable_table_candidate(
    candidates: list[str],
) -> tuple[str, MarkdownTableIR, str] | None:
    best: tuple[str, MarkdownTableIR, str] | None = None
    for candidate in candidates:
        table_text = str(candidate or "").strip()
        if not table_text:
            continue
        table_format = detect_table_format(table_text)
        if table_format != "markdown":
            continue
        table_ir = parse_markdown_table(table_text)
        if not table_ir.parse_valid or not table_ir.has_table:
            continue
        if best is None or len(table_ir.cells) > len(best[1].cells):
            best = (table_text, table_ir, table_format)
    return best


def _looks_like_markdown_table(value: str) -> bool:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return _is_markdown_table_start(lines, 0)


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header_cells = _split_markdown_table_row(lines[index])
    if header_cells is None:
        return False
    return _is_markdown_table_separator(
        lines[index + 1], expected_columns=len(header_cells)
    )


def _split_markdown_table_row(line: str) -> list[str] | None:
    stripped = str(line or "").strip()
    if "|" not in stripped:
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    if not cells or all(not cell for cell in cells):
        return None
    return cells


def _is_markdown_table_separator(line: str, expected_columns: int) -> bool:
    cells = _split_markdown_table_row(line)
    if cells is None or len(cells) != expected_columns:
        return False
    return all(_MARKDOWN_TABLE_SEPARATOR_RE.fullmatch(cell) for cell in cells)


def _build_table_candidate(
    block: CanonicalBlock,
    table_text: str,
    table_ir: MarkdownTableIR | None,
    table_format: str,
) -> dict[str, Any]:
    return {
        "block": block,
        "table_text": table_text,
        "table_ir": table_ir,
        "table_format": table_format,
        "block_id": block.block_id,
        "block_type": block.type,
        "sub_type": block.sub_type,
        "caption": block.caption_structured.brief or block.text,
        "visible_text": list(block.visible_text),
        "ocr_texts": [
            str(region.text or "").strip()
            for region in block.ocr_regions
            if str(region.text or "").strip()
        ],
    }


def _build_chart_fallback_candidate(block: CanonicalBlock) -> dict[str, Any] | None:
    raw_text = next(
        (
            str(candidate or "").strip()
            for candidate in _block_table_candidates(block)
            if str(candidate or "").strip()
        ),
        "",
    )
    if not raw_text:
        return None
    table_format = detect_table_format(raw_text)
    table_ir = parse_markdown_table(raw_text) if table_format == "markdown" else None
    return _build_table_candidate(
        block=block,
        table_text=raw_text,
        table_ir=table_ir if table_ir and table_ir.parse_valid else None,
        table_format=table_format,
    )
