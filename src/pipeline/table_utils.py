from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

from src.schema import CanonicalBlock, CanonicalDocument, ParsedLabel

_HTML_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MATHML_RE = re.compile(r"(?is)<math\b[^>]*>.*?</math>")
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


def is_html_table_like(label_or_document: Any) -> bool:
    if isinstance(label_or_document, CanonicalDocument):
        return any(is_html_table_like(block) for block in label_or_document.blocks)

    if isinstance(label_or_document, CanonicalBlock):
        if _is_flowchart_block(label_or_document):
            return False
        if label_or_document.type == "table":
            return True
        if label_or_document.type != "chart":
            return False
        if str(label_or_document.sub_type or "").strip().lower() == "flowchart":
            return False
        return bool(
            _first_parseable_html_table_candidate(
                _block_table_candidates(label_or_document)
            )
        )

    if isinstance(label_or_document, ParsedLabel):
        image_type = str(label_or_document.image_type or "").strip().lower()
        if image_type == "flowchart":
            return False
        if image_type == "table":
            return True
        if image_type != "chart":
            return False
        if label_or_document.structured_label.kind == "mermaid" or label_or_document.flowchart_graph:
            return False
        candidates = [
            label_or_document.structured_label.content,
            *list(label_or_document.visible_text),
        ]
        return bool(_first_parseable_html_table_candidate(candidates))

    if isinstance(label_or_document, dict):
        if isinstance(label_or_document.get("blocks"), list):
            return is_html_table_like(
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
            return is_html_table_like(
                ParsedLabel(
                    image_type=str(label_or_document.get("image_type", "") or ""),
                    structured_label=structured if isinstance(structured, dict) else {},
                    visible_text=label_or_document.get("visible_text") or [],
                    flowchart_graph=label_or_document.get("flowchart_graph"),
                )
            )
        if "type" in label_or_document:
            try:
                return is_html_table_like(CanonicalBlock(**label_or_document))
            except Exception:
                block_type = str(label_or_document.get("type", "") or "").strip().lower()
                if block_type == "table":
                    return True
                if block_type != "chart":
                    return False
                if str(label_or_document.get("sub_type", "") or "").strip().lower() == "flowchart":
                    return False
                content = label_or_document.get("content")
                candidates: list[str] = []
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
                return bool(_first_parseable_html_table_candidate(candidates))
    return False


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
        if label_or_document.structured_label.kind == "mermaid" or label_or_document.flowchart_graph:
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
    if _looks_like_html_table(text):
        return "html"
    if _looks_like_markdown_table(text):
        return "markdown"
    return "none"


def extract_best_table_candidate(
    document: CanonicalDocument | None,
) -> dict[str, Any] | None:
    if not isinstance(document, CanonicalDocument):
        return None

    best_candidate: dict[str, Any] | None = None
    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        if not is_table_like(block):
            continue
        parsed_candidate = _first_parseable_table_candidate(_block_table_candidates(block))
        if parsed_candidate is None:
            continue
        table_text, html_text, table_ir, table_format = parsed_candidate
        candidate = {
            "block": block,
            "table_text": table_text,
            "html": html_text,
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
        if best_candidate is None or _candidate_cell_count(candidate) > _candidate_cell_count(best_candidate):
            best_candidate = candidate
    return best_candidate


def extract_best_html_table_candidate(
    document: CanonicalDocument | None,
) -> dict[str, Any] | None:
    return extract_best_table_candidate(document)


def normalize_cell_text(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFKC", value)
    value = _HTML_TAG_RE.sub(" ", value)
    value = value.replace("\xa0", " ")
    value = _THOUSAND_SEPARATOR_RE.sub("", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_latex_formulas(text_or_html: str) -> list[str]:
    value = str(text_or_html or "")
    formulas: list[str] = []
    seen: set[str] = set()

    for pattern in (
        _MATHML_RE,
        _DISPLAY_DOLLAR_RE,
        _DISPLAY_BRACKET_RE,
        _INLINE_PAREN_RE,
        _INLINE_DOLLAR_RE,
    ):
        for match in pattern.finditer(value):
            formula = match.group(0)
            normalized = normalize_latex_formula(formula)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            formulas.append(normalized)
    return formulas


def normalize_latex_formula(formula: str) -> str:
    value = html.unescape(str(formula or "")).strip()
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value)
    if value.lower().startswith("<math"):
        return re.sub(r">\s+<", "><", re.sub(r"\s+", " ", value)).strip()

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
    if lowered in {"l", "hz", "khz", "mhz", "ghz", "kg", "g", "mg", "ug", "lb", "lbs", "ms", "s", "min", "h", "mm", "cm", "m", "km"}:
        return lowered
    return unit


def _candidate_cell_count(candidate: dict[str, Any]) -> int:
    table_ir = candidate.get("table_ir")
    if table_ir is None:
        return 0
    return len(getattr(table_ir, "cells", []) or [])


def _is_flowchart_block(block: CanonicalBlock) -> bool:
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if block.structured_label.kind == "mermaid":
        return True
    return bool(block.flowchart_graph)


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


def _first_parseable_html_table_candidate(candidates: list[str]) -> tuple[str, Any] | None:
    best: tuple[str, Any] | None = None
    for candidate in candidates:
        if not _looks_like_html_table(candidate):
            continue
        table_ir = _parse_html_table(candidate)
        if not table_ir.parse_valid or not table_ir.has_table:
            continue
        if best is None or len(table_ir.cells) > len(best[1].cells):
            best = (candidate, table_ir)
    return best


def _first_parseable_table_candidate(
    candidates: list[str],
) -> tuple[str, str, Any, str] | None:
    best: tuple[str, str, Any, str] | None = None
    for candidate in candidates:
        table_text = str(candidate or "").strip()
        if not table_text:
            continue
        table_format = detect_table_format(table_text)
        if table_format == "html":
            html_text = table_text
        elif table_format == "markdown":
            html_text = _markdown_table_to_html(table_text)
        else:
            continue
        if not html_text:
            continue
        table_ir = _parse_html_table(html_text)
        if not table_ir.parse_valid or not table_ir.has_table:
            continue
        if best is None or len(table_ir.cells) > len(best[2].cells):
            best = (table_text, html_text, table_ir, table_format)
    return best


def _parse_html_table(html_text: str) -> Any:
    from src.pipeline.table_ir import parse_html_table

    return parse_html_table(html_text)


def _looks_like_html_table(value: str) -> bool:
    return bool(_HTML_TABLE_RE.search(str(value or "")))


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


def _markdown_table_to_html(markdown_text: str) -> str:
    lines = [line.rstrip() for line in str(markdown_text or "").splitlines()]
    start_index = -1
    for index, line in enumerate(lines):
        if _is_markdown_table_start(lines, index):
            start_index = index
            break
    if start_index < 0:
        return ""

    end_index = start_index + 2
    while end_index < len(lines):
        row = _split_markdown_table_row(lines[end_index])
        if not lines[end_index].strip() or row is None:
            break
        end_index += 1
    return _render_markdown_table_to_html(lines[start_index:end_index])


def _render_markdown_table_to_html(lines: list[str]) -> str:
    if len(lines) < 2:
        return ""
    header_cells = _split_markdown_table_row(lines[0]) or []
    body_rows = [
        _split_markdown_table_row(line) or []
        for line in lines[2:]
        if str(line).strip()
    ]
    column_count = len(header_cells)
    thead_html = "".join(f"<th>{html.escape(cell)}</th>" for cell in header_cells)
    tbody_html_rows: list[str] = []
    for row in body_rows:
        normalized_row = row[:column_count] + [""] * max(0, column_count - len(row))
        cells_html = "".join(f"<td>{html.escape(cell)}</td>" for cell in normalized_row)
        tbody_html_rows.append(f"<tr>{cells_html}</tr>")
    tbody_html = "".join(tbody_html_rows)
    return (
        "<table>"
        f"<thead><tr>{thead_html}</tr></thead>"
        f"<tbody>{tbody_html}</tbody>"
        "</table>"
    )
