from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.projection import project_document_for_single_block_view
from src.schema import (
    AdjudicationArtifact,
    CanonicalBlock,
    CanonicalDocument,
    ImageTask,
    ModelOutput,
    ParsedLabel,
    StructuredLabel,
)


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        "root": output_dir,
        "raw_mineru": output_dir / "raw" / "mineru",
        "raw_paddle": output_dir / "raw" / "paddle",
        "raw_glm": output_dir / "raw" / "glm",
        "raw_qwen": output_dir / "raw" / "qwen",
        "normalized_mineru": output_dir / "normalized" / "mineru",
        "normalized_paddle": output_dir / "normalized" / "paddle",
        "normalized_glm": output_dir / "normalized" / "glm",
        "normalized_qwen": output_dir / "normalized" / "qwen",
        "final": output_dir / "final",
        "page_md": output_dir / "page_md",
        "judge_stage2": output_dir / "judge_stage2",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def clear_previous_outputs(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_output_dirs(output_dir)


def initialize_summary_file(output_dir: Path) -> Path:
    ensure_output_dirs(output_dir)
    summary_path = output_dir / "summary.jsonl"
    summary_path.write_text("", encoding="utf-8")
    return summary_path


def append_summary_record(summary_path: Path, summary_record: dict[str, Any]) -> None:
    with summary_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary_record, ensure_ascii=False))
        file.write("\n")


def write_image_result(
    output_dir: Path,
    image_task: ImageTask,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    artifact: AdjudicationArtifact,
    stage2_records: list[dict[str, Any]] | None = None,
    extra_stage1_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    directories = ensure_output_dirs(output_dir)

    _write_json(
        directories["raw_mineru"] / f"{image_task.image_id}.json",
        _output_record(mineru_output),
    )
    _write_json(
        directories["raw_qwen"] / f"{image_task.image_id}.json",
        _output_record(qwen_output),
    )
    _write_json(
        directories["normalized_mineru"] / f"{image_task.image_id}.json",
        _normalized_output_payload(document=mineru_document, label=mineru_label),
    )
    _write_json(
        directories["normalized_qwen"] / f"{image_task.image_id}.json",
        _normalized_output_payload(document=qwen_document, label=qwen_label),
    )
    for role, payload in (extra_stage1_results or {}).items():
        normalized_role = str(role or "").strip().lower()
        if normalized_role not in {"paddle", "glm"}:
            continue
        _write_extra_stage1_result(
            directories=directories,
            image_id=image_task.image_id,
            role=normalized_role,
            payload=payload,
        )

    final_document = artifact.final_document
    markdown_pages, markdown_text = build_final_markdown_output(
        image_task=image_task,
        final_document=final_document,
        mineru_document=mineru_document,
    )
    final_output = build_final_output(
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        artifact=artifact,
        markdown_pages=markdown_pages,
    )
    content_list_v2 = build_content_list_v2(final_document)
    _remove_legacy_final_files(directories["final"], image_task.image_id)
    _write_json(directories["final"] / f"{image_task.image_id}.json", final_output)
    _write_text(directories["final"] / f"{image_task.image_id}.md", markdown_text)
    _write_json(
        directories["final"] / f"{image_task.image_id}_artifact.json",
        artifact.model_dump(),
    )
    if stage2_records is not None:
        _write_json(
            directories["judge_stage2"] / f"{image_task.image_id}.json",
            build_stage2_judge_payload(
                image_task=image_task,
                records=stage2_records,
            ),
        )

    return build_summary_record(
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        artifact=artifact,
        content_list_v2=content_list_v2,
    )


def build_final_output(
    image_task: ImageTask,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    artifact: AdjudicationArtifact,
    markdown_pages: list[str],
) -> dict[str, Any]:
    parsed = build_final_parsed_payload(
        image_task=image_task,
        document=artifact.final_document,
        markdown_pages=markdown_pages,
    )
    selected_output = _pick_selected_final_output(
        final_document=artifact.final_document,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
    )
    success = bool(artifact.final_document.blocks)
    errors = [
        str(value).strip()
        for value in (
            mineru_output.error if mineru_output is not None else None,
            qwen_output.error if qwen_output is not None else None,
            "; ".join(artifact.reasons) if artifact.reasons else None,
        )
        if str(value or "").strip()
    ]
    latency_values = [
        output.latency_ms
        for output in (mineru_output, qwen_output)
        if output is not None and output.latency_ms is not None
    ]

    return {
        "image_id": image_task.image_id,
        "model_name": (
            selected_output.model_name
            if selected_output is not None and str(selected_output.model_name or "").strip()
            else str(
                artifact.final_document.raw_metadata.get("selected_model_name", "")
                or artifact.final_document.source
                or "adjudicated"
            ).strip()
        ),
        "success": success,
        "raw_text": json.dumps(parsed, ensure_ascii=False),
        "parsed": parsed,
        "error": None if success else (errors[0] if errors else "no_final_blocks"),
        "latency_ms": sum(latency_values) if latency_values else None,
        "vendor": (
            selected_output.vendor
            if selected_output is not None and str(selected_output.vendor or "").strip()
            else str(
                artifact.final_document.raw_metadata.get("selected_vendor", "")
                or "adjudicated"
            ).strip()
        ),
        "source_type": (
            selected_output.source_type
            if selected_output is not None
            and str(selected_output.source_type or "").strip()
            else str(
                artifact.final_document.raw_metadata.get("selected_source_type", "")
                or "final"
            ).strip()
        ),
    }


def _pick_selected_final_output(
    final_document: CanonicalDocument,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
) -> ModelOutput | None:
    selected_role = str(
        final_document.raw_metadata.get("selected_output_role", "") or ""
    ).strip()
    if selected_role == "qwen":
        return qwen_output
    if selected_role == "mineru":
        return mineru_output
    if selected_role:
        return None
    return mineru_output or qwen_output


def build_final_parsed_payload(
    image_task: ImageTask,
    document: CanonicalDocument,
    markdown_pages: list[str],
) -> dict[str, Any]:
    pages = build_extraction_results(
        document=document,
        file_name=image_task.file_name,
        markdown_pages=markdown_pages,
    )
    return {
        "filename": image_task.file_name,
        "total_pages": len(pages),
        "extraction_results": pages,
    }


def build_extraction_results(
    document: CanonicalDocument,
    file_name: str,
    markdown_pages: list[str],
) -> list[dict[str, Any]]:
    page_count = max(_page_count(document), len(markdown_pages), 1)
    pages: list[dict[str, Any]] = []
    for page_idx in range(page_count):
        page_blocks = [
            _canonical_block_to_extraction_item(block)
            for block in sorted(
                document.blocks,
                key=lambda item: (item.page_idx, item.order_index, item.block_id),
            )
            if block.page_idx == page_idx
        ]
        pages.append(
            {
                "page": page_idx,
                "file_name": file_name,
                "md_res": markdown_pages[page_idx] if page_idx < len(markdown_pages) else "",
                "json_res": page_blocks,
            }
        )
    return pages


def build_final_markdown_output(
    image_task: ImageTask,
    final_document: CanonicalDocument,
    mineru_document: CanonicalDocument,
) -> tuple[list[str], str]:
    page_count = max(_page_count(mineru_document), 1)
    final_special_blocks = _special_blocks_by_page(final_document, page_count=page_count)
    mineru_blocks = _ordered_blocks_by_page(mineru_document, page_count=page_count)
    page_markdown: list[str] = []

    for page_idx in range(page_count):
        pending_special_blocks = list(final_special_blocks[page_idx])
        rendered_fragments: list[str] = []

        for mineru_block in mineru_blocks[page_idx]:
            special_block = _pop_matching_special_block(
                mineru_block=mineru_block,
                pending_special_blocks=pending_special_blocks,
            )
            source_block = special_block if special_block is not None else mineru_block
            fragment = _render_markdown_fragment(
                block=source_block,
                default_image_path=image_task.image_path,
            )
            if fragment:
                rendered_fragments.append(fragment)

        page_markdown.append("\n\n".join(rendered_fragments).strip())

    markdown_text = "\n\n".join(
        fragment for fragment in page_markdown if str(fragment or "").strip()
    ).strip()
    return page_markdown, markdown_text


def build_content_list_v2(document: CanonicalDocument) -> list[list[dict[str, Any]]]:
    page_count = _page_count(document)
    pages: list[list[dict[str, Any]]] = [[] for _ in range(page_count)]
    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        pages[block.page_idx].append(_canonical_block_to_v2_item(block))
    return pages


def build_content_list(document: CanonicalDocument) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        items.append(_canonical_block_to_flat_item(block))
    return items


def build_summary_record(
    image_task: ImageTask,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    artifact: AdjudicationArtifact,
    content_list_v2: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    consensus = artifact.consensus
    final_types = sorted({block.type for block in artifact.final_document.blocks})
    return {
        "image_id": image_task.image_id,
        "file_name": image_task.file_name,
        "decision": consensus.decision if consensus is not None else "review",
        "matched_block_count": artifact.matched_block_count,
        "added_qwen_block_count": artifact.added_qwen_block_count,
        "final_block_count": sum(len(page) for page in content_list_v2),
        "final_types": final_types,
        "review_required": artifact.review_required,
        "reasons": artifact.reasons,
        "graph_fusion_status": (artifact.graph_fusion or {}).get(
            "fusion_status", "none"
        ),
        "graph_confidence": (artifact.graph_fusion or {}).get("graph_confidence", 0.0),
        "mineru_success": bool(mineru_output.success)
        if mineru_output is not None
        else False,
        "qwen_success": bool(qwen_output.success) if qwen_output is not None else False,
    }


def build_stage2_judge_payload(
    image_task: ImageTask,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt_tokens += _coerce_int(usage.get("prompt_tokens"))
        completion_tokens += _coerce_int(usage.get("completion_tokens"))
        total_tokens += _coerce_int(usage.get("total_tokens"))

    return {
        "image_id": image_task.image_id,
        "file_name": image_task.file_name,
        "record_count": len(records),
        "totals": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "records": records,
    }


def _output_record(output: ModelOutput | None) -> dict[str, Any]:
    if output is None:
        return {"success": False, "error": "client_not_configured"}
    return output.model_dump()


def _write_extra_stage1_result(
    directories: dict[str, Path],
    image_id: str,
    role: str,
    payload: dict[str, Any],
) -> None:
    output = payload.get("output")
    document = payload.get("document")
    label = payload.get("label")
    _write_json(
        directories[f"raw_{role}"] / f"{image_id}.json",
        _output_record(output if isinstance(output, ModelOutput) else None),
    )
    _write_json(
        directories[f"normalized_{role}"] / f"{image_id}.json",
        _normalized_output_payload(
            document=document if isinstance(document, CanonicalDocument) else None,
            label=label if isinstance(label, ParsedLabel) else None,
        ),
    )


def _normalized_output_payload(
    document: CanonicalDocument | None,
    label: ParsedLabel | None,
) -> dict[str, Any]:
    projected_document = project_document_for_single_block_view(
        document=document, label=label
    )
    return {
        "document": projected_document.model_dump()
        if isinstance(projected_document, CanonicalDocument)
        else None,
        "derived_label": label.model_dump() if isinstance(label, ParsedLabel) else None,
    }


def _page_count(document: CanonicalDocument) -> int:
    return max(
        document.page_count,
        max((block.page_idx for block in document.blocks), default=-1) + 1,
        1,
    )


def _ordered_blocks_by_page(
    document: CanonicalDocument,
    page_count: int | None = None,
) -> list[list[CanonicalBlock]]:
    resolved_page_count = page_count if page_count is not None else _page_count(document)
    pages: list[list[CanonicalBlock]] = [[] for _ in range(max(resolved_page_count, 1))]
    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        if block.page_idx < 0:
            continue
        while block.page_idx >= len(pages):
            pages.append([])
        pages[block.page_idx].append(block)
    return pages


def _special_blocks_by_page(
    document: CanonicalDocument,
    page_count: int,
) -> list[list[CanonicalBlock]]:
    pages = _ordered_blocks_by_page(document, page_count=page_count)
    return [
        [block for block in page_blocks if _markdown_special_kind(block) is not None]
        for page_blocks in pages
    ]


def _markdown_special_kind(block: CanonicalBlock) -> str | None:
    sub_type = str(block.sub_type or "").strip().lower()
    if (
        sub_type == "flowchart"
        or block.structured_label.kind == "mermaid"
        or block.flowchart_graph is not None
    ):
        return "mermaid"
    if block.type == "image" and (
        sub_type == "seal"
        or any(str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions)
    ):
        return "seal"
    if block.type in {"table", "chart"}:
        return "tablechart"
    return None


def _pop_matching_special_block(
    mineru_block: CanonicalBlock,
    pending_special_blocks: list[CanonicalBlock],
) -> CanonicalBlock | None:
    if _markdown_special_kind(mineru_block) is None:
        return None
    for matcher in (
        _same_block_id,
        _same_special_order,
        _same_special_bbox,
        _same_special_image_path,
    ):
        for index, candidate in enumerate(pending_special_blocks):
            if matcher(mineru_block, candidate):
                return pending_special_blocks.pop(index)
    return None


def _same_block_id(left: CanonicalBlock, right: CanonicalBlock) -> bool:
    left_id = str(left.block_id or "").strip()
    right_id = str(right.block_id or "").strip()
    return bool(left_id and right_id and left_id == right_id)


def _same_special_order(left: CanonicalBlock, right: CanonicalBlock) -> bool:
    return (
        _markdown_special_kind(left) == _markdown_special_kind(right)
        and left.order_index == right.order_index
    )


def _same_special_bbox(left: CanonicalBlock, right: CanonicalBlock) -> bool:
    return (
        _markdown_special_kind(left) == _markdown_special_kind(right)
        and bool(left.bbox)
        and left.bbox == right.bbox
    )


def _same_special_image_path(left: CanonicalBlock, right: CanonicalBlock) -> bool:
    return (
        _markdown_special_kind(left) == _markdown_special_kind(right)
        and _block_image_path(left)
        and _block_image_path(left) == _block_image_path(right)
    )


def _render_markdown_fragment(
    block: CanonicalBlock,
    default_image_path: str,
) -> str:
    special_kind = _markdown_special_kind(block)
    if special_kind is not None:
        return _render_special_markdown_fragment(
            block=block,
            language=special_kind,
            default_image_path=default_image_path,
        )
    return _render_plain_markdown_fragment(block)


def _render_special_markdown_fragment(
    block: CanonicalBlock,
    language: str,
    default_image_path: str,
) -> str:
    image_path = _block_image_path(block) or str(default_image_path or "").strip()
    fenced_content = _special_block_content(block, language=language)
    return "\n".join(
        [
            f"![Figure](<img src={image_path}>)",
            f"```{language}",
            fenced_content,
            "```",
        ]
    ).strip()


def _special_block_content(block: CanonicalBlock, language: str) -> str:
    if language == "mermaid":
        return _preferred_flowchart_content(block)
    if language == "seal":
        return _preferred_seal_content(block)
    if block.type == "table":
        return str(block.content.get("table_body", "") or block.structured_label.content or block.text).strip()
    return str(block.content.get("content", "") or block.structured_label.content or block.text).strip()


def _preferred_seal_content(block: CanonicalBlock) -> str:
    seal_texts = [
        str(region.text or "").strip()
        for region in block.ocr_regions
        if str(region.role or "").strip().lower() == "seal" and str(region.text or "").strip()
    ]
    if seal_texts:
        return "\n".join(seal_texts)
    if str(block.text or "").strip():
        return str(block.text or "").strip()
    captions = block.content.get("image_caption")
    if isinstance(captions, list) and captions:
        return str(captions[0] or "").strip()
    return ""


def _render_plain_markdown_fragment(block: CanonicalBlock) -> str:
    text = _plain_markdown_text(block)
    if not text:
        return ""
    if block.type == "title":
        level = block.text_level or _coerce_int(block.content.get("level")) or 1
        level = min(max(level, 1), 6)
        return f"{'#' * level} {text}".strip()
    if block.type == "list":
        items = [
            str(item).strip()
            for item in block.content.get("list_items", [])
            if str(item).strip()
        ]
        if items:
            return "\n".join(f"- {item}" for item in items)
    if block.type in {"code", "algorithm"}:
        return f"```\n{text}\n```"
    if block.type == "equation_interline":
        return f"$$\n{text}\n$$"
    return text


def _plain_markdown_text(block: CanonicalBlock) -> str:
    if block.type == "title":
        return _join_span_content(block.content.get("title_content")) or str(block.text or "").strip()
    if block.type == "paragraph":
        return _join_span_content(block.content.get("paragraph_content")) or str(block.text or "").strip()
    if block.type == "list":
        items = [
            str(item).strip()
            for item in block.content.get("list_items", [])
            if str(item).strip()
        ]
        return "\n".join(items)
    if block.type == "equation_interline":
        return str(block.content.get("math_content", "") or block.text).strip()
    if block.type == "image":
        captions = block.content.get("image_caption")
        if isinstance(captions, list) and captions:
            return str(captions[0] or "").strip()
        return str(block.text or "").strip()
    if block.type == "table":
        return str(block.content.get("table_body", "") or block.text).strip()
    if block.type == "chart":
        return str(block.content.get("content", "") or block.text).strip()
    return str(block.text or "").strip()


def _join_span_content(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("content", "") or "").strip()
        else:
            text = str(item or "").strip()
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _block_image_path(block: CanonicalBlock) -> str:
    return str(block.content.get("img_path", "") or "").strip()


def _canonical_block_to_v2_item(block: CanonicalBlock) -> dict[str, Any]:
    item = {
        "block_id": block.block_id,
        "type": block.type,
        "bbox": block.bbox,
        "content": _content_for_v2(block),
    }
    if block.sub_type:
        item["sub_type"] = block.sub_type
    if block.text_level is not None:
        item["text_level"] = block.text_level
    if block.flowchart_graph is not None:
        item["flowchart_graph"] = block.flowchart_graph
    if block.ocr_regions:
        item["ocr_regions"] = [region.model_dump() for region in block.ocr_regions]
    return item


def _canonical_block_to_flat_item(block: CanonicalBlock) -> dict[str, Any]:
    if block.type in {"title", "paragraph"}:
        item = {
            "type": "text",
            "page_idx": block.page_idx,
            "bbox": block.bbox,
            "text": block.text,
        }
        if block.type == "title":
            item["text_level"] = block.text_level or 1
        return item

    item = {
        "type": block.type,
        "page_idx": block.page_idx,
        "bbox": block.bbox,
        "sub_type": block.sub_type,
    }
    content = _content_for_flat(block)
    item.update(content)
    return {key: value for key, value in item.items() if value not in (None, [], "")}


def _canonical_block_to_extraction_item(block: CanonicalBlock) -> dict[str, Any]:
    item = {
        "type": _type_for_extraction(block),
        "bbox": block.bbox,
        "angle": _angle_for_extraction(block),
        "content": _content_for_extraction(block),
    }
    if block.sub_type:
        item["sub_type"] = block.sub_type
    if block.flowchart_graph is not None:
        item["flowchart_graph"] = block.flowchart_graph
    if block.type == "title" and block.text_level is not None:
        item["text_level"] = block.text_level
    return item


def _content_for_v2(block: CanonicalBlock) -> dict[str, Any]:
    if block.content:
        content = json.loads(json.dumps(block.content, ensure_ascii=False))
    else:
        content = {}

    if str(block.sub_type or "").strip().lower() == "flowchart":
        content.setdefault("content", _preferred_flowchart_content(block))
        content.setdefault(
            "chart_caption",
            _single_text_list(block.caption_structured.brief or block.text),
        )
        content.setdefault("chart_footnote", [])
        content.setdefault("img_path", content.get("img_path", ""))
        return content

    if block.type == "title":
        content.setdefault("title_content", [{"type": "text", "content": block.text}])
        content.setdefault("level", block.text_level or 1)
    elif block.type == "paragraph":
        content.setdefault(
            "paragraph_content", [{"type": "text", "content": block.text}]
        )
    elif block.type == "table":
        content.setdefault(
            "table_body",
            block.structured_label.content
            if block.structured_label.kind == "table"
            else block.text,
        )
        content.setdefault(
            "table_caption",
            _single_text_list(block.caption_structured.brief or block.text),
        )
        content.setdefault("img_path", "")
    elif block.type == "chart":
        content.setdefault("content", block.structured_label.content or block.text)
        content.setdefault(
            "chart_caption",
            _single_text_list(block.caption_structured.brief or block.text),
        )
        content.setdefault("chart_footnote", [])
        content.setdefault("img_path", "")
    elif block.type == "image":
        content.setdefault(
            "image_caption",
            _single_text_list(block.caption_structured.brief or block.text),
        )
        content.setdefault("image_footnote", [])
        content.setdefault("img_path", "")
    elif block.type == "equation_interline":
        content.setdefault("math_content", block.text)
        content.setdefault("math_type", "latex")
    elif block.type == "list":
        content.setdefault("list_items", _single_text_list(block.text))

    return content


def _content_for_flat(block: CanonicalBlock) -> dict[str, Any]:
    content = _content_for_v2(block)
    if block.type == "table":
        return {
            "img_path": content.get("img_path", ""),
            "table_caption": content.get("table_caption", []),
            "table_footnote": content.get("table_footnote", []),
            "table_body": content.get("table_body", ""),
        }
    if block.type == "chart":
        return {
            "img_path": content.get("img_path", ""),
            "sub_type": block.sub_type,
            "chart_caption": content.get("chart_caption", []),
            "chart_footnote": content.get("chart_footnote", []),
            "content": content.get("content", ""),
        }
    if block.type == "image":
        return {
            "img_path": content.get("img_path", ""),
            "sub_type": block.sub_type,
            "image_caption": content.get("image_caption", []),
            "image_footnote": content.get("image_footnote", []),
        }
    if block.type == "equation_interline":
        return {
            "text": content.get("math_content", block.text),
            "text_format": content.get("math_type", "latex"),
        }
    if block.type == "list":
        return {"text": "\n".join(content.get("list_items", []))}
    return content


def _type_for_extraction(block: CanonicalBlock) -> str:
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return "chart"
    source_type = str(block.provenance.get("source_block_type", "") or "").strip()
    if source_type:
        return source_type
    if block.type == "paragraph":
        return "text"
    return block.type


def _angle_for_extraction(block: CanonicalBlock) -> int:
    angle = block.provenance.get("source_angle")
    try:
        return int(round(float(angle)))
    except (TypeError, ValueError):
        return 0


def _content_for_extraction(block: CanonicalBlock) -> str:
    content = _content_for_v2(block)
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return _preferred_flowchart_content(block)
    if block.type in {"title", "paragraph"}:
        return block.text
    if block.type == "table":
        return str(content.get("table_body", "") or "").strip()
    if block.type == "chart":
        return str(content.get("content", "") or block.text).strip()
    if block.type == "image":
        captions = content.get("image_caption", [])
        if isinstance(captions, list) and captions:
            return str(captions[0] or "").strip()
        return block.text
    if block.type == "equation_interline":
        return str(content.get("math_content", "") or block.text).strip()
    if block.type == "list":
        items = content.get("list_items", [])
        return "\n".join(str(item).strip() for item in items if str(item).strip())
    return block.text


def _single_text_list(text: str) -> list[str]:
    normalized = str(text or "").strip()
    return [normalized] if normalized else []


def _preferred_flowchart_content(block: CanonicalBlock) -> str:
    structured_mermaid = str(block.structured_label.content or "").strip()
    if block.structured_label.kind == "mermaid" and structured_mermaid:
        return structured_mermaid
    direct_content = str(block.content.get("content", "") or "").strip()
    if direct_content:
        return direct_content
    if str(block.text or "").strip():
        return str(block.text or "").strip()
    image_captions = block.content.get("image_caption")
    if isinstance(image_captions, list) and image_captions:
        return str(image_captions[0] or "").strip()
    chart_captions = block.content.get("chart_caption")
    if isinstance(chart_captions, list) and chart_captions:
        return str(chart_captions[0] or "").strip()
    return ""


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _remove_legacy_final_files(final_dir: Path, image_id: str) -> None:
    for suffix in ("_content_list_v2.json", "_content_list.json"):
        legacy_path = final_dir / f"{image_id}{suffix}"
        if legacy_path.exists():
            legacy_path.unlink()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.write_text(str(payload or ""), encoding="utf-8")


def write_page_merged_markdown(output_dir: Path, image_tasks: list[ImageTask]) -> list[Path]:
    directories = ensure_output_dirs(output_dir)
    grouped_tasks: dict[str, list[ImageTask]] = {}
    for image_task in image_tasks:
        if not image_task.is_page_crop or not str(image_task.page_output_id or "").strip():
            continue
        grouped_tasks.setdefault(image_task.page_output_id, []).append(image_task)

    written_paths: list[Path] = []
    for grouped_image_tasks in grouped_tasks.values():
        merged_markdown_path = write_page_merged_markdown_for_page(
            output_dir=output_dir,
            image_tasks=grouped_image_tasks,
            directories=directories,
        )
        if merged_markdown_path is not None:
            written_paths.append(merged_markdown_path)

    return written_paths


def write_page_merged_markdown_for_page(
    output_dir: Path,
    image_tasks: list[ImageTask],
    directories: dict[str, Path] | None = None,
) -> Path | None:
    eligible_tasks = [
        image_task
        for image_task in image_tasks
        if image_task.is_page_crop and str(image_task.page_output_id or "").strip()
    ]
    if not eligible_tasks:
        return None

    page_output_ids = {
        str(image_task.page_output_id or "").strip() for image_task in eligible_tasks
    }
    if len(page_output_ids) != 1:
        raise ValueError(
            "write_page_merged_markdown_for_page expects image tasks from a single page_output_id"
        )

    resolved_directories = directories or ensure_output_dirs(output_dir)
    final_dir = resolved_directories["final"]
    page_md_dir = resolved_directories["page_md"]
    fragments: list[str] = []
    for image_task in sorted(eligible_tasks, key=_page_crop_sort_key):
        crop_markdown_path = final_dir / f"{image_task.image_id}.md"
        if not crop_markdown_path.exists():
            continue
        fragment = crop_markdown_path.read_text(encoding="utf-8").strip()
        if fragment:
            fragments.append(fragment)

    merged_markdown_path = page_md_dir / f"{next(iter(page_output_ids))}.md"
    _write_text(merged_markdown_path, "\n\n".join(fragments).strip())
    return merged_markdown_path


def _page_crop_sort_key(image_task: ImageTask) -> tuple[int, int, str, str]:
    merge_order = str(image_task.merge_order or "").strip()
    if merge_order.lstrip("-").isdigit():
        return (0, int(merge_order), "", image_task.image_id)
    return (1, 0, merge_order, image_task.image_id)
