from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.schema import (
    AdjudicationArtifact,
    CanonicalBlock,
    CanonicalDocument,
    ImageTask,
    ModelOutput,
    ParsedLabel,
    StructuredLabel,
)

_SINGLE_BLOCK_VIEW_IMAGE_TYPES = {
    "seal",
    "natural_image",
    "document",
    "screenshot",
    "diagram",
    "mixed",
    "unknown",
}


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
    final_output = build_final_output(
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        artifact=artifact,
    )
    content_list_v2 = build_content_list_v2(final_document)
    _remove_legacy_final_files(directories["final"], image_task.image_id)
    _write_json(directories["final"] / f"{image_task.image_id}.json", final_output)
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
) -> dict[str, Any]:
    parsed = build_final_parsed_payload(
        image_task=image_task, document=artifact.final_document
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
    return mineru_output or qwen_output


def build_final_parsed_payload(
    image_task: ImageTask, document: CanonicalDocument
) -> dict[str, Any]:
    pages = build_extraction_results(document=document, file_name=image_task.file_name)
    return {
        "filename": image_task.file_name,
        "total_pages": len(pages),
        "extraction_results": pages,
    }


def build_extraction_results(
    document: CanonicalDocument, file_name: str
) -> list[dict[str, Any]]:
    page_count = max(
        document.page_count,
        max((block.page_idx for block in document.blocks), default=-1) + 1,
        1,
    )
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
                "md_res": "",
                "json_res": page_blocks,
            }
        )
    return pages


def build_content_list_v2(document: CanonicalDocument) -> list[list[dict[str, Any]]]:
    page_count = max(
        document.page_count,
        max((block.page_idx for block in document.blocks), default=-1) + 1,
        1,
    )
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
    projected_document = _project_normalized_view_document(document=document, label=label)
    return {
        "document": projected_document.model_dump()
        if isinstance(projected_document, CanonicalDocument)
        else None,
        "derived_label": label.model_dump() if isinstance(label, ParsedLabel) else None,
    }


def _project_normalized_view_document(
    document: CanonicalDocument | None,
    label: ParsedLabel | None,
) -> CanonicalDocument | None:
    if not isinstance(document, CanonicalDocument):
        return None
    if not isinstance(label, ParsedLabel):
        return document

    image_type = str(label.image_type or "").strip().lower()
    if image_type not in _SINGLE_BLOCK_VIEW_IMAGE_TYPES or len(document.blocks) <= 1:
        return document

    ordered_blocks = sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    )
    if not ordered_blocks:
        return document

    merged_text_parts: list[str] = []
    merged_ocr_regions = []
    merged_warnings: list[str] = list(document.warnings)
    for block in ordered_blocks:
        merged_text_parts.extend(_text_fragments_for_single_block_view(block))
        merged_ocr_regions.extend(block.ocr_regions)
        merged_warnings.extend(block.warnings)

    merged_text = "\n\n".join(part for part in merged_text_parts if part.strip())
    representative = _pick_single_block_representative(
        blocks=ordered_blocks, image_type=image_type
    )
    img_path = _first_image_path(ordered_blocks)
    bbox = _merge_bboxes([block.bbox for block in ordered_blocks]) or representative.bbox
    sub_type = image_type if image_type else representative.sub_type
    caption = str(label.caption or representative.caption_structured.brief or "").strip()

    content: dict[str, Any] = {}
    if img_path:
        content["img_path"] = img_path
    if merged_text:
        content["image_caption"] = [merged_text]

    caption_structured = label.caption_structured.model_copy(deep=True)
    if not caption_structured.visual_type:
        caption_structured.visual_type = image_type
    if not caption_structured.brief:
        caption_structured.brief = caption
    if not caption_structured.main_subject:
        caption_structured.main_subject = caption_structured.brief
    if not caption_structured.key_visible_text:
        caption_structured.key_visible_text = merged_text_parts[:10]

    projected_block = CanonicalBlock(
        block_id=representative.block_id,
        page_idx=representative.page_idx,
        order_index=1,
        type="image",
        sub_type=sub_type,
        bbox=bbox,
        text=merged_text,
        text_level=None,
        content=content,
        source=representative.source,
        confidence=representative.confidence,
        structured_label=StructuredLabel(),
        caption_structured=caption_structured,
        flowchart_graph=None,
        visible_text=merged_text_parts,
        ocr_regions=merged_ocr_regions,
        warnings=_ordered_unique_strings(merged_warnings),
        provenance={
            **representative.provenance,
            "normalized_view": "single_block_projection",
            "source_block_count": len(ordered_blocks),
        },
    )
    return CanonicalDocument(
        document_id=document.document_id,
        source=document.source,
        backend=document.backend,
        page_count=max(1, document.page_count),
        blocks=[projected_block],
        warnings=_ordered_unique_strings(merged_warnings),
        raw_metadata={
            **document.raw_metadata,
            "normalized_view": "single_block_projection",
            "source_block_count": len(ordered_blocks),
        },
    )


def _pick_single_block_representative(
    blocks: list[CanonicalBlock],
    image_type: str,
) -> CanonicalBlock:
    if image_type == "seal":
        for block in blocks:
            if str(block.sub_type or "").strip().lower() == "seal":
                return block
    for block in blocks:
        if block.type == "image":
            return block
    return blocks[0]


def _text_fragments_for_single_block_view(block: CanonicalBlock) -> list[str]:
    primary = str(block.text or "").strip()
    if primary:
        return [primary]

    fragments: list[str] = []
    for key in (
        "content",
        "table_body",
    ):
        value = block.content.get(key)
        if isinstance(value, str) and value.strip():
            fragments.append(value.strip())
    for key in (
        "image_caption",
        "chart_caption",
        "table_caption",
        "list_items",
    ):
        value = block.content.get(key)
        if isinstance(value, list):
            fragments.extend(str(item).strip() for item in value if str(item).strip())
    for region in block.ocr_regions:
        text = str(region.text or "").strip()
        if text:
            fragments.append(text)
    return fragments


def _first_image_path(blocks: list[CanonicalBlock]) -> str:
    for block in blocks:
        value = block.content.get("img_path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _merge_bboxes(bboxes: list[list[int]]) -> list[int]:
    valid = [bbox for bbox in bboxes if isinstance(bbox, list) and len(bbox) == 4]
    if not valid:
        return []
    return [
        min(int(bbox[0]) for bbox in valid),
        min(int(bbox[1]) for bbox in valid),
        max(int(bbox[2]) for bbox in valid),
        max(int(bbox[3]) for bbox in valid),
    ]


def _ordered_unique_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


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
