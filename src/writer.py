from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.schema import AdjudicationArtifact, CanonicalBlock, CanonicalDocument, ImageTask, ModelOutput, ParsedLabel


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    directories = {
        "root": output_dir,
        "raw_mineru": output_dir / "raw" / "mineru",
        "raw_qwen": output_dir / "raw" / "qwen",
        "normalized_mineru": output_dir / "normalized" / "mineru",
        "normalized_qwen": output_dir / "normalized" / "qwen",
        "final": output_dir / "final",
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
        {
            "document": mineru_document.model_dump(),
            "derived_label": mineru_label.model_dump() if mineru_label is not None else None,
        },
    )
    _write_json(
        directories["normalized_qwen"] / f"{image_task.image_id}.json",
        {
            "document": qwen_document.model_dump(),
            "derived_label": qwen_label.model_dump() if qwen_label is not None else None,
        },
    )

    final_document = artifact.final_document
    content_list_v2 = build_content_list_v2(final_document)
    content_list = build_content_list(final_document)

    _write_json(directories["final"] / f"{image_task.image_id}_content_list_v2.json", content_list_v2)
    _write_json(directories["final"] / f"{image_task.image_id}_content_list.json", content_list)
    _write_json(directories["final"] / f"{image_task.image_id}_artifact.json", artifact.model_dump())

    return build_summary_record(
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        artifact=artifact,
        content_list_v2=content_list_v2,
    )


def build_content_list_v2(document: CanonicalDocument) -> list[list[dict[str, Any]]]:
    page_count = max(document.page_count, max((block.page_idx for block in document.blocks), default=-1) + 1, 1)
    pages: list[list[dict[str, Any]]] = [[] for _ in range(page_count)]
    for block in sorted(document.blocks, key=lambda item: (item.page_idx, item.order_index, item.block_id)):
        pages[block.page_idx].append(_canonical_block_to_v2_item(block))
    return pages


def build_content_list(document: CanonicalDocument) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in sorted(document.blocks, key=lambda item: (item.page_idx, item.order_index, item.block_id)):
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
        "graph_fusion_status": (artifact.graph_fusion or {}).get("fusion_status", "none"),
        "graph_confidence": (artifact.graph_fusion or {}).get("graph_confidence", 0.0),
        "mineru_success": bool(mineru_output.success) if mineru_output is not None else False,
        "qwen_success": bool(qwen_output.success) if qwen_output is not None else False,
    }


def _output_record(output: ModelOutput | None) -> dict[str, Any]:
    if output is None:
        return {"success": False, "error": "client_not_configured"}
    return output.model_dump()


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


def _content_for_v2(block: CanonicalBlock) -> dict[str, Any]:
    if block.content:
        content = json.loads(json.dumps(block.content, ensure_ascii=False))
    else:
        content = {}

    if block.type == "title":
        content.setdefault("title_content", [{"type": "text", "content": block.text}])
        content.setdefault("level", block.text_level or 1)
    elif block.type == "paragraph":
        content.setdefault("paragraph_content", [{"type": "text", "content": block.text}])
    elif block.type == "table":
        content.setdefault("table_body", block.structured_label.content if block.structured_label.kind == "table" else block.text)
        content.setdefault("table_caption", _single_text_list(block.caption_structured.brief or block.text))
        content.setdefault("img_path", "")
    elif block.type == "chart":
        content.setdefault("content", block.structured_label.content or block.text)
        content.setdefault("chart_caption", _single_text_list(block.caption_structured.brief or block.text))
        content.setdefault("chart_footnote", [])
        content.setdefault("img_path", "")
    elif block.type == "image":
        content.setdefault("image_caption", _single_text_list(block.caption_structured.brief or block.text))
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


def _single_text_list(text: str) -> list[str]:
    normalized = str(text or "").strip()
    return [normalized] if normalized else []


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
