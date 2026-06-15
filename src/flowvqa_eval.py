from __future__ import annotations

import json
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Any

from eval_dataset.mermaid_td_f1.evaluator import evaluate_mermaid_flowchart
from src.pipeline.flowchart_utils import (
    flowchart_graph_from_mermaid,
    looks_like_mermaid,
    mermaid_from_flowchart_graph,
    normalize_mermaid_text,
)
from src.schema import CanonicalBlock, CanonicalDocument, ParsedLabel

_METRIC_KEYS = (
    "final_td_f1",
    "structure_f1",
    "semantic_f1",
    "node_text_f1",
    "edge_text_f1",
    "binding_f1",
    "penalty",
)


def validate_flowvqa_root(flowvqa_root: Path) -> Path:
    resolved_root = flowvqa_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(f"FlowVQA root not found: {resolved_root}")
    data_dir = resolved_root / "Data"
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"FlowVQA Data directory not found: {data_dir}")
    return resolved_root


@lru_cache(maxsize=4)
def _load_flowvqa_index(root_text: str) -> dict[str, dict[str, Any]]:
    flowvqa_root = Path(root_text)
    index: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "test"):
        split_path = flowvqa_root / "Data" / f"{split_name}_full.json"
        if not split_path.exists():
            continue
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected FlowVQA payload type in {split_path}")

        for sample_id, sample_payload in payload.items():
            if not isinstance(sample_payload, dict):
                continue
            mermaid = str(sample_payload.get("mermaid", "") or "").strip()
            if not mermaid:
                continue
            qa_payload = sample_payload.get("qa")
            question_count = len(qa_payload) if isinstance(qa_payload, dict) else 0
            index[str(sample_id)] = {
                "dataset": "flowvqa",
                "sample_id": str(sample_id),
                "split": split_name,
                "ground_truth_mermaid": mermaid,
                "question_count": question_count,
                "source_path": f"Data/{split_name}_full.json",
            }
    return index


def find_flowvqa_reference(
    flowvqa_root: Path | None,
    image_id: str,
) -> dict[str, Any] | None:
    if flowvqa_root is None:
        return None
    normalized_id = str(image_id or "").strip()
    if not normalized_id:
        return None
    return _load_flowvqa_index(str(flowvqa_root)).get(normalized_id)


def extract_mermaid_from_document(
    document: CanonicalDocument | None,
    label: ParsedLabel | None = None,
) -> str:
    label_mermaid = extract_mermaid_from_label(label)
    if label_mermaid:
        return label_mermaid
    if not isinstance(document, CanonicalDocument):
        return ""

    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        block_mermaid = _extract_mermaid_from_block(block)
        if block_mermaid:
            return block_mermaid
    return ""


def extract_mermaid_from_label(label: ParsedLabel | None) -> str:
    if not isinstance(label, ParsedLabel):
        return ""

    structured_content = _normalize_candidate_text(label.structured_label.content)
    if str(label.structured_label.kind or "").strip().lower() == "mermaid" and structured_content:
        return structured_content

    derived_mermaid = _derive_mermaid_from_graph(label.flowchart_graph)
    if derived_mermaid:
        return derived_mermaid

    return _normalize_candidate_text(label.caption)


def build_flowvqa_eval_payload(
    reference: dict[str, Any] | None,
    predictions_by_source: dict[str, str],
    source_meta_by_source: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(reference, dict):
        return None

    ground_truth_mermaid = str(reference.get("ground_truth_mermaid", "") or "").strip()
    normalized_ground_truth = _normalize_candidate_text(ground_truth_mermaid)
    if not normalized_ground_truth:
        return None

    metrics_by_source: dict[str, dict[str, Any]] = {}
    mermaid_by_source: dict[str, dict[str, Any]] = {}
    for source_name, prediction in predictions_by_source.items():
        normalized_prediction = normalize_mermaid_text(str(prediction or ""))
        result = evaluate_mermaid_flowchart(
            pred_mermaid=normalized_prediction,
            gold_mermaid=normalized_ground_truth,
        )
        normalized_source_name = str(source_name or "").strip()
        metrics_by_source[normalized_source_name] = _summarize_result(result)
        mermaid_by_source[normalized_source_name] = _build_source_mermaid_payload(
            source_name=normalized_source_name,
            mermaid=str(prediction or ""),
            source_meta=(source_meta_by_source or {}).get(normalized_source_name),
        )

    return {
        "dataset": "flowvqa",
        "sample_id": str(reference.get("sample_id", "") or ""),
        "split": str(reference.get("split", "") or ""),
        "source_path": str(reference.get("source_path", "") or ""),
        "question_count": int(reference.get("question_count", 0) or 0),
        "ground_truth_mermaid": ground_truth_mermaid,
        "ground_truth_render_code": build_mermaid_render_code(normalized_ground_truth),
        "metrics_by_source": metrics_by_source,
        "mermaid_by_source": mermaid_by_source,
    }


def build_mermaid_render_code(mermaid: str) -> str:
    normalized_mermaid = normalize_mermaid_text(str(mermaid or ""))
    if not normalized_mermaid:
        return ""

    if not _needs_render_safe_rebuild(normalized_mermaid):
        return normalized_mermaid

    graph_payload = flowchart_graph_from_mermaid(normalized_mermaid)
    if isinstance(graph_payload, dict):
        rebuilt_mermaid = normalize_mermaid_text(
            mermaid_from_flowchart_graph(graph_payload)
        )
        if rebuilt_mermaid:
            return _sanitize_render_mermaid(rebuilt_mermaid)

    return _sanitize_render_mermaid(normalized_mermaid)


def _extract_mermaid_from_block(block: CanonicalBlock) -> str:
    if not _is_flowchart_like_block(block):
        return ""

    for candidate in _candidate_texts_from_block(block):
        normalized_candidate = _normalize_candidate_text(candidate)
        if normalized_candidate:
            return normalized_candidate

    return _derive_mermaid_from_graph(block.flowchart_graph)


def _is_flowchart_like_block(block: CanonicalBlock) -> bool:
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if str(block.structured_label.kind or "").strip().lower() == "mermaid":
        return True
    return isinstance(block.flowchart_graph, dict)


def _candidate_texts_from_block(block: CanonicalBlock) -> list[str]:
    candidates: list[str] = []

    direct_content = str(block.content.get("content", "") or "").strip()
    if direct_content:
        candidates.append(direct_content)

    structured_content = str(block.structured_label.content or "").strip()
    if structured_content:
        candidates.append(structured_content)

    block_text = str(block.text or "").strip()
    if block_text:
        candidates.append(block_text)

    for key in ("image_caption", "chart_caption"):
        values = block.content.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            text = str(item or "").strip()
            if text:
                candidates.append(text)

    return candidates


def _derive_mermaid_from_graph(graph_payload: dict[str, Any] | None) -> str:
    if not isinstance(graph_payload, dict):
        return ""
    derived_mermaid = mermaid_from_flowchart_graph(graph_payload)
    return _normalize_candidate_text(derived_mermaid)


def _normalize_candidate_text(text: str) -> str:
    normalized_text = normalize_mermaid_text(str(text or ""))
    if not normalized_text:
        return ""
    if not looks_like_mermaid(normalized_text):
        return ""
    return normalized_text


def _sanitize_render_mermaid(mermaid: str) -> str:
    value = normalize_mermaid_text(str(mermaid or ""))
    if not value:
        return ""
    value = unescape(value)
    value = value.replace('\\"', "'").replace("\\'", "'")
    value = value.replace("\\\\", "/").replace("\\", "/")
    value = value.replace("&quot;", "'").replace("&#34;", "'")
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def _needs_render_safe_rebuild(mermaid: str) -> bool:
    text = str(mermaid or "")
    return any(token in text for token in ('\\"', "\\\\", "&quot;", "&#34;"))


def _build_source_mermaid_payload(
    source_name: str,
    mermaid: str,
    source_meta: dict[str, str] | None,
) -> dict[str, Any]:
    raw_mermaid = str(mermaid or "").strip()
    normalized_mermaid = normalize_mermaid_text(raw_mermaid)
    source_meta = source_meta if isinstance(source_meta, dict) else {}
    return {
        "title": str(source_meta.get("title", "") or _source_title(source_name)),
        "source_path": str(source_meta.get("source_path", "") or ""),
        "mermaid": raw_mermaid,
        "render_code": build_mermaid_render_code(normalized_mermaid)
        if normalized_mermaid
        else "",
    }


def _source_title(source_name: str) -> str:
    return {
        "mineru_raw": "MinerU Raw",
        "mineru": "MinerU",
        "qwen": "Qwen",
        "final": "Ours",
        "paddle": "Paddle",
        "glm": "GLM",
    }.get(str(source_name or "").strip(), str(source_name or "").strip() or "Source")


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    debug_payload = result.get("debug")
    debug_errors = (
        [
            str(item).strip()
            for item in list(debug_payload.get("errors") or [])
            if str(item).strip()
        ]
        if isinstance(debug_payload, dict)
        else []
    )
    summary = {
        "parse_valid": bool(result.get("parse_valid", False)),
        "debug_errors": debug_errors,
    }
    for key in _METRIC_KEYS:
        summary[key] = round(float(result.get(key, 0.0) or 0.0), 6)
    return summary
