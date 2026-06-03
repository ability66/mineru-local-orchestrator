from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.pipeline.alignment import align_blocks, bbox_iou
from src.pipeline.flowchart_utils import build_flowchart_candidate_patch, build_flowchart_graph_payload, looks_like_mermaid
from src.schema import CanonicalBlock, CanonicalDocument, ImageTask, Issue, ParsedLabel


def detect_seal_issues(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
) -> list[Issue]:
    del image_task
    matches = align_blocks(mineru_document.blocks, qwen_document.blocks)
    match_lookup = {match.base_index: match for match in matches}
    matched_qwen_indexes = {match.candidate_index for match in matches}

    issues: list[Issue] = []
    for mineru_index, mineru_block in enumerate(mineru_document.blocks):
        match = match_lookup.get(mineru_index)
        qwen_block = qwen_document.blocks[match.candidate_index] if match is not None else None
        issue = _detect_pair_issue(mineru_block=mineru_block, qwen_block=qwen_block)
        if issue is not None:
            issues.append(issue)

    for qwen_index, qwen_block in enumerate(qwen_document.blocks):
        if qwen_index in matched_qwen_indexes or not _is_seal_candidate(qwen_block):
            continue
        target_block = _find_best_target_block(mineru_document.blocks, qwen_block)
        issues.append(
            Issue(
                issue_id=f"seal-unmatched-{qwen_block.block_id}",
                issue_type="seal_unmatched_qwen_candidate",
                page_idx=qwen_block.page_idx,
                target_block_id=target_block.block_id if target_block is not None else None,
                mineru_block=target_block.model_dump() if target_block is not None else None,
                qwen_block=qwen_block.model_dump(),
                reasons=["qwen_detects_unmatched_seal_block"],
            )
        )

    return _deduplicate_issues(issues)


def detect_flowchart_issues(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    graph_fusion_result: Any | None,
) -> list[Issue]:
    del image_task
    if not _has_flowchart_signal(
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        graph_fusion_result=graph_fusion_result,
    ):
        return []

    matches = align_blocks(mineru_document.blocks, qwen_document.blocks)
    qwen_flowchart_block = _pick_flowchart_block(qwen_document.blocks)
    mineru_flowchart_block = _pick_flowchart_block(mineru_document.blocks)
    match_lookup = {match.candidate_index: match for match in matches}

    target_block: CanonicalBlock | None = None
    if qwen_flowchart_block is not None:
        qwen_index = next(
            (
                index
                for index, block in enumerate(qwen_document.blocks)
                if block.block_id == qwen_flowchart_block.block_id
            ),
            None,
        )
        if qwen_index is not None and qwen_index in match_lookup:
            target_block = mineru_document.blocks[match_lookup[qwen_index].base_index]
        if target_block is None:
            target_block = _find_best_flowchart_target_block(mineru_document.blocks, qwen_flowchart_block)
    if target_block is None:
        target_block = mineru_flowchart_block or _pick_visual_block(mineru_document.blocks)

    qwen_mermaid = _extract_flowchart_mermaid(qwen_document.blocks, qwen_label)
    mineru_mermaid = _extract_flowchart_mermaid(mineru_document.blocks, mineru_label)
    candidate_mermaid = str(getattr(graph_fusion_result, "mermaid", "") or "").strip()
    if not looks_like_mermaid(candidate_mermaid):
        candidate_mermaid = ""

    graph_payload = build_flowchart_graph_payload(graph_fusion_result)
    candidate_payload = {
        "candidate_mermaid": candidate_mermaid,
        "candidate_patch": build_flowchart_candidate_patch(candidate_mermaid, graph_payload),
        "graph_fusion": asdict(graph_fusion_result) if graph_fusion_result is not None else None,
        "qwen_mermaid": qwen_mermaid,
        "mineru_mermaid": mineru_mermaid,
    }

    reasons = ["flowchart_requires_second_stage_review"]
    if graph_fusion_result is None:
        reasons.append("graph_fusion_candidate_missing")
    else:
        fusion_status = str(getattr(graph_fusion_result, "fusion_status", "") or "").strip().lower()
        fusion_method = str(getattr(graph_fusion_result, "fusion_method", "") or "").strip().lower()
        graph_confidence = float(getattr(graph_fusion_result, "graph_confidence", 0.0) or 0.0)
        if not candidate_mermaid:
            reasons.append("graph_fusion_candidate_missing_mermaid")
        if fusion_method == "mermaid_fallback":
            reasons.append("graph_fusion_used_mermaid_fallback")
        if fusion_status and fusion_status != "fused":
            reasons.append(f"graph_fusion_status:{fusion_status}")
        if graph_confidence < 0.75:
            reasons.append("graph_fusion_low_confidence")

    if _has_flowchart_block_or_label(qwen_document.blocks, qwen_label) and not qwen_mermaid:
        reasons.append("qwen_flowchart_missing_mermaid")
    if _has_flowchart_block_or_label(mineru_document.blocks, mineru_label) and not mineru_mermaid:
        reasons.append("mineru_flowchart_missing_mermaid")
    if qwen_mermaid and candidate_mermaid and _normalize_text(qwen_mermaid) != _normalize_text(candidate_mermaid):
        reasons.append("qwen_candidate_mermaid_conflict")

    qwen_block_payload = qwen_flowchart_block.model_dump() if qwen_flowchart_block is not None else None
    mineru_block_payload = target_block.model_dump() if target_block is not None else None
    issue_id_target = target_block.block_id if target_block is not None else (
        qwen_flowchart_block.block_id if qwen_flowchart_block is not None else "flowchart"
    )
    page_idx = (
        target_block.page_idx
        if target_block is not None
        else (qwen_flowchart_block.page_idx if qwen_flowchart_block is not None else 0)
    )
    return [
        Issue(
            issue_id=f"flowchart-review-{issue_id_target}",
            issue_type="flowchart_candidate_review",
            page_idx=page_idx,
            target_block_id=target_block.block_id if target_block is not None else None,
            mineru_block=mineru_block_payload,
            qwen_block=qwen_block_payload,
            candidate_payload=candidate_payload,
            reasons=_deduplicate_texts(reasons),
        )
    ]


def _detect_pair_issue(mineru_block: CanonicalBlock, qwen_block: CanonicalBlock | None) -> Issue | None:
    if qwen_block is None:
        return None

    mineru_is_seal = _is_seal_candidate(mineru_block)
    qwen_is_seal = _is_seal_candidate(qwen_block)
    if not qwen_is_seal:
        return None

    mineru_texts = _seal_texts(mineru_block)
    qwen_texts = _seal_texts(qwen_block)
    if not mineru_is_seal:
        return Issue(
            issue_id=f"seal-type-{mineru_block.block_id}",
            issue_type="seal_type_disagreement",
            page_idx=mineru_block.page_idx,
            target_block_id=mineru_block.block_id,
            mineru_block=mineru_block.model_dump(),
            qwen_block=qwen_block.model_dump(),
            reasons=["qwen_marks_block_as_seal", "mineru_does_not_mark_seal"],
        )

    if not mineru_texts and qwen_texts:
        return Issue(
            issue_id=f"seal-missing-ocr-{mineru_block.block_id}",
            issue_type="seal_missing_ocr",
            page_idx=mineru_block.page_idx,
            target_block_id=mineru_block.block_id,
            mineru_block=mineru_block.model_dump(),
            qwen_block=qwen_block.model_dump(),
            reasons=["mineru_seal_without_text", "qwen_provides_seal_text"],
        )

    if mineru_texts and qwen_texts and _texts_conflict(mineru_texts, qwen_texts):
        return Issue(
            issue_id=f"seal-ocr-conflict-{mineru_block.block_id}",
            issue_type="seal_ocr_conflict",
            page_idx=mineru_block.page_idx,
            target_block_id=mineru_block.block_id,
            mineru_block=mineru_block.model_dump(),
            qwen_block=qwen_block.model_dump(),
            reasons=["mineru_and_qwen_seal_text_conflict"],
        )
    return None


def _find_best_target_block(mineru_blocks: list[CanonicalBlock], qwen_block: CanonicalBlock) -> CanonicalBlock | None:
    same_page_visuals = [
        block
        for block in mineru_blocks
        if block.page_idx == qwen_block.page_idx and block.type in {"image", "chart", "table"}
    ]
    if not same_page_visuals:
        return None
    return max(
        same_page_visuals,
        key=lambda block: (
            bbox_iou(block.bbox, qwen_block.bbox),
            1 if block.type == "image" else 0,
        ),
    )


def _find_best_flowchart_target_block(
    mineru_blocks: list[CanonicalBlock],
    qwen_block: CanonicalBlock,
) -> CanonicalBlock | None:
    same_page_visuals = [
        block
        for block in mineru_blocks
        if block.page_idx == qwen_block.page_idx and block.type in {"chart", "image", "table"}
    ]
    if not same_page_visuals:
        return None
    return max(
        same_page_visuals,
        key=lambda block: (
            bbox_iou(block.bbox, qwen_block.bbox),
            1 if block.type == "chart" else 0,
            1 if str(block.sub_type or "").strip().lower() == "flowchart" else 0,
        ),
    )


def _is_seal_candidate(block: CanonicalBlock) -> bool:
    if block.type != "image":
        return False
    if str(block.sub_type or "").strip().lower() == "seal":
        return True
    return any(str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions)


def _is_flowchart_candidate(block: CanonicalBlock) -> bool:
    if block.type != "chart":
        return False
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if block.structured_label.kind == "mermaid":
        return True
    return bool(block.flowchart_graph)


def _has_flowchart_signal(
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    graph_fusion_result: Any | None,
) -> bool:
    if graph_fusion_result is not None and str(getattr(graph_fusion_result, "mermaid", "") or "").strip():
        return True
    if _has_flowchart_block_or_label(mineru_document.blocks, mineru_label):
        return True
    return _has_flowchart_block_or_label(qwen_document.blocks, qwen_label)


def _has_flowchart_block_or_label(
    blocks: list[CanonicalBlock],
    label: ParsedLabel | None,
) -> bool:
    if label is not None and label.image_type == "flowchart":
        return True
    return any(_is_flowchart_candidate(block) for block in blocks)


def _pick_flowchart_block(blocks: list[CanonicalBlock]) -> CanonicalBlock | None:
    for block in blocks:
        if _is_flowchart_candidate(block):
            return block
    return _pick_visual_block(blocks)


def _pick_visual_block(blocks: list[CanonicalBlock]) -> CanonicalBlock | None:
    for block in blocks:
        if block.type in {"chart", "image", "table"}:
            return block
    return None


def _extract_flowchart_mermaid(
    blocks: list[CanonicalBlock],
    label: ParsedLabel | None,
) -> str:
    for block in blocks:
        content = str(block.content.get("content", "") or "").strip()
        if _is_flowchart_candidate(block) and looks_like_mermaid(content):
            return content
    if label is not None and label.image_type == "flowchart":
        candidate = str(label.structured_label.content or "").strip()
        if looks_like_mermaid(candidate):
            return candidate
    return ""


def _seal_texts(block: CanonicalBlock) -> list[str]:
    texts: list[str] = []
    texts.extend(
        str(region.text or "").strip()
        for region in block.ocr_regions
        if str(region.role or "").strip().lower() == "seal" and str(region.text or "").strip()
    )
    image_captions = block.content.get("image_caption")
    if isinstance(image_captions, list):
        texts.extend(str(item).strip() for item in image_captions if str(item).strip())
    if str(block.text or "").strip():
        texts.append(str(block.text or "").strip())
    return _deduplicate_texts(texts)


def _texts_conflict(left: list[str], right: list[str]) -> bool:
    left_norm = {_normalize_text(item) for item in left if _normalize_text(item)}
    right_norm = {_normalize_text(item) for item in right if _normalize_text(item)}
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return False
    for left_item in left_norm:
        for right_item in right_norm:
            if left_item in right_item or right_item in left_item:
                return False
    return True


def _deduplicate_issues(issues: list[Issue]) -> list[Issue]:
    ordered: list[Issue] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for issue in issues:
        qwen_block = issue.qwen_block or {}
        key = (
            issue.issue_type,
            issue.target_block_id,
            str(qwen_block.get("block_id")) if isinstance(qwen_block, dict) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(issue)
    return ordered


def _deduplicate_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(str(value).strip())
    return ordered


def _normalize_text(value: Any) -> str:
    return "".join(str(value or "").split()).lower()
