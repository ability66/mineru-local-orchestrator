from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.pipeline.alignment import align_blocks, bbox_iou
from src.pipeline.flowchart_utils import (
    build_flowchart_patch_from_mermaid,
    diff_flowchart_graphs,
    flowchart_graph_from_mermaid,
    looks_like_mermaid,
    mermaid_from_flowchart_graph,
    normalize_mermaid_text,
)
from src.pipeline.normalizers import derive_label_from_document
from src.projection import (
    is_single_block_projection_block,
    project_document_for_single_block_view,
)
from src.schema import CanonicalBlock, CanonicalDocument, ImageTask, Issue, ParsedLabel


def detect_seal_issues(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
) -> list[Issue]:
    del image_task
    comparison_mineru_document = _project_seal_comparison_document(mineru_document)
    comparison_qwen_document = _project_seal_comparison_document(qwen_document)
    matches = align_blocks(
        comparison_mineru_document.blocks,
        comparison_qwen_document.blocks,
    )
    match_lookup = {match.base_index: match for match in matches}
    matched_qwen_indexes = {match.candidate_index for match in matches}

    issues: list[Issue] = []
    for mineru_index, mineru_block in enumerate(comparison_mineru_document.blocks):
        match = match_lookup.get(mineru_index)
        qwen_block = (
            comparison_qwen_document.blocks[match.candidate_index]
            if match is not None
            else None
        )
        issue = _detect_pair_issue(mineru_block=mineru_block, qwen_block=qwen_block)
        if issue is not None:
            issues.append(issue)

    for qwen_index, qwen_block in enumerate(comparison_qwen_document.blocks):
        if qwen_index in matched_qwen_indexes or not _is_seal_candidate(qwen_block):
            continue
        target_block = _find_best_target_block(
            comparison_mineru_document.blocks, qwen_block
        )
        issues.append(
            Issue(
                issue_id=f"seal-unmatched-{qwen_block.block_id}",
                issue_type="seal_unmatched_qwen_candidate",
                page_idx=qwen_block.page_idx,
                target_block_id=target_block.block_id
                if target_block is not None
                else None,
                mineru_block=target_block.model_dump()
                if target_block is not None
                else None,
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
    graph_fusion_result: Any | None = None,
) -> list[Issue]:
    del graph_fusion_result
    del image_task
    if not _has_flowchart_signal(
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
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
            target_block = _find_best_flowchart_target_block(
                mineru_document.blocks, qwen_flowchart_block
            )
    if target_block is None:
        target_block = mineru_flowchart_block or _pick_visual_block(
            mineru_document.blocks
        )

    current_mermaid = _extract_flowchart_mermaid(mineru_document.blocks, mineru_label)
    reference_mermaid = _extract_flowchart_mermaid(qwen_document.blocks, qwen_label)
    current_graph = _extract_flowchart_graph(
        mineru_document.blocks, mineru_label, current_mermaid
    )
    reference_graph = _extract_flowchart_graph(
        qwen_document.blocks, qwen_label, reference_mermaid
    )

    qwen_block_payload = (
        qwen_flowchart_block.model_dump() if qwen_flowchart_block is not None else None
    )
    mineru_block_payload = (
        target_block.model_dump() if target_block is not None else None
    )
    issue_id_target = (
        target_block.block_id
        if target_block is not None
        else (
            qwen_flowchart_block.block_id
            if qwen_flowchart_block is not None
            else "flowchart"
        )
    )
    page_idx = (
        target_block.page_idx
        if target_block is not None
        else (qwen_flowchart_block.page_idx if qwen_flowchart_block is not None else 0)
    )

    shared_payload = {
        "current_mermaid": current_mermaid,
        "reference_mermaid": reference_mermaid,
        "reference_patch": build_flowchart_patch_from_mermaid(reference_mermaid),
    }

    issues: list[Issue] = []
    if (
        _has_flowchart_block_or_label(mineru_document.blocks, mineru_label)
        and current_graph is None
    ):
        issues.append(
            Issue(
                issue_id=f"flowchart-diff-{issue_id_target}-current-missing",
                issue_type="flowchart_graph_conflict",
                page_idx=page_idx,
                target_block_id=target_block.block_id
                if target_block is not None
                else None,
                mineru_block=mineru_block_payload,
                qwen_block=qwen_block_payload,
                candidate_payload={
                    **shared_payload,
                    "graph_diff": {"diff_kind": "current_graph_missing"},
                },
                reasons=["mineru_flowchart_missing_valid_mermaid_or_graph"],
            )
        )
    if (
        _has_flowchart_block_or_label(qwen_document.blocks, qwen_label)
        and reference_graph is None
    ):
        issues.append(
            Issue(
                issue_id=f"flowchart-diff-{issue_id_target}-reference-missing",
                issue_type="flowchart_graph_conflict",
                page_idx=page_idx,
                target_block_id=target_block.block_id
                if target_block is not None
                else None,
                mineru_block=mineru_block_payload,
                qwen_block=qwen_block_payload,
                candidate_payload={
                    **shared_payload,
                    "graph_diff": {"diff_kind": "reference_graph_missing"},
                },
                reasons=["qwen_flowchart_missing_valid_mermaid_or_graph"],
            )
        )

    for diff_index, diff in enumerate(
        diff_flowchart_graphs(current_graph, reference_graph), start=1
    ):
        diff_kind = (
            str(diff.get("diff_kind", "") or "graph_conflict").strip()
            or "graph_conflict"
        )
        issues.append(
            Issue(
                issue_id=f"flowchart-diff-{issue_id_target}-{diff_kind}-{diff_index}",
                issue_type="flowchart_graph_conflict",
                page_idx=page_idx,
                target_block_id=target_block.block_id
                if target_block is not None
                else None,
                mineru_block=mineru_block_payload,
                qwen_block=qwen_block_payload,
                candidate_payload={
                    **shared_payload,
                    "graph_diff": diff,
                },
                reasons=_deduplicate_texts(
                    ["flowchart_graph_conflict_detected", diff_kind.replace("_", " ")]
                ),
            )
        )

    return _deduplicate_issues(issues)


def detect_flowchart_second_pass_issues(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
) -> list[Issue]:
    del image_task
    if not _has_flowchart_block_or_label(mineru_document.blocks, mineru_label):
        return []

    target_block = _pick_flowchart_block(mineru_document.blocks)
    if target_block is None:
        return []

    current_mermaid = _extract_flowchart_mermaid(mineru_document.blocks, mineru_label)
    return [
        Issue(
            issue_id=f"flowchart-second-pass-{target_block.block_id}",
            issue_type="flowchart_graph_conflict",
            page_idx=target_block.page_idx,
            target_block_id=target_block.block_id,
            mineru_block=target_block.model_dump(),
            qwen_block=None,
            candidate_payload={
                "review_mode": "second_pass",
                "current_mermaid": current_mermaid,
                "graph_diff": {
                    "diff_kind": "second_pass_required",
                },
            },
            reasons=["flowchart_requires_qwen_second_pass"],
        )
    ]


def build_html_table_issue(
    image_task: ImageTask,
    mineru_candidate: dict[str, Any] | None,
    candidate_bundles: list[dict[str, Any]],
    consensus_analysis: dict[str, Any],
) -> Issue | None:
    del image_task
    if not isinstance(mineru_candidate, dict):
        return None
    block = mineru_candidate.get("block")
    if not isinstance(block, CanonicalBlock):
        return None

    candidate_payloads: list[dict[str, Any]] = []
    reference_bundle: dict[str, Any] | None = None
    reference_role = str(consensus_analysis.get("reference_role", "") or "").strip().lower()
    for bundle in candidate_bundles:
        role = str(bundle.get("role", "") or "").strip().lower()
        candidate = bundle.get("html_table_candidate")
        if not isinstance(candidate, dict):
            continue
        block_candidate = candidate.get("block")
        output = bundle.get("output")
        candidate_payloads.append(
            {
                "candidate_id": role,
                "model_name": getattr(output, "model_name", "") or bundle.get("role", ""),
                "block_id": candidate.get("block_id"),
                "block_type": candidate.get("block_type"),
                "sub_type": candidate.get("sub_type"),
                "caption": candidate.get("caption"),
                "visible_text": list(candidate.get("visible_text") or [])[:10],
                "ocr_texts": list(candidate.get("ocr_texts") or [])[:10],
                "table_format": str(candidate.get("table_format", "") or ""),
                "table_content": str(candidate.get("table_text", "") or ""),
                "html_table": str(candidate.get("html", "") or ""),
                "cell_count": len(getattr(candidate.get("table_ir"), "cells", []) or []),
                "row_count": getattr(candidate.get("table_ir"), "row_count", 0),
                "col_count": getattr(candidate.get("table_ir"), "col_count", 0),
            }
        )
        if role == reference_role and isinstance(block_candidate, CanonicalBlock):
            reference_bundle = bundle

    reference_patch = _build_html_table_reference_patch(reference_bundle)
    reasons = list(consensus_analysis.get("review_reasons") or [])
    if not reasons:
        reasons = ["html_table_candidates_diverge"]

    return Issue(
        issue_id=f"html-table-{block.block_id}",
        issue_type="html_table_conflict",
        page_idx=block.page_idx,
        target_block_id=block.block_id,
        mineru_block=block.model_dump(),
        qwen_block=(
            reference_bundle["html_table_candidate"]["block"].model_dump()
            if isinstance(reference_bundle, dict)
            and isinstance(reference_bundle.get("html_table_candidate"), dict)
            and isinstance(reference_bundle["html_table_candidate"].get("block"), CanonicalBlock)
            else None
        ),
        candidate_payload={
            "review_mode": "html_table_disagreement",
            "candidates": candidate_payloads,
            "pairwise_matrix": deepcopy(consensus_analysis.get("matrix") or {}),
            "pairwise_scores": deepcopy(consensus_analysis.get("pairwise") or []),
            "consensus_diagnostics": {
                "stable_consensus": bool(consensus_analysis.get("stable_consensus", False)),
                "consensus_kind": str(consensus_analysis.get("consensus_kind", "") or ""),
                "consensus_cluster": list(consensus_analysis.get("consensus_cluster") or []),
                "severe_conflicts": list(consensus_analysis.get("severe_conflicts") or []),
            },
            "reference_model_role": reference_role or None,
            "reference_model_name": (
                getattr(reference_bundle.get("output"), "model_name", "")
                if isinstance(reference_bundle, dict)
                else ""
            )
            or None,
            "reference_patch": reference_patch,
            "candidate_patch": reference_patch,
        },
        reasons=reasons,
    )


def _detect_pair_issue(
    mineru_block: CanonicalBlock, qwen_block: CanonicalBlock | None
) -> Issue | None:
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

    compare_full_projection = is_single_block_projection_block(
        mineru_block
    ) or is_single_block_projection_block(qwen_block)
    if mineru_texts and qwen_texts and _texts_conflict(
        mineru_texts,
        qwen_texts,
        allow_containment=not compare_full_projection,
    ):
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


def _build_html_table_reference_patch(
    bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(bundle, dict):
        return {}
    candidate = bundle.get("html_table_candidate")
    if not isinstance(candidate, dict):
        return {}
    block = candidate.get("block")
    if not isinstance(block, CanonicalBlock):
        return {}
    patch: dict[str, Any] = {
        "type": block.type,
        "content": deepcopy(block.content),
        "visible_text": list(block.visible_text),
    }
    if block.sub_type is not None:
        patch["sub_type"] = block.sub_type
    candidate_text = str(candidate.get("table_text", "") or candidate.get("html", "") or "")
    if block.type == "table":
        patch["content"]["table_body"] = str(
            candidate_text or patch["content"].get("table_body", "") or ""
        )
    elif block.type == "chart":
        patch["type"] = "table"
        patch["content"]["table_body"] = str(
            candidate_text or patch["content"].get("content", "") or ""
        )
        if "chart_caption" in patch["content"] and "table_caption" not in patch["content"]:
            patch["content"]["table_caption"] = patch["content"].get("chart_caption")
        patch["content"].pop("content", None)
        patch["content"].pop("chart_caption", None)
        patch["sub_type"] = None
    return patch


def _find_best_target_block(
    mineru_blocks: list[CanonicalBlock], qwen_block: CanonicalBlock
) -> CanonicalBlock | None:
    same_page_visuals = [
        block
        for block in mineru_blocks
        if block.page_idx == qwen_block.page_idx
        and block.type in {"image", "chart", "table"}
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
        if block.page_idx == qwen_block.page_idx
        and block.type in {"chart", "image", "table"}
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
    return any(
        str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions
    )


def _is_flowchart_candidate(block: CanonicalBlock) -> bool:
    if block.type not in {"chart", "image"}:
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
) -> bool:
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
        if not _is_flowchart_candidate(block):
            continue
        for candidate in _flowchart_text_candidates(block):
            normalized_candidate = normalize_mermaid_text(candidate)
            if looks_like_mermaid(normalized_candidate):
                return normalized_candidate
    if label is not None and label.image_type == "flowchart":
        candidate = normalize_mermaid_text(
            str(label.structured_label.content or "").strip()
        )
        if looks_like_mermaid(candidate):
            return candidate
    return ""


def _extract_flowchart_graph(
    blocks: list[CanonicalBlock],
    label: ParsedLabel | None,
    mermaid: str,
) -> dict[str, Any] | None:
    for block in blocks:
        if _is_flowchart_candidate(block) and isinstance(block.flowchart_graph, dict):
            return block.flowchart_graph
    if mermaid:
        derived = flowchart_graph_from_mermaid(mermaid)
        if derived is not None:
            return derived
    if label is not None and isinstance(label.flowchart_graph, dict):
        return label.flowchart_graph
    return None


def _flowchart_text_candidates(block: CanonicalBlock) -> list[str]:
    values: list[str] = []
    content_value = str(block.content.get("content", "") or "").strip()
    if content_value:
        values.append(content_value)
    if isinstance(block.content.get("chart_caption"), list):
        values.extend(
            str(item).strip()
            for item in block.content.get("chart_caption", [])
            if str(item).strip()
        )
    if isinstance(block.content.get("image_caption"), list):
        values.extend(
            str(item).strip()
            for item in block.content.get("image_caption", [])
            if str(item).strip()
        )
    if str(block.text or "").strip():
        values.append(str(block.text or "").strip())
    if (
        block.structured_label.kind == "mermaid"
        and block.structured_label.content.strip()
    ):
        values.append(block.structured_label.content.strip())
    return _deduplicate_texts(values)


def _seal_texts(block: CanonicalBlock) -> list[str]:
    texts: list[str] = []
    texts.extend(
        str(region.text or "").strip()
        for region in block.ocr_regions
        if str(region.role or "").strip().lower() == "seal"
        and str(region.text or "").strip()
    )
    image_captions = block.content.get("image_caption")
    if isinstance(image_captions, list):
        texts.extend(str(item).strip() for item in image_captions if str(item).strip())
    if str(block.text or "").strip():
        texts.append(str(block.text or "").strip())
    return _deduplicate_texts(texts)


def _project_seal_comparison_document(
    document: CanonicalDocument,
) -> CanonicalDocument:
    label = derive_label_from_document(document)
    projected = project_document_for_single_block_view(document=document, label=label)
    return projected if isinstance(projected, CanonicalDocument) else document


def _texts_conflict(
    left: list[str],
    right: list[str],
    allow_containment: bool = True,
) -> bool:
    left_norm = {_normalize_text(item) for item in left if _normalize_text(item)}
    right_norm = {_normalize_text(item) for item in right if _normalize_text(item)}
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return False
    if allow_containment:
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
