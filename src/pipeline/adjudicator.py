from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.decision import decide_consensus
from src.graph_fusion import fuse_mermaid_outputs
from src.pipeline.alignment import BlockMatch, align_blocks
from src.pipeline.normalizers import derive_label_from_document
from src.schema import (
    AdjudicationArtifact,
    CanonicalBlock,
    CanonicalDocument,
    ConsensusResult,
    ImageTask,
    Issue,
    ModelOutput,
    OcrRegion,
    PatchDecision,
    ParsedLabel,
    StructuredLabel,
)
from src.scorer import score_consensus
from src.seal_utils import is_stamp_mode
from src.validators import ValidationResult


def adjudicate_documents(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    issues: list[Issue] | None = None,
    patch_decisions: list[PatchDecision] | None = None,
) -> AdjudicationArtifact:
    base_document = mineru_document if mineru_document.blocks or not qwen_document.blocks else qwen_document
    candidate_document = qwen_document if base_document is mineru_document else mineru_document
    matches = align_blocks(base_document.blocks, candidate_document.blocks)
    match_lookup = {match.base_index: match for match in matches}

    merged_blocks: list[CanonicalBlock] = []
    for base_index, base_block in enumerate(base_document.blocks):
        match = match_lookup.get(base_index)
        candidate_block = (
            candidate_document.blocks[match.candidate_index] if match is not None else None
        )
        merged_blocks.append(_merge_block(base_block=base_block, candidate_block=candidate_block, match=match))

    matched_candidate_indexes = {match.candidate_index for match in matches}
    added_qwen_blocks = 0
    if candidate_document is qwen_document:
        for candidate_index, candidate_block in enumerate(candidate_document.blocks):
            if candidate_index in matched_candidate_indexes:
                continue
            if not _should_add_unmatched_qwen_block(candidate_block):
                continue
            added_qwen_blocks += 1
            added_block = candidate_block.model_copy(deep=True)
            added_block.provenance["added_from_qwen"] = True
            merged_blocks.append(added_block)

    merged_blocks.sort(key=lambda item: (item.page_idx, item.order_index, item.block_id))
    final_document = CanonicalDocument(
        document_id=image_task.image_id,
        source="adjudicated",
        backend="mineru_plus_qwen",
        page_count=max(mineru_document.page_count, qwen_document.page_count, 1),
        blocks=merged_blocks,
        warnings=_collect_document_warnings(mineru_document, qwen_document),
        raw_metadata={
            "base_source": base_document.source,
            "candidate_source": candidate_document.source,
        },
    )

    graph_fusion_result = _maybe_fuse_flowchart(
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        fallback_visible_text=_collect_visible_text(final_document.blocks),
    )
    validation_result = _build_validation_result(
        image_id=image_task.image_id,
        base_document=base_document,
        candidate_document=candidate_document,
        matches=matches,
        added_qwen_blocks=added_qwen_blocks,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
    )
    consensus = _build_consensus(
        image_id=image_task.image_id,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        validation_result=validation_result,
        graph_fusion_result=graph_fusion_result,
    )
    consensus = _override_stamp_mode_consensus(
        image_id=image_task.image_id,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        issues=issues or [],
        patch_decisions=patch_decisions or [],
        existing=consensus,
    )
    preferred_label, allow_type_override, allow_graph_override = _pick_enrichment_policy(
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        consensus=consensus,
    )
    _inject_label_enrichment(
        final_document=final_document,
        preferred_label=preferred_label,
        graph_fusion_result=graph_fusion_result if allow_graph_override else None,
        allow_type_override=allow_type_override,
    )
    final_label = derive_label_from_document(final_document)

    local_reasons = _build_local_reasons(
        base_document=base_document,
        candidate_document=candidate_document,
        matches=matches,
        added_qwen_blocks=added_qwen_blocks,
    )
    reasons = list(local_reasons)
    if consensus is not None:
        reasons.extend(consensus.reasons)

    return AdjudicationArtifact(
        image_id=image_task.image_id,
        final_document=final_document,
        consensus=consensus,
        final_label=final_label,
        graph_fusion=asdict(graph_fusion_result) if graph_fusion_result is not None else None,
        matched_block_count=len(matches),
        added_qwen_block_count=added_qwen_blocks,
        review_required=consensus is None or consensus.decision != "accepted",
        reasons=_deduplicate(reasons),
        warnings=_deduplicate(final_document.warnings),
        issues=list(issues or []),
        patch_decisions=list(patch_decisions or []),
    )


def _merge_block(
    base_block: CanonicalBlock,
    candidate_block: CanonicalBlock | None,
    match: BlockMatch | None,
) -> CanonicalBlock:
    merged = base_block.model_copy(deep=True)
    if candidate_block is None:
        return merged

    if not merged.text.strip() and candidate_block.text.strip():
        merged.text = candidate_block.text
    elif (
        merged.type in {"title", "paragraph"}
        and candidate_block.text.strip()
        and len(candidate_block.text.strip()) > len(merged.text.strip())
    ):
        merged.text = candidate_block.text.strip()

    merged.visible_text = _deduplicate(merged.visible_text + candidate_block.visible_text)
    merged.ocr_regions = _merge_ocr_regions(merged.ocr_regions, candidate_block.ocr_regions)
    merged.warnings = _deduplicate(merged.warnings + candidate_block.warnings)
    merged.provenance["matched_qwen_block_id"] = candidate_block.block_id

    if match is not None:
        merged.provenance["alignment_score"] = match.score
        merged.provenance["alignment_bbox_iou"] = match.bbox_iou
        merged.provenance["alignment_text_similarity"] = match.text_similarity

    if merged.type == "table":
        merged = _merge_table_block(merged, candidate_block)
    elif merged.type in {"chart", "image"}:
        merged = _merge_visual_block(merged, candidate_block)

    return merged


def _merge_table_block(base_block: CanonicalBlock, candidate_block: CanonicalBlock) -> CanonicalBlock:
    table_body = str(base_block.content.get("table_body", "") or "").strip()
    candidate_body = str(candidate_block.content.get("table_body", "") or "").strip()
    if not table_body and candidate_block.structured_label.kind == "table":
        candidate_body = candidate_block.structured_label.content.strip()
    if not table_body and candidate_body:
        base_block.content["table_body"] = candidate_body
        base_block.structured_label = StructuredLabel(
            kind="table",
            content=candidate_body,
            format="markdown",
            source="model",
        )

    captions = _deduplicate(
        _as_text_list(base_block.content.get("table_caption")) +
        _as_text_list(candidate_block.content.get("table_caption"))
    )
    if captions:
        base_block.content["table_caption"] = captions
    return base_block


def _merge_visual_block(base_block: CanonicalBlock, candidate_block: CanonicalBlock) -> CanonicalBlock:
    if candidate_block.caption_structured.brief.strip():
        if base_block.type == "chart":
            captions = _deduplicate(
                _as_text_list(base_block.content.get("chart_caption")) +
                [candidate_block.caption_structured.brief.strip()]
            )
            base_block.content["chart_caption"] = captions
        elif base_block.type == "image":
            captions = _deduplicate(
                _as_text_list(base_block.content.get("image_caption")) +
                [candidate_block.caption_structured.brief.strip()]
            )
            base_block.content["image_caption"] = captions

    if candidate_block.structured_label.kind == "table" and not str(base_block.content.get("content", "")).strip():
        base_block.content["content"] = candidate_block.structured_label.content
    return base_block


def _inject_label_enrichment(
    final_document: CanonicalDocument,
    preferred_label: ParsedLabel | None,
    graph_fusion_result: Any | None,
    allow_type_override: bool,
) -> None:
    if preferred_label is None or not final_document.blocks:
        return

    target = _pick_visual_target(final_document.blocks)
    if target is None:
        target = final_document.blocks[0]

    if graph_fusion_result is not None and str(graph_fusion_result.mermaid or "").strip():
        target.type = "chart"
        target.sub_type = "flowchart"
        target.content["img_path"] = target.content.get("img_path") or ""
        target.content["content"] = graph_fusion_result.mermaid
        if preferred_label.caption.strip():
            target.content["chart_caption"] = _deduplicate(
                _as_text_list(target.content.get("chart_caption")) + [preferred_label.caption.strip()]
            )
        target.structured_label = StructuredLabel(
            kind="mermaid",
            content=graph_fusion_result.mermaid,
            format="mermaid",
            source="fused_graph",
            graph_confidence=graph_fusion_result.graph_confidence,
        )
        if preferred_label.flowchart_graph:
            target.flowchart_graph = preferred_label.flowchart_graph
        return

    if preferred_label.image_type == "table":
        if not allow_type_override and target.type != "table":
            return
        target.type = "table"
        target.content["table_body"] = preferred_label.structured_label.content
        if preferred_label.caption.strip():
            target.content["table_caption"] = _deduplicate(
                _as_text_list(target.content.get("table_caption")) + [preferred_label.caption.strip()]
            )
        target.structured_label = preferred_label.structured_label
        return

    if preferred_label.image_type in {"chart", "flowchart"}:
        if not allow_type_override and target.type != "chart":
            return
        target.type = "chart"
        if preferred_label.image_type == "flowchart":
            target.sub_type = "flowchart"
        if preferred_label.structured_label.content.strip():
            target.content["content"] = preferred_label.structured_label.content
            target.structured_label = preferred_label.structured_label
        if preferred_label.caption.strip():
            target.content["chart_caption"] = _deduplicate(
                _as_text_list(target.content.get("chart_caption")) + [preferred_label.caption.strip()]
            )

    if any(str(region.role or "").strip() == "seal" for region in preferred_label.ocr_regions):
        if not allow_type_override and target.type != "image" and str(target.sub_type or "").strip().lower() != "seal":
            return
        target.type = "image"
        target.sub_type = "seal"
        target.ocr_regions = _merge_ocr_regions(target.ocr_regions, preferred_label.ocr_regions)
        if preferred_label.caption.strip():
            target.content["image_caption"] = _deduplicate(
                _as_text_list(target.content.get("image_caption")) + [preferred_label.caption.strip()]
            )


def _pick_visual_target(blocks: list[CanonicalBlock]) -> CanonicalBlock | None:
    for block in blocks:
        if block.type in {"chart", "image", "table"}:
            return block
    return None


def _should_add_unmatched_qwen_block(block: CanonicalBlock) -> bool:
    if block.type in {"chart", "table"}:
        return True
    if block.type == "image" and str(block.sub_type or "").strip().lower() == "seal":
        return True
    return False


def _maybe_fuse_flowchart(
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    fallback_visible_text: list[str],
) -> Any | None:
    paired = [
        (output, label)
        for output, label in (
            (mineru_output, mineru_label),
            (qwen_output, qwen_label),
        )
        if output is not None and label is not None
    ]
    if len(paired) < 2:
        return None

    labels = [label for _, label in paired]
    if not any(label.image_type == "flowchart" for label in labels):
        return None

    outputs = [output for output, _ in paired]
    return fuse_mermaid_outputs(labels, outputs, fallback_visible_text)


def _build_validation_result(
    image_id: str,
    base_document: CanonicalDocument,
    candidate_document: CanonicalDocument,
    matches: list[BlockMatch],
    added_qwen_blocks: int,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
) -> ValidationResult:
    base_count = len(base_document.blocks)
    candidate_count = len(candidate_document.blocks)
    match_ratio = len(matches) / max(base_count, 1)
    qwen_add_ratio = added_qwen_blocks / max(candidate_count, 1)

    type_conflict = 0.0
    structure_conflict = 0.0
    if mineru_label is not None and qwen_label is not None:
        if mineru_label.image_type != qwen_label.image_type:
            type_conflict = 1.0
        if mineru_label.structured_label.kind != qwen_label.structured_label.kind:
            structure_conflict = 1.0

    evidence_score = max(0.0, min(1.0, 0.55 + 0.35 * match_ratio - 0.10 * qwen_add_ratio))
    validator_score = max(
        0.0,
        min(1.0, 0.60 + 0.25 * match_ratio - 0.10 * type_conflict - 0.10 * structure_conflict),
    )
    hallucination_risk = max(
        0.0,
        min(1.0, 0.10 + 0.35 * (1 - match_ratio) + 0.20 * qwen_add_ratio + 0.15 * structure_conflict),
    )

    warnings: list[str] = []
    if candidate_count > 0 and not matches:
        warnings.append("no_block_alignment_between_mineru_and_qwen")
    if added_qwen_blocks > 0:
        warnings.append("unmatched_qwen_blocks_were_appended")

    details = {
        "base_block_count": base_count,
        "candidate_block_count": candidate_count,
        "match_ratio": round(match_ratio, 4),
        "qwen_add_ratio": round(qwen_add_ratio, 4),
        "type_conflict": type_conflict,
        "structure_conflict": structure_conflict,
    }
    return ValidationResult(
        image_id=image_id,
        evidence_score=round(evidence_score, 4),
        validator_score=round(validator_score, 4),
        hallucination_risk=round(hallucination_risk, 4),
        critical_errors=[],
        warnings=warnings,
        details=details,
    )


def _build_consensus(
    image_id: str,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    validation_result: ValidationResult,
    graph_fusion_result: Any | None,
) -> ConsensusResult | None:
    paired = [
        (output, label)
        for output, label in (
            (mineru_output, mineru_label),
            (qwen_output, qwen_label),
        )
        if output is not None and label is not None
    ]
    if not paired:
        return None

    model_outputs = [output for output, _ in paired]
    labels = [label for _, label in paired]
    score_result = score_consensus(image_id=image_id, labels=labels, model_outputs=model_outputs)
    return decide_consensus(
        image_id=image_id,
        labels=labels,
        model_outputs=model_outputs,
        score_result=score_result,
        validation_result=validation_result,
        graph_fusion_result=graph_fusion_result,
    )


def _override_stamp_mode_consensus(
    image_id: str,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    issues: list[Issue],
    patch_decisions: list[PatchDecision],
    existing: ConsensusResult | None,
) -> ConsensusResult | None:
    labels = [label for label in (mineru_label, qwen_label) if label is not None]
    if not _is_issue_driven_stamp_mode(
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        labels=labels,
        issues=issues,
    ):
        return existing

    both_succeeded = bool(
        mineru_output is not None
        and qwen_output is not None
        and mineru_output.success
        and qwen_output.success
    )
    both_parsed = mineru_label is not None and qwen_label is not None
    if not both_succeeded:
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="review",
            reasons=["stamp mode requires both models to succeed before auto-accept"],
            escalation_reasons=["stamp_mode_model_failure"],
            existing=existing,
        )
    if not both_parsed:
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="review",
            reasons=["stamp mode requires both models to produce parsable labels before auto-accept"],
            escalation_reasons=["stamp_mode_missing_parsed_label"],
            existing=existing,
        )
    if not issues:
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="accepted",
            reasons=["no seal issues detected"],
            escalation_reasons=[],
            existing=existing,
        )

    unresolved_issues = _find_unresolved_stamp_issues(issues=issues, patch_decisions=patch_decisions)
    if unresolved_issues:
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="review",
            reasons=["seal issues remain unresolved after second-stage adjudication"] + unresolved_issues,
            escalation_reasons=["stamp_issues_unresolved"],
            existing=existing,
        )

    return _build_issue_driven_consensus(
        image_id=image_id,
        decision="accepted",
        reasons=["seal issues resolved by second-stage adjudication"],
        escalation_reasons=[],
        existing=existing,
    )


def _is_issue_driven_stamp_mode(
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    labels: list[ParsedLabel],
    issues: list[Issue],
) -> bool:
    if issues:
        return True
    if labels and is_stamp_mode(labels):
        return True
    return any(_is_seal_like_block(block) for block in mineru_document.blocks + qwen_document.blocks)


def _is_seal_like_block(block: CanonicalBlock) -> bool:
    if block.type != "image":
        return False
    if str(block.sub_type or "").strip().lower() == "seal":
        return True
    return any(str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions)


def _find_unresolved_stamp_issues(
    issues: list[Issue],
    patch_decisions: list[PatchDecision],
) -> list[str]:
    decision_lookup = {decision.issue_id: decision for decision in patch_decisions}
    unresolved: list[str] = []
    failure_reasons = {"llm_patch_unavailable", "llm_patch_invalid_json"}

    for issue in issues:
        decision = decision_lookup.get(issue.issue_id)
        if decision is None:
            unresolved.append(f"missing_patch_decision:{issue.issue_id}")
            continue
        if str(decision.reason or "").strip() in failure_reasons:
            unresolved.append(f"patch_decision_unavailable:{issue.issue_id}")
    return unresolved


def _build_issue_driven_consensus(
    image_id: str,
    decision: str,
    reasons: list[str],
    escalation_reasons: list[str],
    existing: ConsensusResult | None,
) -> ConsensusResult:
    metrics = existing.model_dump() if existing is not None else {}
    accepted = decision == "accepted"
    return ConsensusResult(
        image_id=image_id,
        type_agreement=1.0 if accepted else float(metrics.get("type_agreement", 0.0)),
        caption_agreement=1.0 if accepted else float(metrics.get("caption_agreement", 0.0)),
        structure_agreement=1.0 if accepted else float(metrics.get("structure_agreement", 0.0)),
        seal_agreement=1.0 if accepted else float(metrics.get("seal_agreement", 1.0)),
        overall_score=1.0 if accepted else float(metrics.get("overall_score", 0.0)),
        evidence_score=1.0 if accepted else float(metrics.get("evidence_score", 0.0)),
        validator_score=1.0 if accepted else float(metrics.get("validator_score", 0.0)),
        hallucination_risk=0.0 if accepted else float(metrics.get("hallucination_risk", 1.0)),
        accept_score=1.0 if accepted else float(metrics.get("accept_score", 0.0)),
        decision=decision,  # type: ignore[arg-type]
        reasons=_deduplicate(reasons),
        validation_errors=[],
        validation_warnings=[],
        escalation_reasons=_deduplicate(escalation_reasons),
    )


def _pick_enrichment_policy(
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    consensus: ConsensusResult | None,
) -> tuple[ParsedLabel | None, bool, bool]:
    if qwen_label is None:
        return mineru_label, True, True
    if mineru_label is None:
        return qwen_label, True, True
    if consensus is not None and consensus.decision == "accepted":
        return qwen_label, True, True
    return mineru_label, False, False


def _build_local_reasons(
    base_document: CanonicalDocument,
    candidate_document: CanonicalDocument,
    matches: list[BlockMatch],
    added_qwen_blocks: int,
) -> list[str]:
    reasons: list[str] = []
    if not base_document.blocks:
        reasons.append("base_document_empty")
    if candidate_document.blocks and not matches:
        reasons.append("no_block_matches_found")
    if added_qwen_blocks > 0:
        reasons.append("qwen_added_unmatched_visual_blocks")
    return reasons


def _collect_visible_text(blocks: list[CanonicalBlock]) -> list[str]:
    texts: list[str] = []
    for block in blocks:
        if block.text.strip():
            texts.append(block.text.strip())
        texts.extend(block.visible_text)
    return _deduplicate(texts)


def _collect_document_warnings(
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
) -> list[str]:
    return _deduplicate(mineru_document.warnings + qwen_document.warnings)


def _merge_ocr_regions(left: list[OcrRegion], right: list[OcrRegion]) -> list[OcrRegion]:
    merged: list[OcrRegion] = []
    seen: set[tuple[str, str]] = set()
    for region in list(left) + list(right):
        key = (str(region.role or "").strip(), str(region.text or "").strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append(region)
    return merged


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _deduplicate(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(text)
    return ordered
