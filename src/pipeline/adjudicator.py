from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.decision import decide_consensus
from src.pipeline.alignment import BlockMatch, align_blocks
from src.pipeline.flowchart_utils import looks_like_mermaid
from src.pipeline.normalizers import derive_label_from_document
from src.pipeline.table_evaluator import analyze_html_table_candidate_consensus
from src.pipeline.table_utils import extract_best_html_table_candidate, is_html_table_like
from src.projection import (
    is_single_block_projection_document,
    project_document_for_single_block_view,
)
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
    SealSelectionDecision,
    StructuredLabel,
)
from src.scorer import score_consensus
from src.seal_utils import is_stamp_mode
from src.validators import ValidationResult


def analyze_html_table_bundles(
    mineru_bundle: dict[str, Any],
    auxiliary_bundles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ordered_bundles = [{"role": "mineru", **mineru_bundle}] + list(auxiliary_bundles)
    candidate_bundles: list[dict[str, Any]] = []
    candidate_payloads: list[dict[str, Any]] = []
    signaled_roles: list[str] = []
    for bundle in ordered_bundles:
        role = str(bundle.get("role", "") or "").strip().lower() or "candidate"
        document = bundle.get("document")
        output = bundle.get("output")
        label = bundle.get("label")
        if not isinstance(document, CanonicalDocument):
            continue
        if isinstance(output, ModelOutput) and not output.success:
            continue
        if not is_html_table_like(document) and not is_html_table_like(label):
            continue
        signaled_roles.append(role)
        html_table_candidate = extract_best_html_table_candidate(document)
        if not isinstance(html_table_candidate, dict):
            continue
        candidate_bundle = {
            **bundle,
            "role": role,
            "html_table_candidate": html_table_candidate,
        }
        candidate_bundles.append(candidate_bundle)
        candidate_payloads.append(
            {
                "role": role,
                "html": str(html_table_candidate.get("html", "") or ""),
            }
        )

    analysis = analyze_html_table_candidate_consensus(candidate_payloads)
    if analysis is None:
        if signaled_roles:
            return {
                "fallback": True,
                "reason": "html_table_candidate_extraction_failed",
                "candidate_roles": list(dict.fromkeys(signaled_roles)),
                "candidate_bundles": candidate_bundles,
                "role_lookup": {
                    str(bundle.get("role", "") or "").strip().lower(): bundle
                    for bundle in candidate_bundles
                },
                "mineru_candidate": next(
                    (
                        bundle.get("html_table_candidate")
                        for bundle in candidate_bundles
                        if str(bundle.get("role", "") or "").strip().lower() == "mineru"
                    ),
                    None,
                ),
            }
        return None
    role_lookup = {
        str(bundle.get("role", "") or "").strip().lower(): bundle
        for bundle in candidate_bundles
    }
    analysis["candidate_bundles"] = candidate_bundles
    analysis["role_lookup"] = role_lookup
    analysis["mineru_candidate"] = next(
        (
            bundle.get("html_table_candidate")
            for bundle in candidate_bundles
            if str(bundle.get("role", "") or "").strip().lower() == "mineru"
        ),
        None,
    )
    return analysis


def pick_html_table_reference_bundle(
    analysis: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(analysis, dict):
        return None
    role_lookup = analysis.get("role_lookup")
    if not isinstance(role_lookup, dict):
        return None
    reference_role = str(analysis.get("reference_role", "") or "").strip().lower()
    if not reference_role:
        return None
    bundle = role_lookup.get(reference_role)
    return bundle if isinstance(bundle, dict) else None


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
    graph_fusion_result_override: Any | None = None,
    seal_selection: SealSelectionDecision | None = None,
    seal_selected_role: str | None = None,
    seal_selected_document: CanonicalDocument | None = None,
    seal_selected_label: ParsedLabel | None = None,
    seal_selected_output: ModelOutput | None = None,
) -> AdjudicationArtifact:
    base_document = (
        mineru_document
        if mineru_document.blocks or not qwen_document.blocks
        else qwen_document
    )
    candidate_document = (
        qwen_document if base_document is mineru_document else mineru_document
    )
    merge_matches = align_blocks(base_document.blocks, candidate_document.blocks)
    match_lookup = {match.base_index: match for match in merge_matches}

    merged_blocks: list[CanonicalBlock] = []
    for base_index, base_block in enumerate(base_document.blocks):
        match = match_lookup.get(base_index)
        candidate_block = (
            candidate_document.blocks[match.candidate_index]
            if match is not None
            else None
        )
        merged_blocks.append(
            _merge_block(
                base_block=base_block, candidate_block=candidate_block, match=match
            )
        )

    matched_candidate_indexes = {match.candidate_index for match in merge_matches}
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

    merged_blocks.sort(
        key=lambda item: (item.page_idx, item.order_index, item.block_id)
    )
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

    (
        comparison_base_document,
        comparison_candidate_document,
        comparison_matches,
        comparison_added_qwen_blocks,
    ) = _build_comparison_alignment_context(
        base_document=base_document,
        candidate_document=candidate_document,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
    )
    if (
        is_single_block_projection_document(comparison_base_document)
        or is_single_block_projection_document(comparison_candidate_document)
    ):
        final_document.raw_metadata["comparison_view"] = "single_block_projection"

    graph_fusion_result = graph_fusion_result_override
    validation_result = _build_validation_result(
        image_id=image_task.image_id,
        base_document=comparison_base_document,
        candidate_document=comparison_candidate_document,
        matches=comparison_matches,
        added_qwen_blocks=comparison_added_qwen_blocks,
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
        seal_selection=seal_selection,
        existing=consensus,
    )
    consensus = _override_flowchart_mode_consensus(
        image_id=image_task.image_id,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        issues=issues or [],
        patch_decisions=patch_decisions or [],
        graph_fusion_result=graph_fusion_result,
        existing=consensus,
    )
    selected_seal_document = _select_issue_driven_seal_document(
        image_id=image_task.image_id,
        selection=seal_selection,
        mineru_document=mineru_document,
        mineru_output=mineru_output,
        selected_role=seal_selected_role,
        selected_document=seal_selected_document,
        selected_output=seal_selected_output,
        qwen_document=qwen_document,
    )
    if selected_seal_document is not None:
        final_document = selected_seal_document
    selected_flowchart_document = _select_issue_driven_flowchart_document(
        image_id=image_task.image_id,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        issues=issues or [],
        patch_decisions=patch_decisions or [],
        mineru_output=mineru_output,
        qwen_output=qwen_output,
    )
    if selected_flowchart_document is not None:
        final_document = selected_flowchart_document
    preferred_label, allow_type_override, allow_graph_override = (
        _pick_enrichment_policy(
            mineru_label=mineru_label,
            qwen_label=qwen_label,
            consensus=consensus,
        )
    )
    if selected_seal_document is not None:
        preferred_label = (
            seal_selected_label
            if seal_selected_label is not None
            else (
                mineru_label
                if str(seal_selected_role or "").strip().lower() == "mineru"
                else preferred_label
            )
        )
        allow_type_override = False
        allow_graph_override = False
    if _is_issue_driven_flowchart_mode(
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        labels=[label for label in (mineru_label, qwen_label) if label is not None],
        issues=issues or [],
    ):
        selected_role = str(
            final_document.raw_metadata.get("selected_output_role", "") or ""
        ).strip()
        preferred_label = (
            qwen_label or mineru_label
            if selected_role == "qwen"
            else mineru_label or qwen_label
        )
        allow_type_override = False
        allow_graph_override = False
    _inject_label_enrichment(
        final_document=final_document,
        preferred_label=preferred_label,
        graph_fusion_result=graph_fusion_result if allow_graph_override else None,
        allow_type_override=allow_type_override,
    )
    final_label = derive_label_from_document(final_document)

    local_reasons = _build_local_reasons(
        base_document=comparison_base_document,
        candidate_document=comparison_candidate_document,
        matches=comparison_matches,
        added_qwen_blocks=comparison_added_qwen_blocks,
    )
    reasons = list(local_reasons)
    if consensus is not None:
        reasons.extend(consensus.reasons)

    return AdjudicationArtifact(
        image_id=image_task.image_id,
        final_document=final_document,
        consensus=consensus,
        final_label=final_label,
        graph_fusion=asdict(graph_fusion_result)
        if graph_fusion_result is not None
        else None,
        matched_block_count=len(comparison_matches),
        added_qwen_block_count=comparison_added_qwen_blocks,
        review_required=consensus is None or consensus.decision != "accepted",
        reasons=_deduplicate(reasons),
        warnings=_deduplicate(final_document.warnings),
        issues=list(issues or []),
        patch_decisions=list(patch_decisions or []),
        seal_selection=seal_selection,
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

    merged.visible_text = _deduplicate(
        merged.visible_text + candidate_block.visible_text
    )
    merged.ocr_regions = _merge_ocr_regions(
        merged.ocr_regions, candidate_block.ocr_regions
    )
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


def _merge_table_block(
    base_block: CanonicalBlock, candidate_block: CanonicalBlock
) -> CanonicalBlock:
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
        _as_text_list(base_block.content.get("table_caption"))
        + _as_text_list(candidate_block.content.get("table_caption"))
    )
    if captions:
        base_block.content["table_caption"] = captions
    return base_block


def _merge_visual_block(
    base_block: CanonicalBlock, candidate_block: CanonicalBlock
) -> CanonicalBlock:
    if _is_flowchart_like_block(base_block) or _is_flowchart_like_block(
        candidate_block
    ):
        return base_block

    if candidate_block.caption_structured.brief.strip():
        if base_block.type == "chart":
            captions = _deduplicate(
                _as_text_list(base_block.content.get("chart_caption"))
                + [candidate_block.caption_structured.brief.strip()]
            )
            base_block.content["chart_caption"] = captions
        elif base_block.type == "image":
            captions = _deduplicate(
                _as_text_list(base_block.content.get("image_caption"))
                + [candidate_block.caption_structured.brief.strip()]
            )
            base_block.content["image_caption"] = captions

    if (
        candidate_block.structured_label.kind == "table"
        and not str(base_block.content.get("content", "")).strip()
    ):
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

    if (
        graph_fusion_result is not None
        and str(graph_fusion_result.mermaid or "").strip()
    ):
        target.type = "chart"
        target.sub_type = "flowchart"
        target.content["img_path"] = target.content.get("img_path") or ""
        target.content["content"] = graph_fusion_result.mermaid
        if preferred_label.caption.strip():
            target.content["chart_caption"] = _deduplicate(
                _as_text_list(target.content.get("chart_caption"))
                + [preferred_label.caption.strip()]
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
                _as_text_list(target.content.get("table_caption"))
                + [preferred_label.caption.strip()]
            )
        target.structured_label = preferred_label.structured_label
        return

    if preferred_label.image_type in {"chart", "flowchart"}:
        if not allow_type_override and target.type != "chart":
            return
        target.type = "chart"
        if preferred_label.image_type == "flowchart":
            target.sub_type = "flowchart"
        if (
            preferred_label.image_type != "flowchart"
            and preferred_label.structured_label.content.strip()
        ) or (
            preferred_label.image_type == "flowchart"
            and looks_like_mermaid(preferred_label.structured_label.content)
        ):
            target.content["content"] = preferred_label.structured_label.content
            target.structured_label = preferred_label.structured_label
        if preferred_label.caption.strip():
            target.content["chart_caption"] = _deduplicate(
                _as_text_list(target.content.get("chart_caption"))
                + [preferred_label.caption.strip()]
            )

    if any(
        str(region.role or "").strip() == "seal"
        for region in preferred_label.ocr_regions
    ):
        if (
            not allow_type_override
            and target.type != "image"
            and str(target.sub_type or "").strip().lower() != "seal"
        ):
            return
        target.type = "image"
        target.sub_type = "seal"
        target.ocr_regions = _merge_ocr_regions(
            target.ocr_regions, preferred_label.ocr_regions
        )
        if preferred_label.caption.strip():
            target.content["image_caption"] = _deduplicate(
                _as_text_list(target.content.get("image_caption"))
                + [preferred_label.caption.strip()]
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


def build_flowchart_candidate_result(
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    fallback_visible_text: list[str],
) -> Any | None:
    del mineru_label, qwen_label, mineru_output, qwen_output, fallback_visible_text
    return None


def _build_comparison_alignment_context(
    base_document: CanonicalDocument,
    candidate_document: CanonicalDocument,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
) -> tuple[CanonicalDocument, CanonicalDocument, list[BlockMatch], int]:
    projected_mineru_document = _project_comparison_document(
        document=mineru_document,
        label=mineru_label,
    )
    projected_qwen_document = _project_comparison_document(
        document=qwen_document,
        label=qwen_label,
    )
    comparison_base_document = (
        projected_mineru_document
        if base_document is mineru_document
        else projected_qwen_document
    )
    comparison_candidate_document = (
        projected_qwen_document
        if candidate_document is qwen_document
        else projected_mineru_document
    )
    comparison_matches = align_blocks(
        comparison_base_document.blocks,
        comparison_candidate_document.blocks,
    )
    comparison_added_qwen_blocks = 0
    if candidate_document is qwen_document:
        matched_candidate_indexes = {
            match.candidate_index for match in comparison_matches
        }
        for candidate_index, candidate_block in enumerate(
            comparison_candidate_document.blocks
        ):
            if candidate_index in matched_candidate_indexes:
                continue
            if _should_add_unmatched_qwen_block(candidate_block):
                comparison_added_qwen_blocks += 1
    return (
        comparison_base_document,
        comparison_candidate_document,
        comparison_matches,
        comparison_added_qwen_blocks,
    )


def _project_comparison_document(
    document: CanonicalDocument,
    label: ParsedLabel | None,
) -> CanonicalDocument:
    effective_label = label if label is not None else derive_label_from_document(document)
    projected_document = project_document_for_single_block_view(
        document=document,
        label=effective_label,
    )
    return projected_document if isinstance(projected_document, CanonicalDocument) else document


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

    evidence_score = max(
        0.0, min(1.0, 0.55 + 0.35 * match_ratio - 0.10 * qwen_add_ratio)
    )
    validator_score = max(
        0.0,
        min(
            1.0,
            0.60
            + 0.25 * match_ratio
            - 0.10 * type_conflict
            - 0.10 * structure_conflict,
        ),
    )
    hallucination_risk = max(
        0.0,
        min(
            1.0,
            0.10
            + 0.35 * (1 - match_ratio)
            + 0.20 * qwen_add_ratio
            + 0.15 * structure_conflict,
        ),
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
    score_result = score_consensus(
        image_id=image_id, labels=labels, model_outputs=model_outputs
    )
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
    seal_selection: SealSelectionDecision | None,
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

    if seal_selection is not None:
        selected_candidate = str(
            seal_selection.selected_candidate or ""
        ).strip().lower()
        if not selected_candidate or selected_candidate == "review":
            return _build_issue_driven_consensus(
                image_id=image_id,
                decision="review",
                reasons=[
                    "seal candidate selection requires manual review",
                    str(seal_selection.reason or "").strip() or "seal_selection_review",
                ],
                escalation_reasons=["seal_candidate_selection_review"],
                existing=existing,
            )
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="accepted",
            reasons=[
                f"seal candidate selected by second-stage adjudication: {selected_candidate}",
                str(seal_selection.reason or "").strip() or "seal_candidate_selected",
            ],
            escalation_reasons=[],
            existing=existing,
        )

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
            reasons=[
                "stamp mode requires both models to produce parsable labels before auto-accept"
            ],
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

    unresolved_issues = _find_unresolved_stamp_issues(
        issues=issues, patch_decisions=patch_decisions
    )
    if unresolved_issues:
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="review",
            reasons=["seal issues remain unresolved after second-stage adjudication"]
            + unresolved_issues,
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


def _select_issue_driven_seal_document(
    image_id: str,
    selection: SealSelectionDecision | None,
    mineru_document: CanonicalDocument,
    mineru_output: ModelOutput | None,
    selected_role: str | None,
    selected_document: CanonicalDocument | None,
    selected_output: ModelOutput | None,
    qwen_document: CanonicalDocument,
) -> CanonicalDocument | None:
    if selection is None:
        return None

    normalized_role = str(selection.selected_candidate or "").strip().lower()
    if not normalized_role or normalized_role == "review":
        return None

    if normalized_role == "mineru":
        source_document = mineru_document
        source_output = mineru_output
    else:
        if selected_document is None or normalized_role != str(selected_role or "").strip().lower():
            return None
        source_document = selected_document
        source_output = selected_output

    selected_document_copy = source_document.model_copy(deep=True)
    selected_document_copy.raw_metadata = dict(selected_document_copy.raw_metadata or {})
    selected_document_copy.raw_metadata["selected_output_role"] = normalized_role
    selected_document_copy.raw_metadata["selected_model_name"] = (
        source_output.model_name
        if source_output is not None and str(source_output.model_name or "").strip()
        else source_document.source
    )
    selected_document_copy.raw_metadata["selected_vendor"] = (
        source_output.vendor
        if source_output is not None and str(source_output.vendor or "").strip()
        else source_document.source
    )
    selected_document_copy.raw_metadata["selected_source_type"] = (
        source_output.source_type
        if source_output is not None and str(source_output.source_type or "").strip()
        else "final"
    )
    selected_document_copy.raw_metadata["selected_by"] = "seal_candidate_selection"
    selected_document_copy.raw_metadata["selected_image_id"] = image_id
    if str(selection.reason or "").strip():
        selected_document_copy.raw_metadata["selection_reason"] = str(
            selection.reason or ""
        ).strip()
    selected_document_copy.warnings = _deduplicate(
        selected_document_copy.warnings
        + _collect_document_warnings(mineru_document, qwen_document)
    )
    return selected_document_copy


def _override_flowchart_mode_consensus(
    image_id: str,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_label: ParsedLabel | None,
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
    issues: list[Issue],
    patch_decisions: list[PatchDecision],
    graph_fusion_result: Any | None,
    existing: ConsensusResult | None,
) -> ConsensusResult | None:
    labels = [label for label in (mineru_label, qwen_label) if label is not None]
    if not _is_issue_driven_flowchart_mode(
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        labels=labels,
        issues=issues,
    ):
        return existing

    del graph_fusion_result

    flowchart_issues = [
        issue
        for issue in issues
        if issue.issue_type
        in {"flowchart_graph_conflict", "flowchart_candidate_review"}
    ]

    adopted_qwen = any(
        str(decision.decision or "").strip() == "use_qwen_fields"
        and str(decision.reason or "").strip() == "qwen_flowchart_preferred_on_conflict"
        for decision in patch_decisions
    )
    fallback_to_mineru = any(
        str(decision.decision or "").strip() == "keep_mineru"
        and str(decision.reason or "").strip() == "qwen_flowchart_incomplete"
        for decision in patch_decisions
    )

    if adopted_qwen:
        if _has_valid_flowchart_mermaid(qwen_document):
            return _build_issue_driven_consensus(
                image_id=image_id,
                decision="accepted",
                reasons=["flowchart conflicts detected and qwen flowchart was adopted"],
                escalation_reasons=[],
                existing=existing,
            )
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="review",
            reasons=[
                "flowchart conflicts detected and qwen was selected, but qwen final mermaid is missing"
            ],
            escalation_reasons=["qwen_selected_flowchart_missing_final_mermaid"],
            existing=existing,
        )

    if _has_valid_flowchart_mermaid(mineru_document):
        if not flowchart_issues:
            return _build_issue_driven_consensus(
                image_id=image_id,
                decision="accepted",
                reasons=["no flowchart graph conflicts detected"],
                escalation_reasons=[],
                existing=existing,
            )
        if fallback_to_mineru:
            return _build_issue_driven_consensus(
                image_id=image_id,
                decision="accepted",
                reasons=[
                    "flowchart conflicts detected but qwen flowchart was incomplete, fallback to mineru"
                ],
                escalation_reasons=[],
                existing=existing,
            )
        return _build_issue_driven_consensus(
            image_id=image_id,
            decision="accepted",
            reasons=[
                "flowchart final mermaid remains valid after heuristic comparison"
            ],
            escalation_reasons=[],
            existing=existing,
        )

    return _build_issue_driven_consensus(
        image_id=image_id,
        decision="review",
        reasons=[
            "flowchart detected but final mermaid is still missing after heuristic comparison"
        ],
        escalation_reasons=["flowchart_missing_final_mermaid"],
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
    return any(
        _is_seal_like_block(block)
        for block in mineru_document.blocks + qwen_document.blocks
    )


def _is_issue_driven_flowchart_mode(
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    labels: list[ParsedLabel],
    issues: list[Issue],
) -> bool:
    if any(
        issue.issue_type in {"flowchart_graph_conflict", "flowchart_candidate_review"}
        for issue in issues
    ):
        return True
    if any(label.image_type == "flowchart" for label in labels):
        return True
    return any(
        _is_flowchart_like_block(block)
        for block in mineru_document.blocks + qwen_document.blocks
    )


def _select_issue_driven_flowchart_document(
    image_id: str,
    mineru_document: CanonicalDocument,
    qwen_document: CanonicalDocument,
    issues: list[Issue],
    patch_decisions: list[PatchDecision],
    mineru_output: ModelOutput | None,
    qwen_output: ModelOutput | None,
) -> CanonicalDocument | None:
    if not _is_issue_driven_flowchart_mode(
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        labels=[],
        issues=issues,
    ):
        return None

    adopted_qwen = any(
        str(decision.decision or "").strip() == "use_qwen_fields"
        and str(decision.reason or "").strip() == "qwen_flowchart_preferred_on_conflict"
        for decision in patch_decisions
    )
    selected_role = "qwen" if adopted_qwen else "mineru"
    selected_output = qwen_output if adopted_qwen else mineru_output
    source_document = qwen_document if adopted_qwen else mineru_document

    selected_document = source_document.model_copy(deep=True)
    selected_document.raw_metadata = dict(selected_document.raw_metadata or {})
    selected_document.raw_metadata["selected_output_role"] = selected_role
    selected_document.raw_metadata["selected_model_name"] = (
        selected_output.model_name
        if selected_output is not None and str(selected_output.model_name or "").strip()
        else source_document.source
    )
    selected_document.raw_metadata["selected_vendor"] = (
        selected_output.vendor
        if selected_output is not None and str(selected_output.vendor or "").strip()
        else source_document.source
    )
    selected_document.raw_metadata["selected_source_type"] = (
        selected_output.source_type
        if selected_output is not None and str(selected_output.source_type or "").strip()
        else "final"
    )
    selected_document.raw_metadata["selected_by"] = "flowchart_issue_resolution"
    selected_document.raw_metadata["selected_image_id"] = image_id
    selected_document.warnings = _deduplicate(
        selected_document.warnings
        + _collect_document_warnings(mineru_document, qwen_document)
    )
    return selected_document


def _is_seal_like_block(block: CanonicalBlock) -> bool:
    if block.type != "image":
        return False
    if str(block.sub_type or "").strip().lower() == "seal":
        return True
    return any(
        str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions
    )


def _is_flowchart_like_block(block: CanonicalBlock) -> bool:
    if block.type not in {"chart", "image"}:
        return False
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if block.structured_label.kind == "mermaid":
        return True
    return bool(block.flowchart_graph)


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


def _find_unresolved_flowchart_issues(
    mineru_document: CanonicalDocument,
    issues: list[Issue],
    patch_decisions: list[PatchDecision],
) -> list[str]:
    decision_lookup = {decision.issue_id: decision for decision in patch_decisions}
    block_lookup = {block.block_id: block for block in mineru_document.blocks}
    unresolved: list[str] = []
    failure_reasons = {"llm_patch_unavailable", "llm_patch_invalid_json"}

    for issue in issues:
        decision = decision_lookup.get(issue.issue_id)
        if decision is None:
            unresolved.append(f"missing_patch_decision:{issue.issue_id}")
            continue
        if str(decision.reason or "").strip() in failure_reasons:
            unresolved.append(f"patch_decision_unavailable:{issue.issue_id}")
            continue
        target_block_id = str(
            decision.target_block_id or issue.target_block_id or ""
        ).strip()
        target_block = block_lookup.get(target_block_id) if target_block_id else None
        if target_block is None:
            unresolved.append(f"missing_flowchart_target:{issue.issue_id}")
            continue
        if str(target_block.sub_type or "").strip().lower() != "flowchart":
            unresolved.append(f"target_not_flowchart:{issue.issue_id}")
            continue
        final_mermaid = str(target_block.content.get("content", "") or "").strip()
        if not looks_like_mermaid(final_mermaid):
            unresolved.append(f"target_missing_valid_mermaid:{issue.issue_id}")
    return unresolved


def _has_valid_flowchart_mermaid(document: CanonicalDocument) -> bool:
    for block in document.blocks:
        if not _is_flowchart_like_block(block):
            continue
        mermaid = str(block.content.get("content", "") or "").strip()
        if looks_like_mermaid(mermaid):
            return True
    return False


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
        caption_agreement=1.0
        if accepted
        else float(metrics.get("caption_agreement", 0.0)),
        structure_agreement=1.0
        if accepted
        else float(metrics.get("structure_agreement", 0.0)),
        seal_agreement=1.0 if accepted else float(metrics.get("seal_agreement", 1.0)),
        overall_score=1.0 if accepted else float(metrics.get("overall_score", 0.0)),
        evidence_score=1.0 if accepted else float(metrics.get("evidence_score", 0.0)),
        validator_score=1.0 if accepted else float(metrics.get("validator_score", 0.0)),
        hallucination_risk=0.0
        if accepted
        else float(metrics.get("hallucination_risk", 1.0)),
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


def _merge_ocr_regions(
    left: list[OcrRegion], right: list[OcrRegion]
) -> list[OcrRegion]:
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
