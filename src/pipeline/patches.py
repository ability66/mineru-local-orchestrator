from __future__ import annotations

from typing import Any

from src.pipeline.flowchart_utils import flowchart_graph_from_mermaid, looks_like_mermaid, normalize_mermaid_text
from src.pipeline.table_utils import is_html_table_like
from src.schema import CanonicalBlock, CanonicalDocument, OcrRegion, PatchDecision, StructuredLabel


def apply_patch_decisions(
    mineru_document: CanonicalDocument,
    issues: list[Any],
    patch_decisions: list[PatchDecision],
) -> CanonicalDocument:
    patched = mineru_document.model_copy(deep=True)
    issue_lookup = {str(issue.issue_id): issue for issue in issues}
    block_lookup = {block.block_id: block for block in patched.blocks}

    applied_issue_ids: list[str] = []
    for decision in patch_decisions:
        if decision.decision in {"keep_mineru", "reject_issue"}:
            continue
        issue = issue_lookup.get(decision.issue_id)
        target_block = block_lookup.get(str(decision.target_block_id or ""))
        if target_block is None and issue is not None:
            target_block = block_lookup.get(str(issue.target_block_id or ""))

        patch_payload = _resolve_patch_payload(issue=issue, decision=decision)
        if not patch_payload and decision.decision != "use_qwen_fields":
            continue

        if target_block is None:
            added_block = _build_block_from_issue(issue)
            if added_block is None or decision.decision not in {"keep_candidate", "merge", "add_qwen_block", "use_qwen_fields"}:
                continue
            _apply_patch_to_block(added_block, patch_payload)
            _append_block(patched, added_block)
            block_lookup[added_block.block_id] = added_block
            applied_issue_ids.append(decision.issue_id)
            continue

        _apply_patch_to_block(target_block, patch_payload)
        target_block.provenance["llm_patch_decision"] = decision.decision
        target_block.provenance["llm_patch_issue_id"] = decision.issue_id
        if decision.reason.strip():
            target_block.warnings = _deduplicate_texts(target_block.warnings + [decision.reason.strip()])
        applied_issue_ids.append(decision.issue_id)

    if applied_issue_ids:
        patched.raw_metadata["applied_issue_ids"] = applied_issue_ids
    patched.blocks.sort(key=lambda item: (item.page_idx, item.order_index, item.block_id))
    patched.page_count = max(patched.page_count, max((block.page_idx for block in patched.blocks), default=-1) + 1, 1)
    return patched


def _build_block_from_issue(issue: Any) -> CanonicalBlock | None:
    if issue is None or not isinstance(getattr(issue, "qwen_block", None), dict):
        return None
    block_payload = issue.qwen_block or {}
    try:
        block = CanonicalBlock(**block_payload)
    except Exception:
        return None
    block.provenance["added_from_llm_patch"] = True
    block.provenance["source_issue_id"] = getattr(issue, "issue_id", "")
    return block


def _append_block(document: CanonicalDocument, block: CanonicalBlock) -> None:
    same_page_blocks = [item for item in document.blocks if item.page_idx == block.page_idx]
    block.order_index = max((item.order_index for item in same_page_blocks), default=0) + 1
    document.blocks.append(block)


def _apply_patch_to_block(block: CanonicalBlock, patch: dict[str, Any]) -> None:
    if not isinstance(patch, dict):
        return
    if isinstance(patch.get("type"), str) and str(patch.get("type")).strip():
        block.type = str(patch.get("type")).strip()  # type: ignore[assignment]
    if "sub_type" in patch:
        value = str(patch.get("sub_type") or "").strip()
        block.sub_type = value or None
    if isinstance(patch.get("text"), str):
        block.text = str(patch.get("text") or "").strip()
    if isinstance(patch.get("content"), dict):
        block.content.update(_clean_content_dict(patch.get("content")))
    if isinstance(patch.get("flowchart_graph"), dict):
        block.flowchart_graph = patch.get("flowchart_graph")
    if isinstance(patch.get("ocr_regions"), list):
        incoming_regions = [_region_from_payload(item) for item in patch.get("ocr_regions") if isinstance(item, dict)]
        block.ocr_regions = _merge_regions(block.ocr_regions, [item for item in incoming_regions if item is not None])
    if isinstance(patch.get("visible_text"), list):
        block.visible_text = _deduplicate_texts(block.visible_text + [str(item).strip() for item in patch.get("visible_text") if str(item).strip()])

    _refresh_flowchart_payload(block)
    _refresh_block_semantics(block)

    if block.type in {"image", "chart", "table"} and "img_path" not in block.content:
        block.content["img_path"] = ""


def _resolve_patch_payload(issue: Any, decision: PatchDecision) -> dict[str, Any]:
    payload = dict(decision.patch or {})
    candidate_payload = issue.candidate_payload if issue is not None else None
    if not isinstance(candidate_payload, dict):
        return payload

    issue_type = str(getattr(issue, "issue_type", "") or "").strip()
    if decision.decision == "keep_candidate":
        candidate_patch = dict(candidate_payload.get("candidate_patch") or {})
        candidate_patch.update(payload)
        return candidate_patch
    if decision.decision == "use_qwen_fields":
        reference_patch = dict(candidate_payload.get("reference_patch") or {})
        reference_patch.update(payload)
        return reference_patch
    if issue_type == "flowchart_graph_conflict" and decision.decision == "merge":
        if "content" in payload or "flowchart_graph" in payload:
            reference_patch = dict(candidate_payload.get("reference_patch") or {})
            reference_patch.update(payload)
            return reference_patch
    return payload


def _refresh_flowchart_payload(block: CanonicalBlock) -> None:
    if str(block.sub_type or "").strip().lower() != "flowchart":
        return

    mermaid = normalize_mermaid_text(str(block.content.get("content", "") or "").strip())
    if looks_like_mermaid(mermaid):
        block.content["content"] = mermaid
        if block.flowchart_graph is None:
            derived_graph = flowchart_graph_from_mermaid(mermaid)
            if derived_graph is not None:
                block.flowchart_graph = derived_graph


def _refresh_block_semantics(block: CanonicalBlock) -> None:
    if block.type == "chart" and str(block.sub_type or "").strip().lower() == "flowchart":
        mermaid = str(block.content.get("content", "") or "").strip()
        if looks_like_mermaid(mermaid):
            block.structured_label = StructuredLabel(
                kind="mermaid",
                content=mermaid,
                format="mermaid",
                source="fused_graph" if block.flowchart_graph else "model",
            )
        elif block.flowchart_graph is not None:
            block.structured_label = StructuredLabel(
                kind="text",
                content=mermaid,
                format="plain_text" if mermaid else "none",
                source="fused_graph",
            )
        return

    if block.type == "table":
        table_body = str(block.content.get("table_body", "") or "").strip()
        if table_body:
            structured_format = "html" if is_html_table_like({"type": "table", "content": {"table_body": table_body}}) else "markdown"
            block.structured_label = StructuredLabel(
                kind="table",
                content=table_body,
                format=structured_format,  # type: ignore[arg-type]
                source="model",
            )
        return

    if (
        block.type == "chart"
        and str(block.sub_type or "").strip().lower() != "flowchart"
        and is_html_table_like(
            {
                "type": "chart",
                "sub_type": block.sub_type,
                "content": {"content": str(block.content.get("content", "") or "")},
                "visible_text": block.visible_text,
            }
        )
    ):
        block.sub_type = block.sub_type or "html_table"
        block.structured_label = StructuredLabel(
            kind="table",
            content=str(block.content.get("content", "") or ""),
            format="html",
            source="model",
        )


def _clean_content_dict(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            cleaned[key] = [item for item in value if item not in (None, "")]
        else:
            cleaned[key] = value
    return cleaned


def _region_from_payload(payload: dict[str, Any]) -> OcrRegion | None:
    text = str(payload.get("text", "") or "").strip()
    if not text:
        return None
    return OcrRegion(
        role=_normalize_role(payload.get("role")),
        text=text,
        bbox_hint=_normalize_bbox_hint(payload.get("bbox_hint")),
        confidence=_normalize_confidence(payload.get("confidence")),
    )


def _merge_regions(left: list[OcrRegion], right: list[OcrRegion]) -> list[OcrRegion]:
    merged: list[OcrRegion] = []
    seen: set[tuple[str, str]] = set()
    for region in list(left) + list(right):
        key = (str(region.role or "").strip(), str(region.text or "").strip())
        if key in seen:
            continue
        seen.add(key)
        merged.append(region)
    return merged


def _normalize_role(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"seal", "stamp", "印章", "公章"}:
        return "seal"
    if normalized in {"watermark", "水印"}:
        return "watermark"
    if normalized in {"footer", "页脚"}:
        return "footer"
    if normalized in {"body", "正文"}:
        return "body"
    if normalized in {"title", "标题"}:
        return "title"
    return "other"


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric >= 0.85:
            return "high"
        if numeric >= 0.5:
            return "medium"
        return "low"
    normalized = str(value or "").strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    try:
        numeric = float(normalized)
    except (TypeError, ValueError):
        return "medium"
    if numeric >= 0.85:
        return "high"
    if numeric >= 0.5:
        return "medium"
    return "low"


def _normalize_bbox_hint(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    numbers: list[float] = []
    for item in value:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            return None
    return numbers


def _deduplicate_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        normalized = "".join(text.split()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(text)
    return ordered
