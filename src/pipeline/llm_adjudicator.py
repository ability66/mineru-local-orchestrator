from __future__ import annotations

import json
from time import sleep
from typing import Any

from src.clients import BaseLocalClient
from src.normalizer import _extract_first_json_object, _strip_code_fences
from src.pipeline.flowchart_utils import (
    diff_flowchart_graphs,
    flowchart_graph_from_mermaid,
    looks_like_mermaid,
    normalize_mermaid_text,
)
from src.schema import (
    ImageTask,
    Issue,
    ModelOutput,
    PatchDecision,
    SealSelectionDecision,
)


def adjudicate_issues_with_llm(
    client: BaseLocalClient | None,
    image_task: ImageTask,
    prompt: str,
    issues: list[Issue],
    mode: str,
    retry: int = 0,
) -> tuple[list[PatchDecision], list[ModelOutput]]:
    if client is None or not issues:
        return [], []

    decisions: list[PatchDecision] = []
    outputs: list[ModelOutput] = []
    for issue in issues:
        prompt_payload = build_issue_prompt_payload(issue=issue, mode=mode)
        output = _call_with_retry(
            client=client,
            image_task=image_task,
            prompt=prompt,
            retry=retry,
            context={
                "mode": mode,
                "issue_payload": prompt_payload,
            },
        )
        if output is not None:
            outputs.append(output)
        decisions.append(_parse_patch_decision(issue=issue, output=output))
    return decisions, outputs


def adjudicate_seal_candidates_with_llm(
    client: BaseLocalClient | None,
    image_task: ImageTask,
    prompt: str,
    selection_payload: dict[str, object],
    retry: int = 0,
) -> tuple[SealSelectionDecision, ModelOutput | None]:
    if client is None:
        return (
            SealSelectionDecision(
                selected_candidate="review",
                reason="seal_candidate_selection_client_unavailable",
                confidence="low",
            ),
            None,
        )

    output = _call_with_retry(
        client=client,
        image_task=image_task,
        prompt=prompt,
        retry=retry,
        context={
            "mode": "seal_adjudication",
            "selection_payload": selection_payload,
        },
    )
    return _parse_seal_selection_decision(
        selection_payload=selection_payload,
        output=output,
    ), output


def _call_with_retry(
    client: BaseLocalClient,
    image_task: ImageTask,
    prompt: str,
    retry: int,
    context: dict[str, object],
) -> ModelOutput | None:
    attempts = max(0, retry) + 1
    last_output: ModelOutput | None = None
    for attempt in range(attempts):
        output = client.analyze(image_task=image_task, prompt=prompt, context=context)
        last_output = output
        if output.success:
            return output
        if attempt < attempts - 1:
            sleep(2**attempt)
    return last_output


def _parse_patch_decision(issue: Issue, output: ModelOutput | None) -> PatchDecision:
    fallback = PatchDecision(
        issue_id=issue.issue_id,
        target_block_id=issue.target_block_id,
        decision="keep_mineru",
        patch={},
        reason="llm_patch_unavailable",
    )
    if output is None or not output.success:
        if output is not None and str(output.error or "").strip():
            fallback.reason = str(output.error or "").strip()
        return fallback

    payload = _parse_json_object(output.raw_text)
    if not isinstance(payload, dict):
        payload = _parse_json_object(_extract_raw_text_from_parsed(output.parsed))
    if not isinstance(payload, dict):
        fallback.reason = "llm_patch_invalid_json"
        return fallback

    decision = _normalize_decision(payload.get("decision"))
    patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
    reason = str(payload.get("reason", "") or "").strip()
    decision, patch, reason = _validate_patch_decision(
        issue=issue,
        decision=decision,
        patch=patch,
        reason=reason,
    )
    decision, patch, reason = _reject_false_positive_flowchart_conflict(
        issue=issue,
        decision=decision,
        patch=patch,
        reason=reason,
    )
    return PatchDecision(
        issue_id=str(payload.get("issue_id") or issue.issue_id),
        target_block_id=str(
            payload.get("target_block_id") or issue.target_block_id or ""
        )
        or None,
        decision=decision,
        patch=patch,
        reason=reason or "llm_patch_applied",
    )


def _parse_seal_selection_decision(
    selection_payload: dict[str, object],
    output: ModelOutput | None,
) -> SealSelectionDecision:
    valid_candidates = {
        str(candidate.get("candidate_id", "")).strip()
        for candidate in selection_payload.get("candidates", [])
        if isinstance(candidate, dict) and str(candidate.get("candidate_id", "")).strip()
    }
    fallback = SealSelectionDecision(
        selected_candidate="review",
        reason="llm_selection_unavailable",
        confidence="low",
    )
    if output is None or not output.success:
        if output is not None and str(output.error or "").strip():
            fallback.reason = str(output.error or "").strip()
        return fallback

    payload = _parse_json_object(output.raw_text)
    if not isinstance(payload, dict):
        payload = _parse_json_object(_extract_raw_text_from_parsed(output.parsed))
    if not isinstance(payload, dict):
        fallback.reason = "llm_selection_invalid_json"
        return fallback

    selected_candidate = _normalize_selected_candidate(
        value=payload.get("selected_candidate")
        or payload.get("selected_role")
        or payload.get("candidate_id")
        or payload.get("decision"),
        valid_candidates=valid_candidates,
    )
    reason = str(payload.get("reason", "") or "").strip()
    confidence = _normalize_selection_confidence(payload.get("confidence"))
    return SealSelectionDecision(
        selected_candidate=selected_candidate,
        reason=reason or "llm_candidate_selected",
        confidence=confidence,
    )


def _parse_json_object(raw_text: str) -> object:
    cleaned_text = _strip_code_fences(raw_text)
    json_text, _ = _extract_first_json_object(cleaned_text)
    if json_text is None:
        return None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def build_issue_prompt_payload(issue: Issue, mode: str) -> dict[str, object]:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "flowchart_adjudication":
        return _build_flowchart_prompt_payload(issue)
    if normalized_mode == "table_adjudication":
        return _build_table_prompt_payload(issue)
    return issue.model_dump()


def build_seal_selection_prompt_payload(
    selection_payload: dict[str, object],
) -> dict[str, object]:
    return dict(selection_payload)


def _build_flowchart_prompt_payload(issue: Issue) -> dict[str, object]:
    candidate_payload = (
        issue.candidate_payload if isinstance(issue.candidate_payload, dict) else {}
    )
    review_mode = (
        str(candidate_payload.get("review_mode", "") or "").strip() or "disagreement"
    )
    graph_diff = candidate_payload.get("graph_diff")
    current_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("current_mermaid", "") or "")
    )
    reference_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("reference_mermaid", "") or "")
    )
    focus_terms = _collect_flowchart_focus_terms(graph_diff)
    ocr_reference_sources = _normalize_ocr_reference_sources(
        candidate_payload.get("ocr_reference_sources")
    )
    ocr_reference_texts = _deduplicate_texts(
        _as_text_list(candidate_payload.get("ocr_reference_texts"))
        + [
            text
            for source in ocr_reference_sources
            for text in list(source.get("ocr_reference_texts") or [])
        ]
    )
    ocr_reference_model = str(
        candidate_payload.get("ocr_reference_model", "") or ""
    ).strip()
    ocr_reference_models = _deduplicate_texts(
        [
            str(source.get("reference_model_name", "") or "").strip()
            or str(source.get("reference_model_role", "") or "").strip()
            for source in ocr_reference_sources
        ]
    )

    payload = {
        "issue_id": issue.issue_id,
        "issue_type": issue.issue_type,
        "review_mode": review_mode,
        "target_block_id": issue.target_block_id,
        "page_idx": issue.page_idx,
        "reasons": issue.reasons,
        "current_block": _compact_block_summary(issue.mineru_block),
        "reference_block": _compact_block_summary(issue.qwen_block),
        "graph_diff": _compact_graph_diff(graph_diff),
        "current_mermaid": current_mermaid,
        "reference_mermaid": reference_mermaid,
        "focus_terms": focus_terms,
        "thinking_mode": "disabled_requested_for_flowchart_adjudication",
    }
    if ocr_reference_texts:
        if ocr_reference_sources:
            payload["ocr_reference_sources"] = ocr_reference_sources
        if ocr_reference_models:
            payload["ocr_reference_models"] = ocr_reference_models
        payload["ocr_reference_model"] = (
            ocr_reference_model
            or (
                ocr_reference_models[0]
                if len(ocr_reference_models) == 1
                else "multi_source"
            )
            or "auxiliary"
        )
        payload["ocr_reference_usage"] = (
            "仅用于节点/连线文字校对，不可用于结构推断、节点增删或边关系改写"
        )
        payload["ocr_reference_texts"] = ocr_reference_texts[:40]
    return payload


def _build_table_prompt_payload(issue: Issue) -> dict[str, object]:
    candidate_payload = (
        issue.candidate_payload if isinstance(issue.candidate_payload, dict) else {}
    )
    candidates = candidate_payload.get("candidates")
    pairwise_scores = candidate_payload.get("pairwise_scores")
    pairwise_matrix = candidate_payload.get("pairwise_matrix")
    consensus_diagnostics = candidate_payload.get("consensus_diagnostics")
    review_mode = str(
        candidate_payload.get("review_mode", "") or "table_disagreement"
    ).strip() or "table_disagreement"
    payload = {
        "issue_id": issue.issue_id,
        "issue_type": issue.issue_type,
        "review_mode": review_mode,
        "target_block_id": issue.target_block_id,
        "page_idx": issue.page_idx,
        "reasons": list(issue.reasons or []),
        "current_block": _compact_block_summary(issue.mineru_block),
        "reference_block": _compact_block_summary(issue.qwen_block),
        "candidates": candidates if isinstance(candidates, list) else [],
        "pairwise_scores": pairwise_scores if isinstance(pairwise_scores, list) else [],
        "pairwise_matrix": pairwise_matrix if isinstance(pairwise_matrix, dict) else {},
        "consensus_diagnostics": (
            consensus_diagnostics
            if isinstance(consensus_diagnostics, dict)
            else {}
        ),
        "thinking_mode": "disabled_requested_for_table_adjudication",
    }
    for key in (
        "branch_mode",
        "forced_second_pass",
        "must_output_final_table",
        "must_include_caption",
        "final_table_target",
        "task_instruction",
    ):
        value = candidate_payload.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    reference_model_role = str(
        candidate_payload.get("reference_model_role", "") or ""
    ).strip()
    reference_model_name = str(
        candidate_payload.get("reference_model_name", "") or ""
    ).strip()
    if reference_model_role:
        payload["reference_model_role"] = reference_model_role
    if reference_model_name:
        payload["reference_model_name"] = reference_model_name
    return payload


def _compact_block_summary(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    summary: dict[str, object] = {}
    for key in ("block_id", "page_idx", "type", "sub_type"):
        value = payload.get(key)
        if value not in (None, ""):
            summary[key] = value
    return summary or None


def _compact_graph_diff(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    compact: dict[str, object] = {}
    diff_kind = str(payload.get("diff_kind", "") or "").strip()
    if diff_kind:
        compact["diff_kind"] = diff_kind
    node_key = str(payload.get("node_key", "") or "").strip()
    if node_key:
        compact["node_key"] = node_key
    edge_key = str(payload.get("edge_key", "") or "").strip()
    if edge_key:
        compact["edge_key"] = edge_key
    reference_node = _compact_graph_item(payload.get("reference_node"))
    if reference_node is not None:
        compact["reference_node"] = reference_node
    current_node = _compact_graph_item(payload.get("current_node"))
    if current_node is not None:
        compact["current_node"] = current_node
    reference_edge = _compact_graph_item(payload.get("reference_edge"))
    if reference_edge is not None:
        compact["reference_edge"] = reference_edge
    current_edge = _compact_graph_item(payload.get("current_edge"))
    if current_edge is not None:
        compact["current_edge"] = current_edge
    return compact or None


def _compact_graph_item(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    compact: dict[str, object] = {}
    for key in ("node_id", "text", "shape", "source", "target", "label"):
        value = payload.get(key)
        if value not in (None, ""):
            compact[key] = value
    return compact or None


def _collect_flowchart_focus_terms(graph_diff: object) -> list[str]:
    if not isinstance(graph_diff, dict):
        return []
    terms: list[str] = []
    for key in ("node_key", "edge_key"):
        value = str(graph_diff.get(key, "") or "").strip()
        if value:
            terms.extend(
                part.strip()
                for part in value.replace("|", " ").replace("->", " ").split()
                if part.strip()
            )
    for item_key in (
        "reference_node",
        "current_node",
        "reference_edge",
        "current_edge",
    ):
        item = graph_diff.get(item_key)
        if not isinstance(item, dict):
            continue
        for value_key in ("node_id", "text", "source", "target", "label"):
            value = str(item.get(value_key, "") or "").strip()
            if value:
                terms.append(value)
    return _deduplicate_texts(terms)


def _build_mermaid_excerpt(
    mermaid: str,
    focus_terms: list[str],
    review_mode: str = "disagreement",
) -> str:
    text = normalize_mermaid_text(mermaid)
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""

    header = lines[0].strip()
    body = lines[1:]
    if len(body) <= 24:
        return text

    selected_indexes: set[int] = set()

    lowered_terms = [term.lower() for term in focus_terms if term.strip()]
    if lowered_terms:
        for index, line in enumerate(body):
            lowered_line = line.lower()
            if any(term in lowered_line for term in lowered_terms):
                for candidate in range(max(0, index - 1), min(len(body), index + 2)):
                    selected_indexes.add(candidate)
    if not selected_indexes:
        selected_indexes.update(range(min(len(body), 8)))
        selected_indexes.update(range(max(0, len(body) - 8), len(body)))
    elif str(review_mode or "").strip().lower() == "disagreement":
        selected_indexes.update(range(min(len(body), 2)))
        selected_indexes.update(range(max(0, len(body) - 2), len(body)))

    selected_lines = _select_excerpt_lines(
        body=body,
        selected_indexes=selected_indexes,
        max_lines=20,
    )
    excerpt_lines = [header] + selected_lines if header else selected_lines
    return "\n".join(line for line in excerpt_lines if str(line).strip())


def _select_excerpt_lines(
    body: list[str],
    selected_indexes: set[int],
    max_lines: int,
) -> list[str]:
    ordered_indexes = sorted(
        index for index in selected_indexes if 0 <= index < len(body)
    )
    if len(ordered_indexes) <= max_lines:
        return [body[index] for index in ordered_indexes]

    pinned_indexes = sorted(
        {
            *range(min(len(body), 2)),
            *range(max(0, len(body) - 2), len(body)),
        }
    )
    remaining_slots = max(0, max_lines - len(pinned_indexes))
    middle_indexes = [
        index for index in ordered_indexes if index not in set(pinned_indexes)
    ]
    trimmed_middle = middle_indexes[:remaining_slots]
    final_indexes = sorted({*pinned_indexes, *trimmed_middle})
    return [body[index] for index in final_indexes]


def _validate_patch_decision(
    issue: Issue,
    decision: str,
    patch: dict[str, Any],
    reason: str,
) -> tuple[str, dict[str, Any], str]:
    if issue.issue_type != "flowchart_graph_conflict":
        return decision, patch, reason
    if decision != "merge":
        return decision, patch, reason

    candidate_payload = (
        issue.candidate_payload if isinstance(issue.candidate_payload, dict) else {}
    )
    review_mode = str(candidate_payload.get("review_mode", "") or "disagreement")
    if review_mode.strip().lower() == "second_pass":
        return "keep_mineru", {}, "llm_patch_merge_not_allowed_for_second_pass"

    content_payload = patch.get("content") if isinstance(patch.get("content"), dict) else {}
    proposed_mermaid = normalize_mermaid_text(
        str(content_payload.get("content", "") or "")
    )
    if not looks_like_mermaid(proposed_mermaid):
        return "keep_mineru", {}, "llm_patch_invalid_flowchart_merge"

    current_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("current_mermaid", "") or "")
    )
    reference_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("reference_mermaid", "") or "")
    )
    evidence_nodes, evidence_edges = _merge_flowchart_signatures(
        current_mermaid,
        reference_mermaid,
    )
    proposed_nodes, proposed_edges = _collect_flowchart_signatures(proposed_mermaid)
    novel_nodes = proposed_nodes - evidence_nodes
    novel_edges = proposed_edges - evidence_edges
    if len(novel_nodes) > 1 or len(novel_edges) > 2:
        return "keep_mineru", {}, "llm_patch_overreach_on_disagreement"
    return decision, patch, reason


def _reject_false_positive_flowchart_conflict(
    issue: Issue,
    decision: str,
    patch: dict[str, Any],
    reason: str,
) -> tuple[str, dict[str, Any], str]:
    if issue.issue_type != "flowchart_graph_conflict":
        return decision, patch, reason

    candidate_payload = (
        issue.candidate_payload if isinstance(issue.candidate_payload, dict) else {}
    )
    review_mode = str(candidate_payload.get("review_mode", "") or "disagreement")
    if review_mode.strip().lower() != "disagreement":
        return decision, patch, reason

    current_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("current_mermaid", "") or "")
    )
    reference_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("reference_mermaid", "") or "")
    )
    if not looks_like_mermaid(current_mermaid) or not looks_like_mermaid(reference_mermaid):
        return decision, patch, reason

    current_graph = flowchart_graph_from_mermaid(current_mermaid)
    reference_graph = flowchart_graph_from_mermaid(reference_mermaid)
    if diff_flowchart_graphs(current_graph, reference_graph) == []:
        return "reject_issue", {}, "flowchart_conflict_false_positive"
    return decision, patch, reason


def _merge_flowchart_signatures(*mermaid_texts: str) -> tuple[set[str], set[str]]:
    node_signatures: set[str] = set()
    edge_signatures: set[str] = set()
    for mermaid in mermaid_texts:
        nodes, edges = _collect_flowchart_signatures(mermaid)
        node_signatures.update(nodes)
        edge_signatures.update(edges)
    return node_signatures, edge_signatures


def _collect_flowchart_signatures(mermaid: str) -> tuple[set[str], set[str]]:
    graph_payload = flowchart_graph_from_mermaid(mermaid)
    if not isinstance(graph_payload, dict):
        return set(), set()

    node_id_to_signature: dict[str, str] = {}
    node_signatures: set[str] = set()
    for item in graph_payload.get("nodes", []):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", "") or "").strip()
        node_text = str(item.get("text", "") or "").strip()
        node_signature = _normalize_graph_signature_text(node_text) or (
            f"id:{_normalize_graph_signature_text(node_id)}"
            if node_id
            else ""
        )
        if not node_signature:
            continue
        node_signatures.add(node_signature)
        if node_id:
            node_id_to_signature[node_id] = node_signature

    edge_signatures: set[str] = set()
    for item in graph_payload.get("edges", []):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source", "") or "").strip()
        target_id = str(item.get("target", "") or "").strip()
        if not source_id or not target_id:
            continue
        source_signature = node_id_to_signature.get(
            source_id,
            f"id:{_normalize_graph_signature_text(source_id)}",
        )
        target_signature = node_id_to_signature.get(
            target_id,
            f"id:{_normalize_graph_signature_text(target_id)}",
        )
        edge_label = _normalize_graph_signature_text(item.get("label"))
        edge_signatures.add(
            f"{source_signature}|{edge_label}|{target_signature}"
        )
    return node_signatures, edge_signatures


def _normalize_graph_signature_text(value: object) -> str:
    return "".join(str(value or "").split()).lower()


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [
            str(item or "").strip()
            for item in value
            if str(item or "").strip()
        ]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_ocr_reference_sources(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    normalized_sources: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        texts = _as_text_list(item.get("ocr_reference_texts"))
        if not texts:
            continue
        role = str(item.get("reference_model_role", "") or "").strip()
        model_name = str(item.get("reference_model_name", "") or "").strip()
        normalized_sources.append(
            {
                "reference_model_role": role or "candidate",
                "reference_model_name": model_name or role or "candidate",
                "ocr_reference_texts": _deduplicate_texts(texts)[:20],
            }
        )
    return normalized_sources


def _extract_raw_text_from_parsed(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if isinstance(message, dict):
        for key in ("content", "text", "output_text"):
            extracted = _extract_text_like_payload(message.get(key))
            if extracted:
                return extracted
    for key in ("text", "output_text"):
        extracted = _extract_text_like_payload(first_choice.get(key))
        if extracted:
            return extracted
    return _extract_text_like_payload(payload.get("output_text"))


def _extract_text_like_payload(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "content", "value", "output_text"):
            extracted = _extract_text_like_payload(payload.get(key))
            if extracted:
                return extracted
        return ""
    if isinstance(payload, list):
        fragments: list[str] = []
        for item in payload:
            extracted = _extract_text_like_payload(item)
            if extracted:
                fragments.append(extracted)
        return "\n".join(fragment for fragment in fragments if fragment)
    return ""


def _deduplicate_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _normalize_decision(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {
        "keep_mineru",
        "keep_candidate",
        "use_qwen_fields",
        "merge",
        "add_qwen_block",
        "reject_issue",
    }:
        return normalized
    if normalized in {"keep", "preserve"}:
        return "keep_mineru"
    if normalized in {"keep_fusion", "keep_candidate_patch", "keep_graph_candidate"}:
        return "keep_candidate"
    if normalized in {
        "use_qwen",
        "use_qwen_patch",
        "use_reference",
        "adopt_qwen",
        "adopt_reference",
    }:
        return "use_qwen_fields"
    return "keep_mineru"


def _normalize_selected_candidate(
    value: object,
    valid_candidates: set[str],
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in valid_candidates:
        return normalized
    if normalized in {
        "review",
        "manual_review",
        "human_review",
        "uncertain",
        "unknown",
        "none",
        "all_wrong",
        "reject_all",
        "no_candidate",
    }:
        return "review"
    if normalized in {"keep", "keep_mineru", "use_mineru", "adopt_mineru"}:
        return "mineru" if "mineru" in valid_candidates else "review"
    if normalized in {"keep_candidate", "use_reference", "reference", "candidate"}:
        non_mineru = sorted(candidate for candidate in valid_candidates if candidate != "mineru")
        return non_mineru[0] if len(non_mineru) == 1 else "review"
    return "review"


def _normalize_selection_confidence(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return "low"
