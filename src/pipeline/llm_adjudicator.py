from __future__ import annotations

import json
from time import sleep

from src.clients import BaseLocalClient
from src.normalizer import _extract_first_json_object, _strip_code_fences
from src.pipeline.flowchart_utils import normalize_mermaid_text
from src.schema import ImageTask, ModelOutput, Issue, PatchDecision


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
    if str(mode or "").strip().lower() == "flowchart_adjudication":
        return _build_flowchart_prompt_payload(issue)
    return issue.model_dump()


def _build_flowchart_prompt_payload(issue: Issue) -> dict[str, object]:
    candidate_payload = (
        issue.candidate_payload if isinstance(issue.candidate_payload, dict) else {}
    )
    graph_diff = candidate_payload.get("graph_diff")
    current_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("current_mermaid", "") or "")
    )
    reference_mermaid = normalize_mermaid_text(
        str(candidate_payload.get("reference_mermaid", "") or "")
    )
    focus_terms = _collect_flowchart_focus_terms(graph_diff)

    return {
        "issue_id": issue.issue_id,
        "issue_type": issue.issue_type,
        "target_block_id": issue.target_block_id,
        "page_idx": issue.page_idx,
        "reasons": issue.reasons,
        "current_block": _compact_block_summary(issue.mineru_block),
        "reference_block": _compact_block_summary(issue.qwen_block),
        "graph_diff": _compact_graph_diff(graph_diff),
        "current_excerpt": _build_mermaid_excerpt(current_mermaid, focus_terms),
        "reference_excerpt": _build_mermaid_excerpt(reference_mermaid, focus_terms),
    }


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


def _build_mermaid_excerpt(mermaid: str, focus_terms: list[str]) -> str:
    text = normalize_mermaid_text(mermaid)
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""

    header = lines[0].strip()
    body = lines[1:]
    selected_indexes: set[int] = set()

    lowered_terms = [term.lower() for term in focus_terms if term.strip()]
    if lowered_terms:
        for index, line in enumerate(body):
            lowered_line = line.lower()
            if any(term in lowered_line for term in lowered_terms):
                selected_indexes.add(index)
    if not selected_indexes:
        selected_indexes.update(range(min(len(body), 12)))

    selected_lines = [body[index] for index in sorted(selected_indexes)[:16]]
    excerpt_lines = [header] + selected_lines if header else selected_lines
    return "\n".join(line for line in excerpt_lines if str(line).strip())


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
