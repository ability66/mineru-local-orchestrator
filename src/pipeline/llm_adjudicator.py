from __future__ import annotations

import json
from time import sleep

from src.clients import BaseLocalClient
from src.normalizer import _extract_first_json_object, _strip_code_fences
from src.schema import ImageTask, ModelOutput, Issue, PatchDecision


def adjudicate_issues_with_llm(
    client: BaseLocalClient | None,
    image_task: ImageTask,
    prompt: str,
    issues: list[Issue],
    retry: int = 0,
) -> tuple[list[PatchDecision], list[ModelOutput]]:
    if client is None or not issues:
        return [], []

    decisions: list[PatchDecision] = []
    outputs: list[ModelOutput] = []
    for issue in issues:
        output = _call_with_retry(
            client=client,
            image_task=image_task,
            prompt=prompt,
            retry=retry,
            context={
                "mode": "seal_adjudication",
                "issue_payload": issue.model_dump(),
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
        fallback.reason = "llm_patch_invalid_json"
        return fallback

    decision = _normalize_decision(payload.get("decision"))
    patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
    reason = str(payload.get("reason", "") or "").strip()
    return PatchDecision(
        issue_id=str(payload.get("issue_id") or issue.issue_id),
        target_block_id=str(payload.get("target_block_id") or issue.target_block_id or "") or None,
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


def _normalize_decision(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"keep_mineru", "use_qwen_fields", "merge", "add_qwen_block", "reject_issue"}:
        return normalized
    if normalized in {"keep", "preserve"}:
        return "keep_mineru"
    if normalized in {"use_qwen", "use_qwen_patch"}:
        return "use_qwen_fields"
    return "keep_mineru"
