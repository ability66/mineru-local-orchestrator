from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, desc=None):  # type: ignore[no-redef]
        del desc
        return iterable


from src.clients import CLIENT_REGISTRY, BaseLocalClient
from src.flowvqa_eval import (
    build_flowvqa_eval_payload,
    extract_mermaid_from_document,
    find_flowvqa_reference,
    validate_flowvqa_root,
)
from src.image_loader import load_image_tasks
from src.pipeline.adjudicator import (
    adjudicate_documents,
    analyze_table_bundles,
    pick_table_reference_bundle,
)
from src.pipeline.issues import (
    build_chart_table_second_pass_issues,
    build_table_issue,
    detect_flowchart_second_pass_issues,
    detect_flowchart_issues,
    detect_seal_issues,
)
from src.pipeline.llm_adjudicator import (
    adjudicate_seal_candidates_with_llm,
    adjudicate_issues_with_llm,
    build_issue_prompt_payload,
    build_seal_selection_prompt_payload,
)
from src.pipeline.normalizers import (
    derive_label_from_document,
    normalize_mineru_payload,
    normalize_paddle_payload,
    normalize_qwen_payload,
)
from src.pipeline.patches import apply_patch_decisions
from src.pipeline.table_utils import is_table_like
from src.projection import project_document_for_single_block_view
from src.prompt_builder import load_prompt
from src.render_compare_dashboard import generate_compare_dashboard
from src.render_mermaid_compare import generate_compare_page
from src.pipeline.flowchart_utils import looks_like_mermaid, normalize_mermaid_text
from src.seal_utils import primary_seal_text
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    ImageTask,
    ModelOutput,
    PatchDecision,
    ParsedLabel,
    SealSelectionDecision,
)
from src.writer import (
    append_summary_record,
    clear_previous_outputs,
    initialize_summary_file,
    write_image_result,
)

_TABLE_ARTIFACT_PLACEHOLDER_REASONS = {
    "only one parsable label",
    "single model result cannot be auto-accepted",
    "overall consensus score below acceptance threshold",
    "evidence score below acceptance threshold",
    "validator score below acceptance threshold",
    "hallucination risk above acceptance threshold",
}
_TABLE_ARTIFACT_PLACEHOLDER_ESCALATIONS = {
    "single_model_result",
    "low_consensus_score",
    "low_evidence_score",
    "low_validator_score",
    "high_hallucination_risk",
}
_TABLE_REVIEW_REASON_MESSAGES = {
    "severe_structure_or_formula_conflict": "table severe structure or formula conflict",
    "pairwise_similarity_matrix_empty": "table pairwise similarity matrix is empty",
    "no_stable_table_consensus": "table candidates do not form stable consensus",
    "chart_table_requires_qwen_second_pass": "chart table branch always escalates to qwen second-stage adjudication",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local MinerU + Qwen orchestration for chart/flowchart/stamp parsing."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models-config", type=Path, default=Path("configs/models.local.yaml")
    )
    parser.add_argument(
        "--prompts-config", type=Path, default=Path("configs/prompts.yaml")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--manual-compare-mode", action="store_true")
    parser.add_argument(
        "--flowvqa-root",
        type=Path,
        default=None,
        help="Local clone root of the FlowVQA repository. When set, matching image_ids will attach gold Mermaid and evaluation metrics.",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load_model_configs(config_path: Path) -> list[dict[str, Any]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load model configs")
    data = yaml.safe_load(raw_text) or {}
    models = data.get("models")
    if not isinstance(models, list):
        raise ValueError("Model config must contain a 'models' list")
    return [item for item in models if isinstance(item, dict)]


def build_clients(
    model_configs: list[dict[str, Any]], request_timeout: int
) -> list[BaseLocalClient]:
    clients: list[BaseLocalClient] = []
    for model_config in model_configs:
        if not bool(model_config.get("enabled", False)):
            continue
        provider = str(model_config.get("provider", "")).strip().lower()
        model_name = str(model_config.get("name", "")).strip()
        client_class = CLIENT_REGISTRY.get(provider)
        if client_class is None or not model_name:
            print(f"Skipping unsupported model config: {model_config}")
            continue
        if "timeout" not in model_config:
            model_config["timeout"] = request_timeout
        clients.append(client_class(model_name=model_name, config=model_config))
    return clients


def pick_client(
    clients: list[BaseLocalClient], provider_prefix: str
) -> BaseLocalClient | None:
    for client in clients:
        provider = str(client.config.get("provider", "")).strip().lower()
        if provider.startswith(provider_prefix):
            return client
    return None


def _client_role(client: BaseLocalClient | None) -> str:
    if client is None:
        return ""
    configured = str(client.config.get("role", "") or "").strip().lower()
    if configured:
        return configured
    provider = str(client.config.get("provider", "") or "").strip().lower()
    if provider.startswith("minerupro"):
        return "mineru"
    if provider.startswith("paddle"):
        return "paddle"
    if provider.startswith("glm"):
        return "glm"
    if provider.startswith("qwen"):
        return "judge"
    return provider


def pick_role_client(
    clients: list[BaseLocalClient],
    role: str,
) -> BaseLocalClient | None:
    normalized_role = str(role or "").strip().lower()
    for client in clients:
        if _client_role(client) == normalized_role:
            return client
    return None


def call_with_retry(
    client: BaseLocalClient | None,
    image_task: ImageTask,
    prompt: str,
    retry: int,
    context: dict[str, Any] | None = None,
) -> ModelOutput | None:
    if client is None:
        return None

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


def empty_document(image_task: ImageTask, source: str) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=image_task.image_id,
        source=source,
        backend="empty",
        page_count=1,
        blocks=[],
        warnings=["source_call_failed_or_not_configured"],
        raw_metadata={},
    )


def build_stage2_records(
    issues: list[Any],
    outputs: list[ModelOutput],
    patch_decisions: list[Any],
    prompt: str,
    mode: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, issue in enumerate(issues):
        output = outputs[index] if index < len(outputs) else None
        decision = patch_decisions[index] if index < len(patch_decisions) else None
        parsed_payload = output.parsed if output is not None else None
        usage = _extract_usage(parsed_payload)
        records.append(
            {
                "mode": mode,
                "issue_id": issue.issue_id,
                "issue_type": issue.issue_type,
                "target_block_id": issue.target_block_id,
                "prompt": prompt,
                "issue_payload": build_issue_prompt_payload(issue=issue, mode=mode),
                "success": bool(output.success) if output is not None else False,
                "error": output.error
                if output is not None
                else "missing_stage2_output",
                "latency_ms": output.latency_ms if output is not None else None,
                "raw_text": output.raw_text if output is not None else "",
                "usage": usage,
                "finish_reason": _extract_finish_reason(parsed_payload),
                "thinking_mode": _extract_thinking_mode(parsed_payload),
                "request_control": _extract_request_control(parsed_payload),
                "patch_decision": decision.model_dump()
                if decision is not None
                else None,
            }
        )
    return records


def build_stage2_selection_record(
    selection_payload: dict[str, Any],
    output: ModelOutput | None,
    selection_decision: SealSelectionDecision | None,
    prompt: str,
    mode: str,
) -> dict[str, Any]:
    parsed_payload = output.parsed if output is not None else None
    usage = _extract_usage(parsed_payload)
    return {
        "mode": mode,
        "prompt": prompt,
        "selection_payload": build_seal_selection_prompt_payload(selection_payload),
        "success": bool(output.success) if output is not None else False,
        "error": output.error if output is not None else "missing_stage2_output",
        "latency_ms": output.latency_ms if output is not None else None,
        "raw_text": output.raw_text if output is not None else "",
        "usage": usage,
        "finish_reason": _extract_finish_reason(parsed_payload),
        "thinking_mode": _extract_thinking_mode(parsed_payload),
        "request_control": _extract_request_control(parsed_payload),
        "selection_decision": selection_decision.model_dump()
        if selection_decision is not None
        else None,
    }


def _extract_usage(parsed_payload: Any) -> dict[str, Any] | None:
    if not isinstance(parsed_payload, dict):
        return None
    usage = parsed_payload.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _extract_request_control(parsed_payload: Any) -> dict[str, Any] | None:
    if not isinstance(parsed_payload, dict):
        return None
    control = parsed_payload.get("_request_control")
    if not isinstance(control, dict):
        return None
    return dict(control)


def _extract_thinking_mode(parsed_payload: Any) -> str | None:
    control = _extract_request_control(parsed_payload)
    if not isinstance(control, dict):
        return None
    value = str(control.get("thinking_mode", "") or "").strip()
    return value or None


def _extract_finish_reason(parsed_payload: Any) -> str | None:
    if not isinstance(parsed_payload, dict):
        return None
    choices = parsed_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    value = str(first_choice.get("finish_reason", "") or "").strip()
    return value or None


def _normalize_primary_output(
    image_task: ImageTask,
    client: BaseLocalClient | None,
    output: ModelOutput | None,
) -> tuple[ModelOutput | None, CanonicalDocument, Any | None]:
    if client is None or output is None:
        return (
            None,
            empty_document(image_task=image_task, source="primary_unconfigured"),
            None,
        )

    provider = str(client.config.get("provider", "")).strip().lower()
    role = _client_role(client)
    if role == "mineru" or provider.startswith("minerupro"):
        normalized_output, document, label = normalize_mineru_payload(
            image_task=image_task,
            model_output=output,
        )
        return normalized_output, document, label

    if role == "paddle" or provider.startswith("paddle"):
        normalized_output, document, label = normalize_paddle_payload(
            image_task=image_task,
            model_output=output,
        )
        return normalized_output, document, label

    if role in {"glm", "judge", "qwen"} or provider.startswith(("glm", "qwen")):
        normalized_output, document, label = normalize_qwen_payload(
            image_task=image_task,
            model_output=output,
        )
        return normalized_output, document, label

    return (
        output,
        empty_document(
            image_task=image_task,
            source=f"{provider or client.model_name}_unsupported_primary",
        ),
        None,
    )


def _run_first_pass_model(
    image_task: ImageTask,
    client: BaseLocalClient | None,
    prompt: str,
    retry: int,
) -> dict[str, Any]:
    role = _client_role(client) or "unconfigured"
    output = call_with_retry(
        client=client,
        image_task=image_task,
        prompt=prompt,
        retry=retry,
    )
    normalized_output, document, label = _normalize_primary_output(
        image_task=image_task,
        client=client,
        output=output,
    )
    return {
        "role": role,
        "client": client,
        "output": normalized_output,
        "document": document,
        "label": label,
    }


def _build_skipped_first_pass_bundle(
    image_task: ImageTask,
    client: BaseLocalClient | None,
    role: str,
    reason: str,
) -> dict[str, Any]:
    document = empty_document(
        image_task=image_task,
        source=f"{role}_{reason}",
    )
    document.warnings = [reason]
    document.raw_metadata["skipped_reason"] = reason
    output = ModelOutput(
        image_id=image_task.image_id,
        model_name=client.model_name if client is not None else role,
        success=False,
        raw_text="",
        error=reason,
        source_type="skipped",
    )
    return {
        "role": role,
        "client": client,
        "output": output,
        "document": document,
        "label": None,
    }


def _is_flowchart_candidate_block(block: CanonicalBlock) -> bool:
    if block.type not in {"chart", "image"}:
        return False
    if str(block.sub_type or "").strip().lower() == "flowchart":
        return True
    if block.structured_label.kind == "mermaid":
        return True
    return bool(block.flowchart_graph)


def _has_flowchart_signal(
    document: CanonicalDocument,
    label: Any | None,
) -> bool:
    if label is not None and str(getattr(label, "image_type", "") or "") == "flowchart":
        return True
    return any(_is_flowchart_candidate_block(block) for block in document.blocks)


def _has_flowchart_path_hint(image_task: ImageTask) -> bool:
    haystack = " ".join(
        [
            str(image_task.image_id or ""),
            str(image_task.file_name or ""),
            str(image_task.image_path or ""),
        ]
    ).lower()
    return any(token in haystack for token in ("flowchart", "流程图"))


def _should_use_flowchart_branch(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
) -> bool:
    if _has_flowchart_path_hint(image_task):
        return True
    return _has_flowchart_signal(mineru_document, mineru_label)


def _should_use_table_branch(
    mineru_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    auxiliary_bundles: list[dict[str, Any]],
) -> bool:
    if _has_flowchart_signal(mineru_document, mineru_label):
        return False
    if is_table_like(mineru_document) or is_table_like(mineru_label):
        return True
    for bundle in auxiliary_bundles:
        document = bundle.get("document")
        label = bundle.get("label")
        if is_table_like(document) or is_table_like(label):
            return True
    return False


def _is_non_flowchart_chart_block(block: CanonicalBlock) -> bool:
    if _is_flowchart_candidate_block(block):
        return False
    source_block_type = str(
        block.provenance.get("source_block_type", "") or ""
    ).strip().lower()
    source_sub_type = str(
        block.provenance.get("source_sub_type", "") or ""
    ).strip().lower()
    if source_sub_type == "flowchart":
        return False
    if block.type == "chart":
        return True
    return source_block_type == "chart"


def _should_force_chart_table_second_pass(
    mineru_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
) -> bool:
    del mineru_label
    return any(
        _is_non_flowchart_chart_block(block)
        for block in mineru_document.blocks
    )


def _pick_stage2_table_reference_bundle(
    table_analysis: dict[str, Any] | None,
) -> dict[str, Any] | None:
    bundle = pick_table_reference_bundle(table_analysis)
    if bundle is not None:
        return bundle
    if not isinstance(table_analysis, dict):
        return None
    candidate_bundles = table_analysis.get("candidate_bundles")
    if not isinstance(candidate_bundles, list):
        return None
    non_mineru = [
        bundle
        for bundle in candidate_bundles
        if isinstance(bundle, dict)
        and str(bundle.get("role", "") or "").strip().lower() != "mineru"
    ]
    if non_mineru:
        return non_mineru[0]
    for bundle in candidate_bundles:
        if isinstance(bundle, dict):
            return bundle
    return None


def _is_seal_candidate_block(block: CanonicalBlock) -> bool:
    if block.type != "image":
        return False
    if str(block.sub_type or "").strip().lower() == "seal":
        return True
    return any(
        str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions
    )


def _normalize_reference_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _ordered_unique_reference_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_reference_text(value)
        if not normalized:
            continue
        signature = normalized.lower()
        if signature in seen:
            continue
        seen.add(signature)
        ordered.append(normalized)
    return ordered


def _deduplicate_text_values(values: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _build_table_analysis_summary(
    table_analysis: dict[str, Any],
    artifact_reference_bundle: dict[str, Any] | None,
    table_issues: list[Any],
    table_patch_decisions: list[PatchDecision],
    table_patch_outputs: list[ModelOutput],
) -> dict[str, Any]:
    candidate_roles = _deduplicate_text_values(
        list(table_analysis.get("candidate_roles") or [])
    )
    artifact_reference_role = (
        str(artifact_reference_bundle.get("role", "") or "").strip().lower()
        if isinstance(artifact_reference_bundle, dict)
        else ""
    )
    pairwise_scores: list[dict[str, Any]] = []
    for item in list(table_analysis.get("pairwise") or []):
        if not isinstance(item, dict):
            continue
        left = str(item.get("left", "") or "").strip().lower()
        right = str(item.get("right", "") or "").strip().lower()
        try:
            score = round(float(item.get("score", 0.0)), 4)
        except (TypeError, ValueError):
            score = 0.0
        if not left or not right:
            continue
        pairwise_scores.append({"left": left, "right": right, "score": score})

    return {
        "branch_used": True,
        "branch_mode": str(table_analysis.get("branch_mode", "") or "").strip()
        or "table",
        "forced_second_pass": bool(table_analysis.get("forced_second_pass", False)),
        "candidate_count": len(candidate_roles),
        "candidate_roles": candidate_roles,
        "fallback": bool(table_analysis.get("fallback", False)),
        "fallback_reason": str(table_analysis.get("reason", "") or "").strip(),
        "stable_consensus": bool(table_analysis.get("stable_consensus", False)),
        "consensus_kind": str(table_analysis.get("consensus_kind", "") or "").strip(),
        "consensus_cluster": _deduplicate_text_values(
            list(table_analysis.get("consensus_cluster") or [])
        ),
        "reference_role": str(table_analysis.get("reference_role", "") or "").strip().lower()
        or None,
        "requires_qwen": bool(table_analysis.get("requires_qwen", False)),
        "review_reasons": _deduplicate_text_values(
            list(table_analysis.get("review_reasons") or [])
        ),
        "severe_conflicts": _deduplicate_text_values(
            list(table_analysis.get("severe_conflicts") or [])
        ),
        "pairwise_matrix": dict(table_analysis.get("matrix") or {}),
        "pairwise_scores": pairwise_scores,
        "artifact_reference_role": artifact_reference_role or None,
        "artifact_reference_included": bool(artifact_reference_role),
        "stage2_issue_count": len(table_issues),
        "stage2_patch_decisions": _deduplicate_text_values(
            [decision.decision for decision in table_patch_decisions]
        ),
        "stage2_output_count": len(table_patch_outputs),
        "stage2_success_count": sum(
            1 for output in table_patch_outputs if bool(getattr(output, "success", False))
        ),
    }


def _build_table_artifact_reason_override(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    forced_second_pass = bool(summary.get("forced_second_pass", False))
    if bool(summary.get("fallback", False)):
        reasons.append("table branch fell back before candidate consensus")
        fallback_reason = str(summary.get("fallback_reason", "") or "").strip()
        if fallback_reason:
            reasons.append(f"table fallback reason: {fallback_reason}")
    else:
        for review_reason in list(summary.get("review_reasons") or []):
            mapped = _TABLE_REVIEW_REASON_MESSAGES.get(str(review_reason or "").strip())
            if mapped:
                reasons.append(mapped)
        if (
            forced_second_pass
            and int(summary.get("stage2_issue_count", 0) or 0) > 0
            and int(summary.get("stage2_success_count", 0) or 0) == 0
        ) or (
            not forced_second_pass
            and bool(summary.get("requires_qwen", False))
            and int(summary.get("stage2_issue_count", 0) or 0) > 0
        ):
            reasons.append(
                "table second-stage adjudication did not produce an adoptable patch"
            )
    reasons.append("table auxiliary candidates were analyzed outside artifact pairwise comparison")
    return _deduplicate_text_values(reasons)


def _annotate_table_artifact(
    artifact: Any,
    table_analysis: dict[str, Any] | None,
    artifact_reference_bundle: dict[str, Any] | None,
    table_issues: list[Any],
    table_patch_decisions: list[PatchDecision],
    table_patch_outputs: list[ModelOutput],
) -> None:
    if not isinstance(table_analysis, dict):
        return

    summary = _build_table_analysis_summary(
        table_analysis=table_analysis,
        artifact_reference_bundle=artifact_reference_bundle,
        table_issues=table_issues,
        table_patch_decisions=table_patch_decisions,
        table_patch_outputs=table_patch_outputs,
    )
    artifact.final_document.raw_metadata["table_analysis"] = summary

    if (
        summary["candidate_count"] < 2
        or summary["artifact_reference_included"]
        or artifact.consensus is None
    ):
        return

    override_reasons = _build_table_artifact_reason_override(summary)
    artifact.consensus.reasons = _deduplicate_text_values(
        override_reasons
        + [
            reason
            for reason in artifact.consensus.reasons
            if reason not in _TABLE_ARTIFACT_PLACEHOLDER_REASONS
        ]
    )
    artifact.consensus.escalation_reasons = _deduplicate_text_values(
        [
            "table_auxiliary_candidates_not_reflected_in_artifact_pairwise_scores",
        ]
        + [
            reason
            for reason in artifact.consensus.escalation_reasons
            if reason not in _TABLE_ARTIFACT_PLACEHOLDER_ESCALATIONS
        ]
    )
    artifact.consensus.validation_warnings = _deduplicate_text_values(
        ["table_consensus_uses_auxiliary_candidate_matrix"]
        + list(artifact.consensus.validation_warnings)
    )
    artifact.reasons = _deduplicate_text_values(
        override_reasons
        + [
            reason
            for reason in artifact.reasons
            if reason not in _TABLE_ARTIFACT_PLACEHOLDER_REASONS
        ]
    )


def _annotate_flowvqa_artifact(
    image_task: ImageTask,
    flowvqa_root: Path | None,
    artifact: Any,
    mineru_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    qwen_document: CanonicalDocument,
    qwen_label: ParsedLabel | None,
    paddle_bundle: dict[str, Any] | None,
    glm_bundle: dict[str, Any] | None,
) -> None:
    reference = find_flowvqa_reference(
        flowvqa_root=flowvqa_root,
        image_id=image_task.image_id,
    )
    if reference is None:
        return

    predictions_by_source: dict[str, str] = {
        "mineru": extract_mermaid_from_document(mineru_document, mineru_label),
        "qwen": extract_mermaid_from_document(qwen_document, qwen_label),
        "final": extract_mermaid_from_document(
            artifact.final_document,
            artifact.final_label if isinstance(artifact.final_label, ParsedLabel) else None,
        ),
    }

    for source_name, bundle in (("paddle", paddle_bundle), ("glm", glm_bundle)):
        if not isinstance(bundle, dict):
            continue
        bundle_document = bundle.get("document")
        bundle_label = bundle.get("label")
        if not isinstance(bundle_document, CanonicalDocument):
            continue
        predictions_by_source[source_name] = extract_mermaid_from_document(
            bundle_document,
            bundle_label if isinstance(bundle_label, ParsedLabel) else None,
        )

    flowvqa_eval = build_flowvqa_eval_payload(
        reference=reference,
        predictions_by_source=predictions_by_source,
    )
    if flowvqa_eval is not None:
        artifact.final_document.raw_metadata["flowvqa_eval"] = flowvqa_eval


def _extract_flowchart_ocr_reference(
    bundle: dict[str, Any] | None,
) -> tuple[list[str], str | None]:
    if not isinstance(bundle, dict):
        return [], None
    document = bundle.get("document")
    output = bundle.get("output")
    if not isinstance(document, CanonicalDocument):
        return [], None

    texts: list[str] = []
    for block in sorted(
        document.blocks,
        key=lambda item: (item.page_idx, item.order_index, item.block_id),
    ):
        if str(block.type or "").strip().lower() == "chart" and str(
            block.sub_type or ""
        ).strip().lower() == "flowchart":
            continue
        if str(block.text or "").strip():
            texts.append(str(block.text or ""))
        texts.extend(str(item or "") for item in block.visible_text if str(item or "").strip())
        texts.extend(
            str(region.text or "")
            for region in block.ocr_regions
            if str(region.text or "").strip()
        )
    references = _ordered_unique_reference_texts(texts)
    model_name = (
        output.model_name
        if isinstance(output, ModelOutput) and str(output.model_name or "").strip()
        else str(bundle.get("role", "") or "").strip().lower() or None
    )
    return references[:20], model_name


def _collect_flowchart_ocr_references(
    bundles: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    reference_sources: list[dict[str, Any]] = []
    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        reference_texts, reference_model = _extract_flowchart_ocr_reference(bundle)
        if not reference_texts:
            continue
        role = str(bundle.get("role", "") or "").strip().lower() or "candidate"
        reference_sources.append(
            {
                "reference_model_role": role,
                "reference_model_name": reference_model or role,
                "ocr_reference_texts": reference_texts[:20],
            }
        )
    return reference_sources


def _attach_flowchart_ocr_reference(
    issues: list[Any],
    reference_sources: list[dict[str, Any]],
) -> list[Any]:
    if not issues or not reference_sources:
        return issues
    combined_texts = _ordered_unique_reference_texts(
        [
            str(text or "")
            for source in reference_sources
            if isinstance(source, dict)
            for text in list(source.get("ocr_reference_texts") or [])
        ]
    )
    if not combined_texts:
        return issues
    for issue in issues:
        payload = (
            dict(issue.candidate_payload)
            if isinstance(issue.candidate_payload, dict)
            else {}
        )
        payload["ocr_reference_texts"] = combined_texts[:40]
        payload["ocr_reference_sources"] = [
            {
                "reference_model_role": str(
                    source.get("reference_model_role", "") or ""
                ).strip()
                or "candidate",
                "reference_model_name": str(
                    source.get("reference_model_name", "") or ""
                ).strip()
                or str(source.get("reference_model_role", "") or "").strip()
                or "candidate",
                "ocr_reference_texts": _ordered_unique_reference_texts(
                    [
                        str(text or "")
                        for text in list(source.get("ocr_reference_texts") or [])
                    ]
                )[:20],
            }
            for source in reference_sources
            if isinstance(source, dict)
        ]
        issue.candidate_payload = payload
    return issues


def _build_seal_adjudication_candidates(
    image_task: ImageTask,
    mineru_bundle: dict[str, Any],
    auxiliary_bundles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    ordered_bundles = [{"role": "mineru", **mineru_bundle}] + list(auxiliary_bundles)
    candidate_bundles: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str, tuple[str, ...]]] = set()
    has_explicit_seal_signal = False

    for bundle in ordered_bundles:
        document = bundle.get("document")
        output = bundle.get("output")
        label = bundle.get("label")
        role = str(bundle.get("role", "") or "").strip().lower() or "candidate"
        if not isinstance(document, CanonicalDocument):
            continue
        if isinstance(output, ModelOutput) and not output.success:
            continue
        effective_label = label if isinstance(label, ParsedLabel) else derive_label_from_document(document)
        if effective_label is None:
            continue
        if effective_label.image_type in {"flowchart", "chart", "table"}:
            continue

        projected_document = project_document_for_single_block_view(
            document=document,
            label=effective_label,
        )
        projected_document = (
            projected_document
            if isinstance(projected_document, CanonicalDocument)
            else document
        )
        full_text = _projected_document_text(projected_document)
        seal_texts = _extract_seal_candidate_texts(
            document=projected_document,
            label=effective_label,
        )
        core_seal_text = _extract_core_seal_text(
            label=effective_label,
            seal_texts=seal_texts,
        )
        signature = _seal_candidate_signature(
            label=effective_label,
            full_text=full_text,
            seal_texts=seal_texts,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        if _has_explicit_seal_signal(document=document, label=effective_label):
            has_explicit_seal_signal = True

        candidate_payload = {
            "candidate_id": role,
            "model_name": (
                output.model_name
                if isinstance(output, ModelOutput)
                and str(output.model_name or "").strip()
                else document.source
            ),
            "image_type": effective_label.image_type,
            "caption": effective_label.caption,
            "core_seal_text": core_seal_text,
            "selection_focus": "只看印章主体文字，忽略水印、重叠字、背景说明和非印章正文",
            "full_text": full_text,
            "visible_text": effective_label.visible_text[:10],
            "seal_texts": seal_texts,
            "source_block_count": len(document.blocks),
        }
        candidate_bundles.append(
            {
                "role": role,
                "output": output,
                "document": document,
                "label": effective_label,
                "candidate_payload": candidate_payload,
                "signature": signature,
            }
        )

    if len(candidate_bundles) < 2 or not has_explicit_seal_signal:
        return candidate_bundles, None

    mineru_candidate = next(
        (candidate for candidate in candidate_bundles if candidate.get("role") == "mineru"),
        None,
    )
    if mineru_candidate is None:
        return candidate_bundles, None

    disagreement_detected = False
    comparisons: list[dict[str, Any]] = []
    mineru_document = mineru_candidate["document"]
    for candidate_bundle in candidate_bundles:
        if candidate_bundle.get("role") == "mineru":
            continue
        issues = detect_seal_issues(
            image_task=image_task,
            mineru_document=mineru_document,
            qwen_document=candidate_bundle["document"],
        )
        if issues or candidate_bundle["signature"] != mineru_candidate["signature"]:
            disagreement_detected = True
        comparisons.append(
            {
                "candidate_id": candidate_bundle["role"],
                "issue_types": [issue.issue_type for issue in issues],
                "reason_tags": _ordered_unique_texts(
                    [
                        reason
                        for issue in issues
                        for reason in list(issue.reasons or [])
                    ]
                ),
            }
        )

    if not disagreement_detected:
        return candidate_bundles, None

    return candidate_bundles, {
        "image_id": image_task.image_id,
        "task": "seal_candidate_selection",
        "selection_focus": "只比较印章主体内容，忽略水印、重叠字、背景说明和非印章正文",
        "candidate_count": len(candidate_bundles),
        "candidates": [
            candidate["candidate_payload"] for candidate in candidate_bundles
        ],
        "comparisons": comparisons,
    }


def _resolve_selected_seal_bundle(
    candidate_bundles: list[dict[str, Any]],
    selection_decision: SealSelectionDecision | None,
) -> dict[str, Any] | None:
    if selection_decision is None:
        return None
    selected_role = str(selection_decision.selected_candidate or "").strip().lower()
    if not selected_role or selected_role == "review":
        return None
    for candidate_bundle in candidate_bundles:
        if str(candidate_bundle.get("role", "") or "").strip().lower() == selected_role:
            return candidate_bundle
    return None


def _has_explicit_seal_signal(
    document: CanonicalDocument,
    label: ParsedLabel,
) -> bool:
    if label.image_type == "seal":
        return True
    if any(str(region.role or "").strip().lower() == "seal" for region in label.ocr_regions):
        return True
    return any(_is_seal_candidate_block(block) for block in document.blocks)


def _projected_document_text(document: CanonicalDocument) -> str:
    texts = [
        str(block.text or "").strip()
        for block in sorted(
            document.blocks,
            key=lambda item: (item.page_idx, item.order_index, item.block_id),
        )
        if str(block.text or "").strip()
    ]
    return "\n\n".join(texts)


def _extract_seal_candidate_texts(
    document: CanonicalDocument,
    label: ParsedLabel,
) -> list[str]:
    texts: list[str] = []
    texts.extend(
        str(region.text or "").strip()
        for region in label.ocr_regions
        if str(region.role or "").strip().lower() == "seal"
        and str(region.text or "").strip()
    )
    for block in document.blocks:
        texts.extend(
            str(region.text or "").strip()
            for region in block.ocr_regions
            if str(region.role or "").strip().lower() == "seal"
            and str(region.text or "").strip()
        )
    if not texts and label.caption.strip():
        texts.append(label.caption.strip())
    return _ordered_unique_texts(texts)


def _seal_candidate_signature(
    label: ParsedLabel,
    full_text: str,
    seal_texts: list[str],
) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(label.image_type or "").strip().lower(),
        _normalize_selection_text(full_text),
        tuple(
            _normalize_selection_text(text)
            for text in seal_texts
            if _normalize_selection_text(text)
        ),
    )


def _extract_core_seal_text(
    label: ParsedLabel,
    seal_texts: list[str],
) -> str:
    primary = str(primary_seal_text(label) or "").strip()
    if primary:
        return primary
    if seal_texts:
        return str(seal_texts[0] or "").strip()
    return ""


def _normalize_selection_text(value: Any) -> str:
    return "".join(str(value or "").split()).lower()


def _ordered_unique_texts(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_selection_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(str(value).strip())
    return ordered


def _seal_signal_score(
    document: CanonicalDocument,
    label: ParsedLabel | None,
) -> tuple[int, int, int, int]:
    seal_blocks = [block for block in document.blocks if _is_seal_candidate_block(block)]
    unique_texts: set[str] = set()
    seal_region_count = 0
    for block in seal_blocks:
        text = str(block.text or "").strip()
        if text:
            unique_texts.add(text)
        content_value = str(block.content.get("content", "") or "").strip()
        if content_value:
            unique_texts.add(content_value)
        captions = block.content.get("image_caption")
        if isinstance(captions, list):
            for item in captions:
                text = str(item or "").strip()
                if text:
                    unique_texts.add(text)
        for region in block.ocr_regions:
            if str(region.role or "").strip().lower() != "seal":
                continue
            seal_region_count += 1
            text = str(region.text or "").strip()
            if text:
                unique_texts.add(text)
    label_bonus = 1 if label is not None and label.image_type == "seal" else 0
    return (
        len(seal_blocks),
        seal_region_count,
        len(unique_texts),
        label_bonus,
    )


def _extract_flowchart_mermaid(
    document: CanonicalDocument,
    label: Any | None,
) -> str:
    if label is not None:
        candidate = normalize_mermaid_text(
            str(getattr(getattr(label, "structured_label", None), "content", "") or "")
        )
        if looks_like_mermaid(candidate):
            return candidate
    for block in document.blocks:
        if not _is_flowchart_candidate_block(block):
            continue
        candidates = [
            str(block.content.get("content", "") or ""),
            str(block.text or ""),
        ]
        for key in ("chart_caption", "image_caption"):
            values = block.content.get(key)
            if isinstance(values, list):
                candidates.extend(str(item or "") for item in values)
        for candidate in candidates:
            normalized = normalize_mermaid_text(candidate)
            if looks_like_mermaid(normalized):
                return normalized
    return ""


def _is_complete_flowchart_result(
    output: ModelOutput | None,
    document: CanonicalDocument,
    label: Any | None,
) -> bool:
    if output is None or not output.success:
        return False
    finish_reason = _extract_finish_reason(output.parsed)
    if finish_reason == "length":
        return False
    mermaid = _extract_flowchart_mermaid(document=document, label=label)
    if not mermaid:
        return False
    if label is not None and str(getattr(label, "image_type", "") or "") not in {
        "",
        "flowchart",
    }:
        return False
    return True


def _flowchart_graph_size(document: CanonicalDocument, label: Any | None) -> int:
    mermaid = _extract_flowchart_mermaid(document=document, label=label)
    if not mermaid:
        return 0
    return max(0, len([line for line in mermaid.splitlines() if line.strip()]) - 1)


def _pick_flowchart_reference_bundle(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    mineru_label: ParsedLabel | None,
    candidate_bundles: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[Any]]:
    candidates: list[tuple[int, dict[str, Any], list[Any]]] = []
    for bundle in candidate_bundles:
        output = bundle.get("output")
        document = bundle.get("document")
        label = bundle.get("label")
        reference_role = str(bundle.get("role", "") or "").strip().lower()
        if reference_role not in {"qwen", "judge"}:
            continue
        if not isinstance(document, CanonicalDocument):
            continue
        if not _is_complete_flowchart_result(output=output, document=document, label=label):
            continue
        issues = detect_flowchart_issues(
            image_task=image_task,
            mineru_document=mineru_document,
            qwen_document=document,
            mineru_label=mineru_label,
            qwen_label=label,
        )
        if not issues:
            continue
        normalized_reference_role = "qwen" if reference_role == "judge" else reference_role
        for issue in issues:
            if isinstance(issue.candidate_payload, dict):
                issue.candidate_payload["reference_model_role"] = normalized_reference_role
                issue.candidate_payload["reference_model_name"] = (
                    output.model_name
                    if isinstance(output, ModelOutput)
                    else normalized_reference_role
                )
        candidates.append(
            (
                _flowchart_graph_size(document=document, label=label),
                bundle,
                issues,
            )
        )
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: item[0], reverse=True)
    _score, bundle, issues = candidates[0]
    return bundle, issues


def _pick_seal_reference_bundle(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    auxiliary_bundles: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[Any]]:
    candidates: list[
        tuple[int, tuple[int, int, int, int], int, dict[str, Any], list[Any]]
    ] = []
    bundle_count = len(auxiliary_bundles)
    for bundle_index, bundle in enumerate(auxiliary_bundles):
        output = bundle.get("output")
        document = bundle.get("document")
        label = bundle.get("label")
        if not isinstance(document, CanonicalDocument):
            continue
        if not isinstance(output, ModelOutput) or not output.success:
            continue
        if not any(_is_seal_candidate_block(block) for block in document.blocks):
            continue
        issues = detect_seal_issues(
            image_task=image_task,
            mineru_document=mineru_document,
            qwen_document=document,
        )
        reference_role = str(bundle.get("role", "") or "").strip().lower()
        reference_name = (
            output.model_name
            if isinstance(output, ModelOutput) and str(output.model_name or "").strip()
            else reference_role
        )
        for issue in issues:
            candidate_payload = (
                dict(issue.candidate_payload)
                if isinstance(issue.candidate_payload, dict)
                else {}
            )
            candidate_payload["reference_model_role"] = reference_role
            candidate_payload["reference_model_name"] = reference_name
            issue.candidate_payload = candidate_payload
        candidates.append(
            (
                1 if issues else 0,
                _seal_signal_score(document=document, label=label),
                bundle_count - bundle_index,
                bundle,
                issues,
            )
        )
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _has_issues, _score, _order, bundle, issues = candidates[0]
    return bundle, issues


def _build_flowchart_first_pass_decisions(
    issues: list[Any],
    qwen_complete: bool,
) -> list[PatchDecision]:
    if not issues:
        return []
    decision = "use_qwen_fields" if qwen_complete else "keep_mineru"
    reason = (
        "qwen_flowchart_preferred_on_conflict"
        if qwen_complete
        else "qwen_flowchart_incomplete"
    )
    return [
        PatchDecision(
            issue_id=str(issue.issue_id),
            target_block_id=str(issue.target_block_id or "") or None,
            decision=decision,
            patch={},
            reason=reason,
        )
        for issue in issues
    ]


def process_image_task(
    image_task: ImageTask,
    args: argparse.Namespace,
    mineru_client: BaseLocalClient | None,
    paddle_client: BaseLocalClient | None,
    glm_client: BaseLocalClient | None,
    qwen_client: BaseLocalClient | None,
    recognition_prompt: str,
    seal_adjudication_prompt: str,
    flowchart_adjudication_prompt: str,
    output_dir: Path,
    table_adjudication_prompt: str = "",
) -> dict[str, Any]:
    mineru_bundle = _run_first_pass_model(
        image_task=image_task,
        client=mineru_client,
        prompt=recognition_prompt,
        retry=args.retry,
    )

    mineru_output = mineru_bundle["output"]
    mineru_document = mineru_bundle["document"]
    mineru_label = mineru_bundle["label"]
    use_flowchart_branch = _should_use_flowchart_branch(
        image_task=image_task,
        mineru_document=mineru_document,
        mineru_label=mineru_label,
    )

    paddle_bundle = _run_first_pass_model(
        image_task=image_task,
        client=paddle_client,
        prompt=recognition_prompt,
        retry=args.retry,
    )
    glm_bundle = _run_first_pass_model(
        image_task=image_task,
        client=glm_client,
        prompt=recognition_prompt,
        retry=args.retry,
    )

    qwen_output: ModelOutput | None = None
    qwen_document = empty_document(image_task=image_task, source="qwen_judge_not_triggered")
    qwen_label = None
    qwen_first_pass_bundle: dict[str, Any] | None = None
    force_chart_table_second_pass = (
        _should_force_chart_table_second_pass(
            mineru_document=mineru_document,
            mineru_label=mineru_label,
        )
        if not use_flowchart_branch
        else False
    )

    if use_flowchart_branch and qwen_client is not None:
        qwen_first_pass_bundle = _run_first_pass_model(
            image_task=image_task,
            client=qwen_client,
            prompt=recognition_prompt,
            retry=args.retry,
        )
        qwen_first_pass_bundle["role"] = "qwen"
        qwen_output = qwen_first_pass_bundle["output"]
        qwen_document = qwen_first_pass_bundle["document"]
        qwen_label = qwen_first_pass_bundle["label"]
    elif force_chart_table_second_pass and qwen_client is not None:
        qwen_first_pass_bundle = _run_first_pass_model(
            image_task=image_task,
            client=qwen_client,
            prompt=recognition_prompt,
            retry=args.retry,
        )
        qwen_first_pass_bundle["role"] = "qwen"
        qwen_output = qwen_first_pass_bundle["output"]

    seal_selection_decision: SealSelectionDecision | None = None
    seal_selection_output: ModelOutput | None = None
    seal_stage2_record: dict[str, Any] | None = None
    selected_seal_bundle: dict[str, Any] | None = None

    auxiliary_bundles = []
    if not use_flowchart_branch:
        if (
            force_chart_table_second_pass
            and isinstance(qwen_first_pass_bundle, dict)
            and qwen_first_pass_bundle.get("client") is not None
        ):
            auxiliary_bundles.append(qwen_first_pass_bundle)
        auxiliary_bundles.extend(
            [
                bundle
                for bundle in (glm_bundle, paddle_bundle)
                if bundle.get("client") is not None
            ]
        )
    use_table_branch = (
        (
            force_chart_table_second_pass
            or _should_use_table_branch(
                mineru_document=mineru_document,
                mineru_label=mineru_label,
                auxiliary_bundles=auxiliary_bundles,
            )
        )
        if not use_flowchart_branch
        else False
    )
    seal_candidate_bundles: list[dict[str, Any]] = []
    seal_selection_payload: dict[str, Any] | None = None
    if not use_flowchart_branch and not use_table_branch:
        seal_candidate_bundles, seal_selection_payload = _build_seal_adjudication_candidates(
            image_task=image_task,
            mineru_bundle=mineru_bundle,
            auxiliary_bundles=auxiliary_bundles,
        )
    reference_bundle: dict[str, Any] | None = None
    flowchart_issues: list[Any] = []
    flowchart_patch_decisions: list[PatchDecision] = []
    flowchart_patch_outputs: list[ModelOutput] = []
    table_issues: list[Any] = []
    table_patch_decisions: list[PatchDecision] = []
    table_patch_outputs: list[ModelOutput] = []
    table_analysis: dict[str, Any] | None = None
    flowchart_ocr_reference_sources = (
        _collect_flowchart_ocr_references([paddle_bundle, glm_bundle])
        if use_flowchart_branch
        else []
    )

    if seal_selection_payload is not None:
        seal_selection_decision, seal_selection_output = adjudicate_seal_candidates_with_llm(
            client=qwen_client,
            image_task=image_task,
            prompt=seal_adjudication_prompt,
            selection_payload=seal_selection_payload,
            retry=args.retry,
        )
        selected_seal_bundle = _resolve_selected_seal_bundle(
            candidate_bundles=seal_candidate_bundles,
            selection_decision=seal_selection_decision,
        )
        seal_stage2_record = build_stage2_selection_record(
            selection_payload=seal_selection_payload,
            output=seal_selection_output,
            selection_decision=seal_selection_decision,
            prompt=seal_adjudication_prompt,
            mode="seal_adjudication",
        )
        if seal_selection_output is not None:
            qwen_output = seal_selection_output

    if use_table_branch:
        table_analysis = analyze_table_bundles(
            mineru_bundle=mineru_bundle,
            auxiliary_bundles=auxiliary_bundles,
            allow_non_table_chart_fallback=force_chart_table_second_pass,
        )
        if isinstance(table_analysis, dict):
            table_analysis["branch_mode"] = (
                "chart_table" if force_chart_table_second_pass else "table"
            )
            table_analysis["forced_second_pass"] = force_chart_table_second_pass
            if force_chart_table_second_pass:
                table_analysis["requires_qwen"] = True
                table_analysis["review_reasons"] = _deduplicate_text_values(
                    list(table_analysis.get("review_reasons") or [])
                    + ["chart_table_requires_qwen_second_pass"]
                )
            reference_bundle = _pick_stage2_table_reference_bundle(
                table_analysis
            )
            if (
                force_chart_table_second_pass
                and isinstance(reference_bundle, dict)
                and not str(table_analysis.get("reference_role", "") or "").strip()
            ):
                table_analysis["reference_role"] = str(
                    reference_bundle.get("role", "") or ""
                ).strip().lower()
            if force_chart_table_second_pass:
                table_issues = build_chart_table_second_pass_issues(
                    image_task=image_task,
                    mineru_document=mineru_document,
                    candidate_bundles=table_analysis.get("candidate_bundles") or [],
                    consensus_analysis=table_analysis,
                )
            else:
                table_issue = build_table_issue(
                    image_task=image_task,
                    mineru_candidate=table_analysis.get("mineru_candidate"),
                    candidate_bundles=table_analysis.get("candidate_bundles") or [],
                    consensus_analysis=table_analysis,
                )
                table_issues = []
            if (
                (
                    force_chart_table_second_pass
                    or (
                        not bool(table_analysis.get("fallback", False))
                        and bool(table_analysis.get("requires_qwen"))
                    )
                )
                and (
                    table_issues
                    if force_chart_table_second_pass
                    else table_issue is not None
                )
            ):
                if not force_chart_table_second_pass and table_issue is not None:
                    table_issues = [table_issue]
                if qwen_client is not None:
                    table_patch_decisions, table_patch_outputs = adjudicate_issues_with_llm(
                        client=qwen_client,
                        image_task=image_task,
                        prompt=table_adjudication_prompt,
                        issues=table_issues,
                        mode="table_adjudication",
                        retry=args.retry,
                    )
                    if table_patch_outputs:
                        qwen_output = table_patch_outputs[-1]

    if use_flowchart_branch:
        reference_bundle, flowchart_issues = _pick_flowchart_reference_bundle(
            image_task=image_task,
            mineru_document=mineru_document,
            mineru_label=mineru_label,
            candidate_bundles=(
                [qwen_first_pass_bundle]
                if qwen_first_pass_bundle is not None
                else []
            ),
        )
        qwen_complete = _is_complete_flowchart_result(
            output=qwen_output,
            document=qwen_document,
            label=qwen_label,
        )
        if (
            not flowchart_issues
            and not qwen_complete
            and _has_flowchart_signal(mineru_document, mineru_label)
        ):
            flowchart_issues = detect_flowchart_second_pass_issues(
                image_task=image_task,
                mineru_document=mineru_document,
                mineru_label=mineru_label,
            )
        flowchart_issues = _attach_flowchart_ocr_reference(
            issues=flowchart_issues,
            reference_sources=flowchart_ocr_reference_sources,
        )
        if flowchart_issues and qwen_client is not None:
            flowchart_patch_decisions, flowchart_patch_outputs = adjudicate_issues_with_llm(
                client=qwen_client,
                image_task=image_task,
                prompt=flowchart_adjudication_prompt,
                issues=flowchart_issues,
                mode="flowchart_adjudication",
                retry=args.retry,
            )

    stage2_records = []
    if seal_stage2_record is not None:
        stage2_records.append(seal_stage2_record)
    stage2_records.extend(
        build_stage2_records(
            issues=table_issues,
            outputs=table_patch_outputs,
            patch_decisions=table_patch_decisions,
            prompt=table_adjudication_prompt,
            mode="table_adjudication",
        )
    )
    stage2_records.extend(
        build_stage2_records(
            issues=flowchart_issues,
            outputs=flowchart_patch_outputs,
            patch_decisions=flowchart_patch_decisions,
            prompt=flowchart_adjudication_prompt,
            mode="flowchart_adjudication",
        )
    )
    if not stage2_records:
        stage2_records = None

    all_issues = table_issues + flowchart_issues
    patch_decisions = table_patch_decisions + flowchart_patch_decisions
    final_mineru_document = mineru_document
    final_mineru_label = mineru_label
    if patch_decisions:
        final_mineru_document = apply_patch_decisions(
            mineru_document=final_mineru_document,
            issues=all_issues,
            patch_decisions=patch_decisions,
        )
        final_mineru_label = derive_label_from_document(final_mineru_document)

    artifact_reference_bundle = reference_bundle
    if force_chart_table_second_pass:
        artifact_reference_bundle = None
    elif table_issues and not any(
        decision.decision in {"use_qwen_fields", "merge", "keep_candidate", "add_qwen_block"}
        for decision in table_patch_decisions
    ):
        artifact_reference_bundle = None

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=final_mineru_document,
        qwen_document=(
            artifact_reference_bundle["document"]
            if artifact_reference_bundle is not None
            and isinstance(artifact_reference_bundle.get("document"), CanonicalDocument)
            else qwen_document
        ),
        mineru_label=final_mineru_label,
        qwen_label=(
            artifact_reference_bundle["label"]
            if artifact_reference_bundle is not None
            and isinstance(artifact_reference_bundle.get("label"), ParsedLabel)
            else qwen_label
        ),
        mineru_output=mineru_output,
        qwen_output=(
            artifact_reference_bundle["output"]
            if artifact_reference_bundle is not None
            and isinstance(artifact_reference_bundle.get("output"), ModelOutput)
            else qwen_output
        ),
        issues=all_issues,
        patch_decisions=patch_decisions,
        seal_selection=seal_selection_decision,
        seal_selected_role=(
            str(selected_seal_bundle.get("role", "") or "").strip().lower()
            if selected_seal_bundle is not None
            else None
        ),
        seal_selected_document=(
            selected_seal_bundle.get("document")
            if selected_seal_bundle is not None
            and isinstance(selected_seal_bundle.get("document"), CanonicalDocument)
            else None
        ),
        seal_selected_label=(
            selected_seal_bundle.get("label")
            if selected_seal_bundle is not None
            and isinstance(selected_seal_bundle.get("label"), ParsedLabel)
            else None
        ),
        seal_selected_output=(
            selected_seal_bundle.get("output")
            if selected_seal_bundle is not None
            and isinstance(selected_seal_bundle.get("output"), ModelOutput)
            else None
        ),
    )
    _annotate_flowvqa_artifact(
        image_task=image_task,
        flowvqa_root=getattr(args, "flowvqa_root", None),
        artifact=artifact,
        mineru_document=mineru_document,
        mineru_label=mineru_label,
        qwen_document=qwen_document,
        qwen_label=qwen_label,
        paddle_bundle=paddle_bundle,
        glm_bundle=glm_bundle,
    )
    _annotate_table_artifact(
        artifact=artifact,
        table_analysis=table_analysis,
        artifact_reference_bundle=artifact_reference_bundle,
        table_issues=table_issues,
        table_patch_decisions=table_patch_decisions,
        table_patch_outputs=table_patch_outputs,
    )
    summary_record = write_image_result(
        output_dir=output_dir,
        image_task=image_task,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        artifact=artifact,
        stage2_records=stage2_records,
        extra_stage1_results={
            "paddle": {
                "output": paddle_bundle["output"],
                "document": paddle_bundle["document"],
                "label": paddle_bundle["label"],
            },
            "glm": {
                "output": glm_bundle["output"],
                "document": glm_bundle["document"],
                "label": glm_bundle["label"],
            },
        },
    )
    if args.manual_compare_mode:
        try:
            generate_compare_page(
                image_id=image_task.image_id,
                output_dir=output_dir,
                compare_dir=output_dir / "compare_mermaid",
            )
        except Exception as exc:
            print(
                f"[manual-compare] failed for {image_task.image_id}: {type(exc).__name__}: {exc}"
            )
    return summary_record


def _refresh_compare_dashboard(output_dir: Path) -> None:
    try:
        generate_compare_dashboard(
            output_dir=output_dir,
            dashboard_dir=output_dir / "compare_dashboard",
        )
    except Exception as exc:
        print(f"[manual-compare-dashboard] failed: {type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    if args.flowvqa_root is not None:
        args.flowvqa_root = validate_flowvqa_root(args.flowvqa_root)
    if args.overwrite:
        clear_previous_outputs(args.output_dir)

    summary_path = initialize_summary_file(args.output_dir)
    model_configs = load_model_configs(args.models_config)
    clients = build_clients(
        model_configs=model_configs, request_timeout=args.request_timeout
    )
    mineru_client = pick_role_client(clients, "mineru") or pick_client(clients, "minerupro")
    paddle_client = pick_role_client(clients, "paddle")
    glm_client = pick_role_client(clients, "glm")
    qwen_client = pick_role_client(clients, "judge") or pick_client(clients, "qwen")
    recognition_prompt = load_prompt(args.prompts_config, "qwen_recognition_prompt")
    seal_adjudication_prompt = load_prompt(
        args.prompts_config, "qwen_adjudication_prompt"
    )
    flowchart_adjudication_prompt = load_prompt(
        args.prompts_config, "qwen_flowchart_adjudication_prompt"
    )
    table_adjudication_prompt = load_prompt(
        args.prompts_config, "qwen_table_adjudication_prompt"
    )

    image_tasks = load_image_tasks(args.data_dir)
    if args.limit is not None:
        image_tasks = image_tasks[: args.limit]
    if not image_tasks:
        print("No images found in data directory")
        return

    worker_count = max(1, int(args.workers or 1))
    if worker_count == 1:
        for image_task in tqdm(image_tasks, desc="Processing images"):
            summary_record = process_image_task(
                image_task=image_task,
                args=args,
                mineru_client=mineru_client,
                paddle_client=paddle_client,
                glm_client=glm_client,
                qwen_client=qwen_client,
                recognition_prompt=recognition_prompt,
                seal_adjudication_prompt=seal_adjudication_prompt,
                flowchart_adjudication_prompt=flowchart_adjudication_prompt,
                table_adjudication_prompt=table_adjudication_prompt,
                output_dir=args.output_dir,
            )
            append_summary_record(summary_path, summary_record)
            if args.manual_compare_mode:
                _refresh_compare_dashboard(args.output_dir)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    process_image_task,
                    image_task,
                    args,
                    mineru_client,
                    paddle_client,
                    glm_client,
                    qwen_client,
                    recognition_prompt,
                    seal_adjudication_prompt,
                    flowchart_adjudication_prompt,
                    args.output_dir,
                    table_adjudication_prompt,
                )
                for image_task in image_tasks
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing images",
            ):
                append_summary_record(summary_path, future.result())
                if args.manual_compare_mode:
                    _refresh_compare_dashboard(args.output_dir)

    if args.manual_compare_mode:
        _refresh_compare_dashboard(args.output_dir)

    print(f"Processed {len(image_tasks)} images")


if __name__ == "__main__":
    main()
