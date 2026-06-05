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
from src.image_loader import load_image_tasks
from src.pipeline.adjudicator import adjudicate_documents
from src.pipeline.issues import (
    detect_flowchart_issues,
    detect_seal_issues,
)
from src.pipeline.llm_adjudicator import (
    adjudicate_issues_with_llm,
    build_issue_prompt_payload,
)
from src.pipeline.normalizers import (
    derive_label_from_document,
    normalize_mineru_payload,
    normalize_paddle_payload,
    normalize_qwen_payload,
)
from src.pipeline.patches import apply_patch_decisions
from src.prompt_builder import load_prompt
from src.render_compare_dashboard import generate_compare_dashboard
from src.render_mermaid_compare import generate_compare_page
from src.pipeline.flowchart_utils import looks_like_mermaid, normalize_mermaid_text
from src.schema import (
    CanonicalBlock,
    CanonicalDocument,
    ImageTask,
    ModelOutput,
    PatchDecision,
    ParsedLabel,
)
from src.writer import (
    append_summary_record,
    clear_previous_outputs,
    initialize_summary_file,
    write_image_result,
)


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
                "patch_decision": decision.model_dump()
                if decision is not None
                else None,
            }
        )
    return records


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


def _is_seal_candidate_block(block: CanonicalBlock) -> bool:
    if block.type != "image":
        return False
    if str(block.sub_type or "").strip().lower() == "seal":
        return True
    return any(
        str(region.role or "").strip().lower() == "seal" for region in block.ocr_regions
    )


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
    auxiliary_bundles: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[Any]]:
    candidates: list[tuple[int, int, dict[str, Any], list[Any]]] = []
    role_priority = {"glm": 2, "paddle": 1}
    for bundle in auxiliary_bundles:
        output = bundle.get("output")
        document = bundle.get("document")
        label = bundle.get("label")
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
        reference_role = str(bundle.get("role", "") or "").strip().lower()
        for issue in issues:
            if isinstance(issue.candidate_payload, dict):
                issue.candidate_payload["reference_model_role"] = reference_role
                issue.candidate_payload["reference_model_name"] = (
                    output.model_name if isinstance(output, ModelOutput) else reference_role
                )
        candidates.append(
            (
                _flowchart_graph_size(document=document, label=label),
                role_priority.get(reference_role, 0),
                bundle,
                issues,
            )
        )
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _score, _priority, bundle, issues = candidates[0]
    return bundle, issues


def _pick_seal_reference_bundle(
    image_task: ImageTask,
    mineru_document: CanonicalDocument,
    auxiliary_bundles: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[Any]]:
    candidates: list[tuple[int, tuple[int, int, int, int], int, dict[str, Any], list[Any]]] = []
    role_priority = {"glm": 2, "paddle": 1}
    for bundle in auxiliary_bundles:
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
                role_priority.get(reference_role, 0),
                bundle,
                issues,
            )
        )
    if not candidates:
        return None, []
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _has_issues, _score, _priority, bundle, issues = candidates[0]
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
) -> dict[str, Any]:
    mineru_bundle = _run_first_pass_model(
        image_task=image_task,
        client=mineru_client,
        prompt=recognition_prompt,
        retry=args.retry,
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

    mineru_output = mineru_bundle["output"]
    mineru_document = mineru_bundle["document"]
    mineru_label = mineru_bundle["label"]

    qwen_output: ModelOutput | None = None
    qwen_document = empty_document(image_task=image_task, source="qwen_judge_not_triggered")
    qwen_label = None

    seal_issues: list[Any] = []
    seal_patch_decisions: list[Any] = []
    seal_patch_outputs: list[ModelOutput] = []

    auxiliary_bundles = [
        bundle
        for bundle in (glm_bundle, paddle_bundle)
        if bundle.get("client") is not None
    ]
    seal_reference_bundle, seal_issues = _pick_seal_reference_bundle(
        image_task=image_task,
        mineru_document=mineru_document,
        auxiliary_bundles=auxiliary_bundles,
    )
    reference_bundle: dict[str, Any] | None = None
    flowchart_issues: list[Any] = []
    flowchart_patch_decisions: list[PatchDecision] = []
    flowchart_patch_outputs: list[ModelOutput] = []

    if seal_issues and qwen_client is not None:
        seal_patch_decisions, seal_patch_outputs = adjudicate_issues_with_llm(
            client=qwen_client,
            image_task=image_task,
            prompt=seal_adjudication_prompt,
            issues=seal_issues,
            mode="seal_adjudication",
            retry=args.retry,
        )
        if seal_patch_outputs:
            qwen_output = seal_patch_outputs[-1]

    if _has_flowchart_signal(mineru_document, mineru_label):
        reference_bundle, flowchart_issues = _pick_flowchart_reference_bundle(
            image_task=image_task,
            mineru_document=mineru_document,
            mineru_label=mineru_label,
            auxiliary_bundles=auxiliary_bundles,
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
            if flowchart_patch_outputs:
                qwen_output = flowchart_patch_outputs[-1]

    stage2_records = build_stage2_records(
        issues=seal_issues,
        outputs=seal_patch_outputs,
        patch_decisions=seal_patch_decisions,
        prompt=seal_adjudication_prompt,
        mode="seal_adjudication",
    ) + build_stage2_records(
        issues=flowchart_issues,
        outputs=flowchart_patch_outputs,
        patch_decisions=flowchart_patch_decisions,
        prompt=flowchart_adjudication_prompt,
        mode="flowchart_adjudication",
    )
    if not stage2_records:
        stage2_records = None

    all_issues = seal_issues + flowchart_issues
    patch_decisions = seal_patch_decisions + flowchart_patch_decisions
    final_mineru_document = mineru_document
    final_mineru_label = mineru_label
    if seal_patch_decisions:
        final_mineru_document = apply_patch_decisions(
            mineru_document=final_mineru_document,
            issues=seal_issues,
            patch_decisions=seal_patch_decisions,
        )
        final_mineru_label = derive_label_from_document(final_mineru_document)
    if flowchart_patch_decisions:
        final_mineru_document = apply_patch_decisions(
            mineru_document=final_mineru_document,
            issues=flowchart_issues,
            patch_decisions=flowchart_patch_decisions,
        )
        final_mineru_label = derive_label_from_document(final_mineru_document)

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=final_mineru_document,
        qwen_document=(
            seal_reference_bundle["document"]
            if seal_reference_bundle is not None
            and isinstance(seal_reference_bundle.get("document"), CanonicalDocument)
            else qwen_document
        ),
        mineru_label=final_mineru_label,
        qwen_label=(
            seal_reference_bundle["label"]
            if seal_reference_bundle is not None
            and isinstance(seal_reference_bundle.get("label"), ParsedLabel)
            else qwen_label
        ),
        mineru_output=mineru_output,
        qwen_output=(
            seal_reference_bundle["output"]
            if seal_reference_bundle is not None
            and isinstance(seal_reference_bundle.get("output"), ModelOutput)
            else qwen_output
        ),
        issues=all_issues,
        patch_decisions=patch_decisions,
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


def main() -> None:
    args = parse_args()
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
                output_dir=args.output_dir,
            )
            append_summary_record(summary_path, summary_record)
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
        try:
            generate_compare_dashboard(
                output_dir=args.output_dir,
                dashboard_dir=args.output_dir / "compare_dashboard",
            )
        except Exception as exc:
            print(f"[manual-compare-dashboard] failed: {type(exc).__name__}: {exc}")

    print(f"Processed {len(image_tasks)} images")


if __name__ == "__main__":
    main()
