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
    detect_flowchart_second_pass_issues,
)
from src.pipeline.llm_adjudicator import (
    adjudicate_issues_with_llm,
    build_issue_prompt_payload,
)
from src.pipeline.normalizers import (
    derive_label_from_document,
    normalize_mineru_payload,
)
from src.pipeline.patches import apply_patch_decisions
from src.prompt_builder import load_prompt
from src.render_compare_dashboard import generate_compare_dashboard
from src.render_mermaid_compare import generate_compare_page
from src.schema import CanonicalDocument, ImageTask, ModelOutput
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


def pick_primary_clients(clients: list[BaseLocalClient]) -> list[BaseLocalClient]:
    primary_clients: list[BaseLocalClient] = []
    for client in clients:
        provider = str(client.config.get("provider", "")).strip().lower()
        if provider.startswith("qwen"):
            continue
        primary_clients.append(client)
    return primary_clients


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
    if provider.startswith("minerupro"):
        normalized_output, document, label = normalize_mineru_payload(
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


def _build_qwen_review_document(
    base_document: CanonicalDocument,
    issues: list[Any],
    patch_decisions: list[Any],
    image_task: ImageTask,
) -> tuple[CanonicalDocument, Any | None]:
    effective_decisions = [
        decision
        for decision in patch_decisions
        if str(decision.decision or "").strip() not in {"keep_mineru", "reject_issue"}
        and str(decision.reason or "").strip()
        not in {"llm_patch_unavailable", "llm_patch_invalid_json"}
    ]
    if not effective_decisions:
        return empty_document(
            image_task=image_task, source="qwen_second_pass_empty"
        ), None

    qwen_document = apply_patch_decisions(
        mineru_document=base_document.model_copy(deep=True),
        issues=issues,
        patch_decisions=effective_decisions,
    )
    qwen_document.source = "qwen_second_pass"
    return qwen_document, derive_label_from_document(qwen_document)


def process_image_task(
    image_task: ImageTask,
    args: argparse.Namespace,
    primary_client: BaseLocalClient | None,
    qwen_client: BaseLocalClient | None,
    recognition_prompt: str,
    seal_adjudication_prompt: str,
    flowchart_adjudication_prompt: str,
    output_dir: Path,
) -> dict[str, Any]:
    mineru_output = call_with_retry(
        client=primary_client,
        image_task=image_task,
        prompt=recognition_prompt,
        retry=args.retry,
    )
    (
        mineru_output,
        mineru_document,
        mineru_label,
    ) = _normalize_primary_output(
        image_task=image_task,
        client=primary_client,
        output=mineru_output,
    )

    qwen_output: ModelOutput | None = None
    qwen_document = empty_document(
        image_task=image_task, source="qwen_second_pass_not_triggered"
    )
    qwen_label = None

    seal_issues: list[Any] = []
    seal_patch_decisions: list[Any] = []
    seal_patch_outputs: list[ModelOutput] = []

    flowchart_issues = detect_flowchart_second_pass_issues(
        image_task=image_task,
        mineru_document=mineru_document,
        mineru_label=mineru_label,
    )
    flowchart_patch_decisions, flowchart_patch_outputs = adjudicate_issues_with_llm(
        client=qwen_client,
        image_task=image_task,
        prompt=flowchart_adjudication_prompt,
        issues=flowchart_issues,
        mode="flowchart_adjudication",
        retry=args.retry,
    )
    if flowchart_patch_outputs:
        qwen_output = flowchart_patch_outputs[0]
    qwen_document, qwen_label = _build_qwen_review_document(
        base_document=mineru_document,
        issues=flowchart_issues,
        patch_decisions=flowchart_patch_decisions,
        image_task=image_task,
    )

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

    all_issues = seal_issues + flowchart_issues
    patch_decisions = seal_patch_decisions + flowchart_patch_decisions
    if patch_decisions:
        mineru_document = apply_patch_decisions(
            mineru_document=mineru_document,
            issues=all_issues,
            patch_decisions=patch_decisions,
        )
        mineru_label = derive_label_from_document(mineru_document)

    artifact = adjudicate_documents(
        image_task=image_task,
        mineru_document=mineru_document,
        qwen_document=qwen_document,
        mineru_label=mineru_label,
        qwen_label=qwen_label,
        mineru_output=mineru_output,
        qwen_output=qwen_output,
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
    primary_clients = pick_primary_clients(clients)
    mineru_client = pick_client(primary_clients, "minerupro")
    qwen_client = pick_client(clients, "qwen")
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
            process_image_task(
                image_task=image_task,
                args=args,
                primary_client=mineru_client,
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
