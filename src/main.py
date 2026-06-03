from __future__ import annotations

import argparse
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
from src.pipeline.issues import detect_seal_issues
from src.pipeline.llm_adjudicator import adjudicate_issues_with_llm
from src.pipeline.normalizers import derive_label_from_document, normalize_mineru_payload, normalize_qwen_payload
from src.pipeline.patches import apply_patch_decisions
from src.prompt_builder import load_prompt
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
    parser.add_argument("--models-config", type=Path, default=Path("configs/models.local.yaml"))
    parser.add_argument("--prompts-config", type=Path, default=Path("configs/prompts.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=180)
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


def build_clients(model_configs: list[dict[str, Any]], request_timeout: int) -> list[BaseLocalClient]:
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


def pick_client(clients: list[BaseLocalClient], provider_prefix: str) -> BaseLocalClient | None:
    for client in clients:
        provider = str(client.config.get("provider", "")).strip().lower()
        if provider.startswith(provider_prefix):
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


def main() -> None:
    args = parse_args()
    if args.overwrite:
        clear_previous_outputs(args.output_dir)

    summary_path = initialize_summary_file(args.output_dir)
    model_configs = load_model_configs(args.models_config)
    clients = build_clients(model_configs=model_configs, request_timeout=args.request_timeout)
    mineru_client = pick_client(clients, "minerupro")
    qwen_client = pick_client(clients, "qwen")
    recognition_prompt = load_prompt(args.prompts_config, "qwen_recognition_prompt")
    adjudication_prompt = load_prompt(args.prompts_config, "qwen_adjudication_prompt")

    image_tasks = load_image_tasks(args.data_dir)
    if args.limit is not None:
        image_tasks = image_tasks[: args.limit]
    if not image_tasks:
        print("No images found in data directory")
        return

    for image_task in tqdm(image_tasks, desc="Processing images"):
        mineru_output = call_with_retry(
            client=mineru_client,
            image_task=image_task,
            prompt=recognition_prompt,
            retry=args.retry,
        )
        if mineru_output is not None:
            mineru_output, mineru_document, mineru_label = normalize_mineru_payload(
                image_task=image_task,
                model_output=mineru_output,
            )
        else:
            mineru_document = empty_document(image_task=image_task, source="mineru_unconfigured")
            mineru_label = None

        qwen_output = call_with_retry(
            client=qwen_client,
            image_task=image_task,
            prompt=recognition_prompt,
            retry=args.retry,
        )
        if qwen_output is not None:
            qwen_output, qwen_document, qwen_label = normalize_qwen_payload(
                image_task=image_task,
                model_output=qwen_output,
            )
        else:
            qwen_document = empty_document(image_task=image_task, source="qwen_unconfigured")
            qwen_label = None

        seal_issues = detect_seal_issues(
            image_task=image_task,
            mineru_document=mineru_document,
            qwen_document=qwen_document,
        )
        patch_decisions, _patch_outputs = adjudicate_issues_with_llm(
            client=qwen_client,
            image_task=image_task,
            prompt=adjudication_prompt,
            issues=seal_issues,
            retry=args.retry,
        )
        if patch_decisions:
            mineru_document = apply_patch_decisions(
                mineru_document=mineru_document,
                issues=seal_issues,
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
        )
        artifact.issues = seal_issues
        artifact.patch_decisions = patch_decisions
        summary_record = write_image_result(
            output_dir=args.output_dir,
            image_task=image_task,
            mineru_output=mineru_output,
            qwen_output=qwen_output,
            mineru_document=mineru_document,
            qwen_document=qwen_document,
            mineru_label=mineru_label,
            qwen_label=qwen_label,
            artifact=artifact,
        )
        append_summary_record(summary_path, summary_record)

    print(f"Processed {len(image_tasks)} images")


if __name__ == "__main__":
    main()
