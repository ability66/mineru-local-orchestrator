from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.clients.glm_local_client import GLMLocalClient
from src.normalizer import _extract_first_json_object, _strip_code_fences
from src.pipeline.normalizers import _extract_document_payload, normalize_qwen_payload
from src.prompt_builder import load_prompt
from src.schema import ImageTask

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.local.yaml"
DEFAULT_PROMPTS_CONFIG = REPO_ROOT / "configs" / "prompts.yaml"
REAL_TEST_ENABLE_ENV = "RUN_GLM_REAL_SERVER_TEST"
REAL_TEST_IMAGE_ENV = "GLM_REAL_TEST_IMAGE"
REAL_TEST_MODELS_CONFIG_ENV = "GLM_REAL_TEST_MODELS_CONFIG"
REAL_TEST_PROMPTS_CONFIG_ENV = "GLM_REAL_TEST_PROMPTS_CONFIG"


def test_glm_real_server_transport_and_normalization() -> None:
    if not _is_truthy(os.getenv(REAL_TEST_ENABLE_ENV, "")):
        pytest.skip(
            f"set {REAL_TEST_ENABLE_ENV}=1 to run the real GLM integration test"
        )
    if yaml is None:
        pytest.skip("PyYAML is required for the real GLM integration test")

    models_config_path = Path(
        os.getenv(REAL_TEST_MODELS_CONFIG_ENV, str(DEFAULT_MODELS_CONFIG))
    )
    prompts_config_path = Path(
        os.getenv(REAL_TEST_PROMPTS_CONFIG_ENV, str(DEFAULT_PROMPTS_CONFIG))
    )
    glm_config = _load_glm_config(models_config_path=models_config_path)
    image_path = _resolve_test_image_path()
    prompt = _load_recognition_prompt(prompts_config_path=prompts_config_path)

    client = GLMLocalClient(
        model_name=str(glm_config.get("name", "glm-real-test")),
        config=glm_config,
    )
    image_task = ImageTask(
        image_id=image_path.stem,
        image_path=str(image_path),
        file_name=image_path.name,
        file_ext=image_path.suffix,
    )

    output = client.analyze(
        image_task=image_task,
        prompt=prompt,
        context={},
    )
    normalized_output, document, label = normalize_qwen_payload(
        image_task=image_task,
        model_output=output,
    )

    raw_message_payload = _parse_message_json(output.raw_text)
    recognized_document_payload = _extract_document_payload(raw_message_payload)
    diagnostics = _build_diagnostics(
        image_path=image_path,
        models_config_path=models_config_path,
        prompts_config_path=prompts_config_path,
        output=output,
        normalized_output=normalized_output,
        raw_message_payload=raw_message_payload,
        recognized_document_payload=recognized_document_payload,
        document=document,
        label=label.model_dump() if label is not None else None,
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    assert output.success, (
        "GLM transport call failed. "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )
    assert output.raw_text.strip(), (
        "GLM returned success=true but raw_text is empty. "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )
    assert isinstance(raw_message_payload, dict), (
        "GLM message content is not a JSON object. "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )
    assert recognized_document_payload is not None, (
        "GLM raw_text contains JSON, but local document extraction did not "
        "recognize its schema. "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )
    assert document.blocks, (
        "GLM transport succeeded, but normalized document has no blocks. "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_glm_config(models_config_path: Path) -> dict[str, Any]:
    if not models_config_path.exists():
        pytest.skip(f"models config not found: {models_config_path}")
    payload = yaml.safe_load(models_config_path.read_text(encoding="utf-8")) or {}
    models = payload.get("models")
    if not isinstance(models, list):
        pytest.skip(f"invalid models config: {models_config_path}")
    for item in models:
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider", "") or "").strip().lower()
        role = str(item.get("role", "") or "").strip().lower()
        if provider.startswith("glm") or role == "glm":
            return dict(item)
    pytest.skip(f"glm config not found in {models_config_path}")


def _load_recognition_prompt(prompts_config_path: Path) -> str:
    if not prompts_config_path.exists():
        return "你是文档视觉解析模型。请只输出 JSON。"
    return load_prompt(
        config_path=prompts_config_path,
        prompt_name="qwen_recognition_prompt",
    )


def _resolve_test_image_path() -> Path:
    override = os.getenv(REAL_TEST_IMAGE_ENV, "").strip()
    if override:
        image_path = Path(override).expanduser()
        assert image_path.exists(), f"{REAL_TEST_IMAGE_ENV} does not exist: {image_path}"
        assert image_path.is_file(), f"{REAL_TEST_IMAGE_ENV} is not a file: {image_path}"
        return image_path

    preferred_paths = [
        REPO_ROOT / "data" / "flowchart_crops" / "figure1.png",
        REPO_ROOT / "data" / "hard" / "hard.png",
        REPO_ROOT / "data" / "chart" / "0.jpg",
        REPO_ROOT / "data" / "stamp" / "circle_Aug09869.png",
    ]
    for path in preferred_paths:
        if path.exists() and path.is_file():
            return path

    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        for path in sorted((REPO_ROOT / "data").rglob(pattern)):
            if path.is_file():
                return path
    pytest.skip(
        "no test image found under data/; set GLM_REAL_TEST_IMAGE to an existing file"
    )


def _parse_message_json(raw_text: str) -> dict[str, Any] | None:
    cleaned_text = _strip_code_fences(raw_text)
    json_text, _ = _extract_first_json_object(cleaned_text)
    if json_text is None:
        return None
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _build_diagnostics(
    image_path: Path,
    models_config_path: Path,
    prompts_config_path: Path,
    output: Any,
    normalized_output: Any,
    raw_message_payload: dict[str, Any] | None,
    recognized_document_payload: Any,
    document: Any,
    label: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed_top_level_keys: list[str] = []
    if isinstance(output.parsed, dict):
        parsed_top_level_keys = sorted(str(key) for key in output.parsed.keys())

    message_payload_keys: list[str] = []
    if isinstance(raw_message_payload, dict):
        message_payload_keys = sorted(str(key) for key in raw_message_payload.keys())

    return {
        "image_path": str(image_path),
        "models_config_path": str(models_config_path),
        "prompts_config_path": str(prompts_config_path),
        "transport": {
            "success": bool(output.success),
            "error": output.error,
            "latency_ms": output.latency_ms,
            "model_name": output.model_name,
            "vendor": output.vendor,
            "source_type": output.source_type,
        },
        "raw_text": {
            "length": len(str(output.raw_text or "")),
            "preview": _truncate_text(str(output.raw_text or ""), 1200),
        },
        "parsed_response": {
            "top_level_keys": parsed_top_level_keys,
        },
        "message_payload": {
            "is_json_object": isinstance(raw_message_payload, dict),
            "top_level_keys": message_payload_keys,
            "recognized_document_payload_kind": _payload_kind(recognized_document_payload),
        },
        "normalized": {
            "success": bool(normalized_output.success),
            "error": normalized_output.error,
            "block_count": len(getattr(document, "blocks", []) or []),
            "warnings": list(getattr(document, "warnings", []) or []),
            "label": label,
        },
    }


def _payload_kind(payload: Any) -> str:
    if payload is None:
        return "none"
    if isinstance(payload, list):
        return "list"
    if isinstance(payload, dict):
        return "dict"
    return type(payload).__name__


def _truncate_text(text: str, limit: int) -> str:
    normalized = str(text or "")
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "\n...<truncated>..."
