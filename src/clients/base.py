from __future__ import annotations

import json
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from src.schema import ImageTask, ModelOutput


class BaseLocalClient(ABC):
    def __init__(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.config = config or {}

    def analyze(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> ModelOutput:
        start = perf_counter()
        try:
            result = self._analyze_impl(
                image_task=image_task,
                prompt=prompt,
                context=context or {},
            )
            if isinstance(result, ModelOutput):
                output = result
            else:
                parsed_payload = result.get("parsed")
                raw_text = result.get("raw_text")
                if raw_text is None and parsed_payload is not None:
                    raw_text = json.dumps(parsed_payload, ensure_ascii=False)
                output = ModelOutput(
                    image_id=image_task.image_id,
                    model_name=self.model_name,
                    success=bool(result.get("success", False)),
                    raw_text=str(raw_text or ""),
                    parsed=parsed_payload,
                    error=self._normalize_optional_text(result.get("error")),
                    latency_ms=result.get("latency_ms"),
                    vendor=self._normalize_optional_text(result.get("vendor")),
                    source_type=self._normalize_optional_text(result.get("source_type")),
                )
        except Exception as exc:
            output = ModelOutput(
                image_id=image_task.image_id,
                model_name=self.model_name,
                success=False,
                raw_text="",
                error=f"{type(exc).__name__}: {exc}",
            )

        output.image_id = image_task.image_id
        output.model_name = self.model_name
        if not output.vendor:
            output.vendor = self._default_vendor()
        if not output.source_type:
            output.source_type = self._default_source_type()
        if output.latency_ms is None:
            output.latency_ms = int((perf_counter() - start) * 1000)
        return output

    @abstractmethod
    def _analyze_impl(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any],
    ) -> ModelOutput | dict[str, Any]:
        raise NotImplementedError

    def _default_vendor(self) -> str | None:
        return self._normalize_optional_text(
            self.config.get("vendor") or self.config.get("provider")
        )

    def _default_source_type(self) -> str | None:
        configured = self._normalize_optional_text(self.config.get("source_type"))
        if configured:
            return configured
        provider = self._normalize_optional_text(self.config.get("provider"))
        if provider and provider.endswith("_local"):
            return "local_service"
        return provider

    def _read_text_config(self, key: str, fallback: str | None = None) -> str | None:
        value = self.config.get(key, fallback)
        if value is None:
            return None
        text = str(value).strip()
        return text or fallback

    def _read_bool_config(self, key: str, default: bool) -> bool:
        value = self.config.get(key, default)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _read_int_config(self, key: str, default: int) -> int:
        value = self.config.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _read_float_config(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        text = str(value).strip() if value is not None else ""
        return text or None

