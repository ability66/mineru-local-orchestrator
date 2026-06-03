from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests

from src.clients.base import BaseLocalClient
from src.schema import ImageTask

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class QwenLocalClient(BaseLocalClient):
    def __init__(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, config=config)
        self.base_url = self._read_text_config("base_url", fallback="http://127.0.0.1:18081/v1")
        self.endpoint = self._read_text_config("endpoint", fallback="/chat/completions")
        self.model = self._read_text_config("model", fallback=model_name) or model_name
        self.input_mode = self._read_text_config("input_mode", fallback="vision_openai_chat")
        self.api_key_env = self._read_text_config("api_key_env")
        self.default_api_key = self._read_text_config("default_api_key", fallback="EMPTY")
        self.timeout = self._read_int_config("timeout", default=180)
        self.max_tokens = self._read_int_config("max_tokens", default=4096)
        self.temperature = self._read_float_config("temperature", default=0.0)

    def _analyze_impl(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._build_payload(image_task=image_task, prompt=prompt, context=context)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._resolve_api_key()}",
        }

        try:
            response = requests.post(
                self._build_url(),
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            return {
                "success": False,
                "raw_text": "",
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return {
                "success": False,
                "raw_text": response.text,
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            response_json = response.json()
        except ValueError as exc:
            return {
                "success": False,
                "raw_text": response.text,
                "error": f"Invalid JSON response: {exc}",
            }

        raw_text = self._extract_message_text(response_json)
        return {
            "success": True,
            "raw_text": raw_text,
            "parsed": response_json,
            "error": None,
        }

    def _build_payload(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if self.input_mode == "text_only_chat":
            user_text = self._build_text_only_prompt(prompt=prompt, context=context)
            messages = [{"role": "user", "content": user_text}]
        else:
            user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if context:
                user_content.append(
                    {
                        "type": "text",
                        "text": self._format_context_block(context),
                    }
                )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._build_image_data_url(Path(image_task.image_path))
                    },
                }
            )
            messages = [{"role": "user", "content": user_content}]

        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _build_text_only_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        base = [prompt]
        if context:
            base.append(self._format_context_block(context))
        return "\n\n".join(part for part in base if part.strip())

    def _format_context_block(self, context: dict[str, Any]) -> str:
        mode = str(context.get("mode", "") or "").strip().lower()
        if mode == "seal_adjudication":
            issue_payload = context.get("issue_payload")
            serialized = json.dumps(issue_payload, ensure_ascii=False, indent=2)
            return (
                "以下是一个需要局部仲裁的印章 issue，请只输出 patch 决策 JSON，不要输出解释性正文：\n"
                f"{serialized}"
            )

        mineru_payload = context.get("mineru_payload")
        if mineru_payload is not None:
            serialized = json.dumps(mineru_payload, ensure_ascii=False, indent=2)
            return (
                "以下是 MinerU 的初步结构化结果，请以纠偏和补充为主，不要机械重写：\n"
                f"{serialized}"
            )

        serialized = json.dumps(context, ensure_ascii=False, indent=2)
        return f"以下是附加上下文信息：\n{serialized}"

    def _build_url(self) -> str:
        base = str(self.base_url or "").rstrip("/")
        endpoint = str(self.endpoint or "").strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{base}{endpoint}"

    def _resolve_api_key(self) -> str:
        if self.api_key_env:
            value = os.getenv(self.api_key_env, "").strip()
            if value:
                return value
        return self.default_api_key or "EMPTY"

    def _build_image_data_url(self, image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        mime_type = _MIME_TYPES.get(suffix)
        if mime_type is None:
            raise ValueError(f"Unsupported image suffix: {suffix}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_message_text(self, response_json: dict[str, Any]) -> str:
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            return json.dumps(response_json, ensure_ascii=False)

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text)
            return "\n".join(fragments)
        return str(content)
