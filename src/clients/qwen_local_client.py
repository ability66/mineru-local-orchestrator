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
        self.base_url = self._read_text_config(
            "base_url", fallback="http://127.0.0.1:18081/v1"
        )
        self.endpoint = self._read_text_config("endpoint", fallback="/chat/completions")
        self.model = self._read_text_config("model", fallback=model_name) or model_name
        self.input_mode = self._read_text_config(
            "input_mode", fallback="vision_openai_chat"
        )
        self.api_key_env = self._read_text_config("api_key_env")
        self.default_api_key = self._read_text_config(
            "default_api_key", fallback="EMPTY"
        )
        self.timeout = self._read_int_config("timeout", default=180)
        self.max_tokens = self._read_int_config("max_tokens", default=4096)
        self.temperature = self._read_float_config("temperature", default=0.0)

    def _analyze_impl(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        request_control = self._build_request_control(
            context=context,
            disable_thinking_applied=self._should_disable_thinking(context),
            fallback_used=False,
        )
        payload = self._build_payload(
            image_task=image_task,
            prompt=prompt,
            context=context,
            disable_thinking=request_control["disable_thinking_applied"],
        )
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
                "parsed": {"_request_control": dict(request_control)},
            }

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if self._should_retry_without_disable_thinking(
                response=response,
                request_control=request_control,
            ):
                fallback_control = self._build_request_control(
                    context=context,
                    disable_thinking_applied=False,
                    fallback_used=True,
                )
                fallback_payload = self._build_payload(
                    image_task=image_task,
                    prompt=prompt,
                    context=context,
                    disable_thinking=False,
                )
                try:
                    fallback_response = requests.post(
                        self._build_url(),
                        headers=headers,
                        json=fallback_payload,
                        timeout=self.timeout,
                    )
                except requests.RequestException as fallback_exc:
                    return {
                        "success": False,
                        "raw_text": "",
                        "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                        "parsed": {"_request_control": dict(fallback_control)},
                    }

                try:
                    fallback_response.raise_for_status()
                except requests.HTTPError as fallback_exc:
                    return {
                        "success": False,
                        "raw_text": fallback_response.text,
                        "error": f"{type(fallback_exc).__name__}: {fallback_exc}",
                        "parsed": {"_request_control": dict(fallback_control)},
                    }

                try:
                    fallback_json = fallback_response.json()
                except ValueError as fallback_exc:
                    return {
                        "success": False,
                        "raw_text": fallback_response.text,
                        "error": f"Invalid JSON response: {fallback_exc}",
                        "parsed": {"_request_control": dict(fallback_control)},
                    }

                fallback_json = self._attach_request_control(
                    payload=fallback_json,
                    request_control=fallback_control,
                )
                raw_text = self._extract_message_text(fallback_json)
                return {
                    "success": True,
                    "raw_text": raw_text,
                    "parsed": fallback_json,
                    "error": None,
                }
            return {
                "success": False,
                "raw_text": response.text,
                "error": f"{type(exc).__name__}: {exc}",
                "parsed": {"_request_control": dict(request_control)},
            }

        try:
            response_json = response.json()
        except ValueError as exc:
            return {
                "success": False,
                "raw_text": response.text,
                "error": f"Invalid JSON response: {exc}",
                "parsed": {"_request_control": dict(request_control)},
            }

        response_json = self._attach_request_control(
            payload=response_json,
            request_control=request_control,
        )
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
        disable_thinking: bool = False,
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

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if disable_thinking:
            payload["extra_body"] = {"enable_thinking": False}
        return payload

    def _build_text_only_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        base = [prompt]
        if context:
            base.append(self._format_context_block(context))
        return "\n\n".join(part for part in base if part.strip())

    def _format_context_block(self, context: dict[str, Any]) -> str:
        mode = str(context.get("mode", "") or "").strip().lower()
        if mode == "seal_adjudication":
            selection_payload = context.get("selection_payload")
            if selection_payload is not None:
                serialized = json.dumps(selection_payload, ensure_ascii=False, indent=2)
                return (
                    "以下是一个需要终裁的印章候选集合，请在候选中选择最可信的一项。"
                    "如果所有候选都不可信，请输出 review。"
                    "请只输出终裁 JSON，不要输出解释性正文：\n"
                    f"{serialized}"
                )
            issue_payload = context.get("issue_payload")
            serialized = json.dumps(issue_payload, ensure_ascii=False, indent=2)
            return (
                "以下是一个需要局部仲裁的印章 issue，请只输出 patch 决策 JSON，不要输出解释性正文：\n"
                f"{serialized}"
            )
        if mode == "flowchart_adjudication":
            issue_payload = context.get("issue_payload")
            serialized = json.dumps(issue_payload, ensure_ascii=False, indent=2)
            return (
                "以下是一个需要局部仲裁的流程图 issue，请重点检查冲突项并输出 patch 决策 JSON。"
                "如果提供了 ocr_reference_texts，它们只可用于文字校对，不可用于结构推断。"
                "请只输出 patch 决策 JSON，不要输出解释性正文：\n"
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
            return self._extract_text_payload(response_json.get("output_text"))

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""

        message = first_choice.get("message")
        if isinstance(message, dict):
            for key in ("content", "text", "output_text"):
                extracted = self._extract_text_payload(message.get(key))
                if extracted:
                    return extracted

        for key in ("text", "output_text"):
            extracted = self._extract_text_payload(first_choice.get(key))
            if extracted:
                return extracted
        return self._extract_text_payload(response_json.get("output_text"))

    def _extract_text_payload(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            for key in ("text", "content", "value", "output_text"):
                extracted = self._extract_text_payload(payload.get(key))
                if extracted:
                    return extracted
            return ""
        if isinstance(payload, list):
            fragments: list[str] = []
            for item in payload:
                extracted = self._extract_text_payload(item)
                if extracted:
                    fragments.append(extracted)
            return "\n".join(fragment for fragment in fragments if fragment)
        return ""

    def _should_disable_thinking(self, context: dict[str, Any]) -> bool:
        mode = str(context.get("mode", "") or "").strip().lower()
        return mode == "flowchart_adjudication"

    def _build_request_control(
        self,
        context: dict[str, Any],
        disable_thinking_applied: bool,
        fallback_used: bool,
    ) -> dict[str, Any]:
        mode = str(context.get("mode", "") or "").strip().lower() or "default"
        disable_requested = self._should_disable_thinking(context)
        thinking_mode = "default"
        if disable_requested and disable_thinking_applied:
            thinking_mode = "disabled_requested"
        elif disable_requested and fallback_used:
            thinking_mode = "disabled_requested_fallback_to_default"

        return {
            "mode": mode,
            "thinking_mode": thinking_mode,
            "disable_thinking_requested": disable_requested,
            "disable_thinking_applied": disable_thinking_applied,
            "disable_thinking_fallback_used": fallback_used,
            "disable_thinking_control": "extra_body.enable_thinking=false"
            if disable_requested
            else "",
        }

    def _attach_request_control(
        self,
        payload: Any,
        request_control: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(payload, dict):
            enriched = dict(payload)
            enriched["_request_control"] = dict(request_control)
            return enriched
        return {"response": payload, "_request_control": dict(request_control)}

    def _should_retry_without_disable_thinking(
        self,
        response: requests.Response,
        request_control: dict[str, Any],
    ) -> bool:
        status_code = int(getattr(response, "status_code", 0) or 0)
        return bool(
            request_control.get("disable_thinking_applied")
            and status_code in {400, 422}
        )
