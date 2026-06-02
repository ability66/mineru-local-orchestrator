from __future__ import annotations

import base64
import json
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


class MinerUProClient(BaseLocalClient):
    def __init__(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, config=config)
        self.base_url = self._read_text_config("base_url", fallback="http://127.0.0.1:18080")
        self.endpoint = self._read_text_config("endpoint", fallback="/analyze")
        self.request_mode = self._read_text_config("request_mode", fallback="multipart_file")
        self.file_field = self._read_text_config("file_field", fallback="file")
        self.send_prompt = self._read_bool_config("send_prompt", default=False)
        self.prompt_field = self._read_text_config("prompt_field", fallback="prompt")
        self.timeout = self._read_int_config("timeout", default=180)
        self.extra_fields = self.config.get("extra_fields", {}) or {}

    def _analyze_impl(
        self,
        image_task: ImageTask,
        prompt: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del context
        url = self._build_url()
        image_path = Path(image_task.image_path)

        try:
            if self.request_mode == "multipart_file":
                response = self._post_multipart(url=url, image_path=image_path, prompt=prompt)
            elif self.request_mode == "base64_json":
                response = self._post_base64_json(url=url, image_path=image_path, prompt=prompt)
            else:
                return {
                    "success": False,
                    "raw_text": "",
                    "error": f"Unsupported MinerU request_mode: {self.request_mode}",
                }
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
            payload = response.json()
            raw_text = json.dumps(payload, ensure_ascii=False)
        except ValueError:
            payload = None
            raw_text = response.text

        return {
            "success": response.ok,
            "raw_text": raw_text,
            "parsed": payload,
            "error": None if response.ok else f"HTTP {response.status_code}",
        }

    def _build_url(self) -> str:
        base = str(self.base_url or "").rstrip("/")
        endpoint = str(self.endpoint or "").strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{base}{endpoint}"

    def _post_multipart(self, url: str, image_path: Path, prompt: str) -> requests.Response:
        data = dict(self.extra_fields) if isinstance(self.extra_fields, dict) else {}
        if self.send_prompt:
            data[self.prompt_field or "prompt"] = prompt

        with image_path.open("rb") as file_obj:
            files = {
                self.file_field or "file": (
                    image_path.name,
                    file_obj,
                    _mime_type_for_path(image_path),
                )
            }
            return requests.post(url, data=data, files=files, timeout=self.timeout)

    def _post_base64_json(self, url: str, image_path: Path, prompt: str) -> requests.Response:
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        payload = dict(self.extra_fields) if isinstance(self.extra_fields, dict) else {}
        payload["file_name"] = image_path.name
        payload["image_base64"] = encoded
        payload["mime_type"] = _mime_type_for_path(image_path)
        if self.send_prompt:
            payload[self.prompt_field or "prompt"] = prompt
        return requests.post(url, json=payload, timeout=self.timeout)


def _mime_type_for_path(path: Path) -> str:
    return _MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")

