from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLayoutClient(ABC):
    @abstractmethod
    def fetch_layout_payload(self, image_path: Path, relative_path: Path) -> Any:
        raise NotImplementedError

    def fetch_blocks(self, image_path: Path, relative_path: Path) -> list[dict[str, Any]]:
        payload = self.fetch_layout_payload(image_path=image_path, relative_path=relative_path)
        return extract_layout_blocks(payload)


class JsonLayoutClient(BaseLayoutClient):
    def __init__(self, layout_dir: Path) -> None:
        self.layout_dir = layout_dir

    def fetch_layout_payload(self, image_path: Path, relative_path: Path) -> Any:
        del image_path
        layout_path = self._resolve_layout_path(relative_path)
        return json.loads(layout_path.read_text(encoding="utf-8"))

    def _resolve_layout_path(self, relative_path: Path) -> Path:
        candidates = [
            self.layout_dir / relative_path.with_suffix(".json"),
            self.layout_dir / f"{relative_path.stem}.json",
            self.layout_dir / relative_path.stem / "layout.json",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Layout json not found for '{relative_path.as_posix()}' in '{self.layout_dir}'"
        )


class MinerUVLLayoutClient(BaseLayoutClient):
    def __init__(
        self,
        server_url: str,
        image_analysis: bool = True,
        backend: str = "http-client",
        model_name: str | None = None,
    ) -> None:
        self.server_url = server_url
        self.image_analysis = image_analysis
        self.backend = backend
        self.model_name = model_name
        self._thread_local = threading.local()

    def fetch_layout_payload(self, image_path: Path, relative_path: Path) -> Any:
        del relative_path
        from PIL import Image

        client = self._get_client()
        with Image.open(image_path) as image:
            layout_result = client.layout_detect(image)
            block_images, prompts, params, indices = client.helper.prepare_for_extract(
                image,
                layout_result,
                image_analysis=self.image_analysis,
            )
            outputs = client._batch_predict(block_images, prompts, params, None, None)
            for index, output in zip(indices, outputs):
                try:
                    layout_result[index].content = str(getattr(output, "text", "") or "")
                    layout_result[index].scored = getattr(output, "scored", None)
                except Exception:
                    continue
            processed_blocks = client.helper.post_process(layout_result)
        return {
            "blocks": _to_jsonable(processed_blocks),
            "layout_scored": _to_jsonable(getattr(layout_result, "layout_scored", None)),
        }

    def _get_client(self) -> Any:
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            return client

        try:
            from mineru_vl_utils import MinerUClient
        except ImportError as exc:
            raise RuntimeError(
                "mineru_vl_utils is required when --layout-source=mineru_vl"
            ) from exc

        kwargs: dict[str, Any] = {
            "backend": self.backend,
            "server_url": self.server_url,
            "image_analysis": self.image_analysis,
        }
        if str(self.model_name or "").strip():
            kwargs["model_name"] = str(self.model_name or "").strip()
        client = MinerUClient(**kwargs)
        self._thread_local.client = client
        return client


def extract_layout_blocks(payload: Any) -> list[dict[str, Any]]:
    blocks = _unwrap_blocks_payload(payload)
    return [item for item in blocks if isinstance(item, dict)]


def _unwrap_blocks_payload(payload: Any) -> list[Any]:
    current = payload
    visited: set[int] = set()
    wrapper_keys = ("blocks", "data", "result", "payload", "output", "response")

    while True:
        if isinstance(current, list):
            return current
        if not isinstance(current, dict):
            return []

        direct_blocks = current.get("blocks")
        if isinstance(direct_blocks, list):
            return direct_blocks

        for key in wrapper_keys:
            candidate = current.get(key)
            if not isinstance(candidate, (dict, list)):
                continue
            candidate_id = id(candidate)
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            current = candidate
            break
        else:
            return []


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)
