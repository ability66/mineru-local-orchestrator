from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLayoutClient(ABC):
    @abstractmethod
    def fetch_blocks(self, image_path: Path, relative_path: Path) -> list[dict[str, Any]]:
        raise NotImplementedError


class JsonLayoutClient(BaseLayoutClient):
    def __init__(self, layout_dir: Path) -> None:
        self.layout_dir = layout_dir

    def fetch_blocks(self, image_path: Path, relative_path: Path) -> list[dict[str, Any]]:
        del image_path
        layout_path = self._resolve_layout_path(relative_path)
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
        blocks = _unwrap_blocks_payload(payload)
        return [item for item in blocks if isinstance(item, dict)]

    def _resolve_layout_path(self, relative_path: Path) -> Path:
        candidates = [
            self.layout_dir / relative_path.with_suffix(".json"),
            self.layout_dir / f"{relative_path.stem}.json",
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Layout json not found for '{relative_path.as_posix()}' in '{self.layout_dir}'"
        )


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
