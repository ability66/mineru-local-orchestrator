from __future__ import annotations

from typing import Any

from src.clients.minerupro_client import MinerUProClient


class PaddleLocalClient(MinerUProClient):
    def __init__(self, model_name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(model_name=model_name, config=config)
        self.base_url = self._read_text_config(
            "base_url", fallback="http://127.0.0.1:18083"
        )
