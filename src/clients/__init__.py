from src.clients.base import BaseLocalClient
from src.clients.minerupro_client import MinerUProClient
from src.clients.qwen_local_client import QwenLocalClient

CLIENT_REGISTRY = {
    "minerupro_local": MinerUProClient,
    "qwen_openai_compatible": QwenLocalClient,
}

__all__ = [
    "BaseLocalClient",
    "MinerUProClient",
    "QwenLocalClient",
    "CLIENT_REGISTRY",
]

