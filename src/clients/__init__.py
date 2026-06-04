from src.clients.base import BaseLocalClient
from src.clients.glm_local_client import GLMLocalClient
from src.clients.minerupro_client import MinerUProClient
from src.clients.paddle_local_client import PaddleLocalClient
from src.clients.qwen_local_client import QwenLocalClient

CLIENT_REGISTRY = {
    "glm_openai_compatible": GLMLocalClient,
    "minerupro_local": MinerUProClient,
    "paddle_local": PaddleLocalClient,
    "qwen_openai_compatible": QwenLocalClient,
}

__all__ = [
    "BaseLocalClient",
    "GLMLocalClient",
    "MinerUProClient",
    "PaddleLocalClient",
    "QwenLocalClient",
    "CLIENT_REGISTRY",
]
