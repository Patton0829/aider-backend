"""GLM client retained for code conversion and backward-compatible callers."""
from typing import Any

from app.core.config import Settings
from app.services.knowledge_llm import KnowledgeLlmService


class GlmService(KnowledgeLlmService):
    def __init__(self, settings: Settings, client: Any | None = None):
        super().__init__(settings, client=client, provider="glm")
