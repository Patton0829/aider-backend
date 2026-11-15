"""Replaceable API adapters for embedding and reranking; no retrieval policy here."""
import math
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from app.core.config import Settings


class RetrievalModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankMatch:
    index: int
    score: float


class Embedder(Protocol):
    async def embed(
        self, texts: list[str], text_type: Literal["document", "query"]
    ) -> list[list[float]]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankMatch]: ...


class DashScopeRetrievalModels:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        if not settings.dashscope_api_key.strip():
            raise RuntimeError("DASHSCOPE_API_KEY is required for embedding/reranking")
        self.settings = settings
        self.base_url = settings.retrieval_api_base_url
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=settings.knowledge_retrieval_timeout_seconds,
        )

    async def _post(self, path: str, body: dict) -> dict:
        try:
            response = await self.client.post(
                self.base_url + path,
                headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}"},
                json=body,
            )
            if not response.is_success:
                # Do not echo provider response bodies, credentials or source text.
                raise RetrievalModelError(f"百炼检索模型请求失败（HTTP {response.status_code}）")
            payload = response.json()
            if payload.get("code"):
                raise RetrievalModelError("百炼检索模型返回错误，请检查模型权限与额度")
            return payload
        except (httpx.HTTPError, ValueError) as error:
            raise RetrievalModelError("百炼检索模型连接失败或响应无效") from error

    async def embed(
        self, texts: list[str], text_type: Literal["document", "query"]
    ) -> list[list[float]]:
        if not texts:
            return []
        payload = await self._post(
            "/services/embeddings/text-embedding/text-embedding",
            {
                "model": self.settings.knowledge_embedding_model,
                "input": {"texts": texts},
                "parameters": {
                    "dimension": self.settings.knowledge_embedding_dimensions,
                    "text_type": text_type,
                },
            },
        )
        try:
            rows = payload["output"]["embeddings"]
            by_index = {row["text_index"]: row["embedding"] for row in rows}
            if len(rows) != len(texts) or set(by_index) != set(range(len(texts))):
                raise ValueError("embedding indices do not match inputs")
            vectors = [by_index[index] for index in range(len(texts))]
            for vector in vectors:
                if len(vector) != self.settings.knowledge_embedding_dimensions:
                    raise ValueError("unexpected embedding dimension")
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
                    raise ValueError("non-finite embedding")
            return vectors
        except (KeyError, TypeError, ValueError) as error:
            raise RetrievalModelError("百炼向量响应维度、数量或索引无效") from error

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankMatch]:
        if not documents:
            return []
        payload = await self._post(
            "/services/rerank/text-rerank/text-rerank",
            {
                "model": self.settings.knowledge_rerank_model,
                "input": {"query": query, "documents": documents},
                "parameters": {"top_n": min(top_n, len(documents))},
            },
        )
        try:
            matches = [
                RerankMatch(index=row["index"], score=float(row["relevance_score"]))
                for row in payload["output"]["results"]
            ]
            if not matches or len({m.index for m in matches}) != len(matches):
                raise ValueError("missing or duplicate rerank results")
            for match in matches:
                if not isinstance(match.index, int) or not 0 <= match.index < len(documents):
                    raise ValueError("invalid rerank index")
                if not math.isfinite(match.score) or not 0 <= match.score <= 1:
                    raise ValueError("invalid rerank score")
            return sorted(matches, key=lambda m: -m.score)[:top_n]
        except (KeyError, TypeError, ValueError) as error:
            raise RetrievalModelError("百炼重排响应的索引或分数无效") from error

    async def close(self):
        if self._owns_client:
            await self.client.aclose()
