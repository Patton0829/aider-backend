"""Snapshot -> recall -> fusion -> rerank. Model adapters can be replaced independently."""
import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import replace

from app.core.config import Settings
from app.services.embedding_cache import EmbeddingCache
from app.services.knowledge import KnowledgeChunk, KnowledgeSearchResult, KnowledgeService
from app.services.rag_models import Embedder, Reranker, RetrievalModelError
from app.services.vector_store import MilvusVectorStore, VectorStore
from app.services.published_index import PublishedIndexManager


logger = logging.getLogger("uvicorn.error")


def chunk_text(chunk: KnowledgeChunk) -> str:
    return f"{chunk.document}\n{chunk.chapter}\n{chunk.section}\n{chunk.content}"


def normalize(vector: list[float]) -> list[float]:
    if not all(math.isfinite(value) for value in vector):
        raise RetrievalModelError("向量包含非有限数值")
    length = math.sqrt(sum(value * value for value in vector))
    if not length:
        raise RetrievalModelError("向量模型返回零向量")
    return [value / length for value in vector]


def reciprocal_rank_fusion(
    rankings: list[list[KnowledgeSearchResult]], top_k: int, k: int = 60,
) -> list[KnowledgeSearchResult]:
    """Fuse ranks, never raw BM25/cosine scores with incompatible scales."""
    scores: dict[str, float] = {}
    by_id = {}
    for ranking in rankings:
        seen = set()
        for rank, result in enumerate(ranking, 1):
            key = result.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            by_id[key] = result.chunk
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
    keys = sorted(scores, key=lambda key: (
        -scores[key], by_id[key].document.lower(), by_id[key].line_start,
    ))[:top_k]
    return [KnowledgeSearchResult(by_id[key], scores[key]) for key in keys]


class RagPipeline:
    def __init__(
        self, corpus: KnowledgeService, settings: Settings,
        embedder: Embedder | None = None, reranker: Reranker | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.corpus = corpus
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        if settings.knowledge_retrieval_mode != "keyword" and embedder is None:
            raise ValueError("Vector retrieval requires an embedder")
        if settings.knowledge_rerank_enabled and reranker is None:
            raise ValueError("Reranking is enabled but no reranker is configured")
        self.cache = (
            EmbeddingCache(settings.knowledge_embedding_cache_path, settings.knowledge_embedding_dimensions)
            if embedder is not None else None
        )
        self._index_lock = asyncio.Lock()
        self._keys: tuple[str, ...] = ()
        self._vectors: tuple[list[float], ...] = ()
        self._index_error: str | None = None
        self._last_embedding_counts = (0, 0)
        self._namespace = json.dumps([
            "chunk-format-v1", settings.retrieval_api_base_url if embedder else "",
            settings.knowledge_embedding_model, settings.knowledge_embedding_dimensions, "document",
        ])
        self.publisher = None
        if settings.knowledge_retrieval_mode != "keyword" and settings.knowledge_vector_store == "milvus_lite":
            store = vector_store or MilvusVectorStore(settings.knowledge_milvus_uri)
            self.publisher = PublishedIndexManager(
                corpus, settings, self._namespace, store, self._publication_vectors,
            )

    def _key(self, chunk: KnowledgeChunk) -> str:
        return hashlib.sha256((self._namespace + "\0" + chunk_text(chunk)).encode()).hexdigest()

    async def _ensure_index(self, chunks: tuple[KnowledgeChunk, ...]) -> tuple[list[float], ...]:
        keys = tuple(self._key(chunk) for chunk in chunks)
        async with self._index_lock:
            if keys == self._keys:
                self._last_embedding_counts = (len(chunks), 0)
                return self._vectors
            assert self.cache is not None and self.embedder is not None
            cached = await asyncio.to_thread(self.cache.get_many, list(keys))
            for key in list(cached):
                try:
                    cached[key] = normalize(cached[key])
                except RetrievalModelError:
                    del cached[key]
            missing = list(dict.fromkeys(key for key in keys if key not in cached))
            texts = {key: chunk_text(chunk) for key, chunk in zip(keys, chunks)}
            batch_size = self.settings.knowledge_embedding_batch_size
            try:
                for start in range(0, len(missing), batch_size):
                    batch = missing[start:start + batch_size]
                    vectors = await self.embedder.embed([texts[key] for key in batch], "document")
                    if len(vectors) != len(batch):
                        raise RetrievalModelError("文档向量数量不匹配")
                    if any(len(vector) != self.settings.knowledge_embedding_dimensions for vector in vectors):
                        raise RetrievalModelError("文档向量维度不匹配")
                    updated = {key: normalize(vector) for key, vector in zip(batch, vectors)}
                    await asyncio.to_thread(self.cache.put_many, updated)
                    cached.update(updated)
                self._vectors = tuple(cached[key] for key in keys)
                self._keys = keys
                self._index_error = None
                self._last_embedding_counts = (len(chunks) - len(missing), len(missing))
            except Exception:
                self._index_error = "向量索引构建失败，请检查模型连接、权限或额度"
                raise
            logger.info("RAG index ready: chunks=%d new_embeddings=%d", len(chunks), len(missing))
            return self._vectors

    async def _publication_vectors(self, chunks):
        vectors = await self._ensure_index(chunks)
        return vectors, *self._last_embedding_counts

    async def prepare_index(self):
        if self.publisher is not None:
            await self.publisher.prepare()
        elif self.settings.knowledge_retrieval_mode != "keyword" and self.corpus.chunks:
            await self._ensure_index(self.corpus.snapshot())

    async def rebuild_index(self):
        if self.publisher is None:
            raise ValueError("索引版本管理需要启用 Milvus Lite")
        await self.publisher.prepare(force=True)
        return self.info()

    async def activate_index(self, version: str):
        if self.publisher is None:
            raise ValueError("索引版本管理需要启用 Milvus Lite")
        await self.publisher.activate(version)
        return self.info()

    def index_versions(self):
        return self.publisher.versions() if self.publisher else []

    async def close(self):
        if self.publisher is not None:
            await self.publisher.close()

    async def search(self, query: str) -> list[KnowledgeSearchResult]:
        started = time.perf_counter()
        if not query.strip():
            return []
        published = await self.publisher.for_query() if self.publisher is not None else None
        chunks = published.chunks if published is not None else self.corpus.snapshot()
        if not chunks:
            return []
        mode = self.settings.knowledge_retrieval_mode
        limit = max(self.settings.knowledge_candidate_k, self.settings.knowledge_top_k)
        # Capture lexical results before awaiting APIs to keep one consistent corpus snapshot.
        keyword = published.keyword if published is not None else self.corpus
        lexical = keyword.search(query, self.settings.knowledge_bm25_recall_k) if mode != "vector" else []
        if mode == "keyword":
            candidates = lexical[:limit]
        else:
            assert self.embedder is not None
            vectors = await self._ensure_index(chunks) if published is None else None
            query_vectors = await self.embedder.embed([query], "query")
            if len(query_vectors) != 1:
                raise RetrievalModelError("查询向量数量不匹配")
            query_vector = normalize(query_vectors[0])
            if len(query_vector) != self.settings.knowledge_embedding_dimensions:
                raise RetrievalModelError("查询向量维度不匹配")
            if published is not None:
                hits = await asyncio.to_thread(
                    self.publisher.store.search, published.collection, query_vector,
                    min(self.settings.knowledge_dense_recall_k, len(chunks)),
                )
                by_id = {chunk.chunk_id: chunk for chunk in chunks}
                if any(hit.chunk_id not in by_id for hit in hits):
                    raise RetrievalModelError("向量结果与发布的知识库快照不一致")
                semantic = [KnowledgeSearchResult(by_id[hit.chunk_id], hit.score) for hit in hits]
            else:
                semantic = sorted([
                    KnowledgeSearchResult(chunk, sum(a * b for a, b in zip(vector, query_vector)))
                    for chunk, vector in zip(chunks, vectors)
                ], key=lambda result: -result.score)[:self.settings.knowledge_dense_recall_k]
            if mode == "vector":
                candidates = semantic[:limit]
            else:
                candidates = reciprocal_rank_fusion(
                    [lexical, semantic], limit, self.settings.knowledge_rrf_k,
                )
        recalled_at = time.perf_counter()
        if self.settings.knowledge_rerank_enabled and candidates:
            assert self.reranker is not None
            matches = await self.reranker.rerank(
                query, [chunk_text(result.chunk) for result in candidates],
                self.settings.knowledge_top_k,
            )
            results = [KnowledgeSearchResult(candidates[m.index].chunk, m.score) for m in matches]
        else:
            results = candidates[:self.settings.knowledge_top_k]
        if published is not None:
            results = [replace(result, kb_version=published.version) for result in results]
        logger.info(
            "RAG retrieval: mode=%s candidates=%d results=%d recall_ms=%.1f rerank_ms=%.1f total_ms=%.1f",
            mode, len(candidates), len(results), (recalled_at - started) * 1000,
            (time.perf_counter() - recalled_at) * 1000, (time.perf_counter() - started) * 1000,
        )
        return results

    def info(self) -> dict:
        vector_enabled = self.settings.knowledge_retrieval_mode != "keyword"
        current_keys = tuple(self._key(chunk) for chunk in self.corpus.chunks) if vector_enabled else ()
        current = current_keys == self._keys and bool(current_keys)
        result = {
            "retrievalMode": self.settings.knowledge_retrieval_mode,
            "bm25RecallK": self.settings.knowledge_bm25_recall_k,
            "denseRecallK": self.settings.knowledge_dense_recall_k,
            "candidateK": self.settings.knowledge_candidate_k,
            "rrfK": self.settings.knowledge_rrf_k,
            "embeddingModel": self.settings.knowledge_embedding_model if vector_enabled else None,
            "embeddingDimensions": self.settings.knowledge_embedding_dimensions if vector_enabled else None,
            "rerankModel": self.settings.knowledge_rerank_model if self.settings.knowledge_rerank_enabled else None,
            "indexedChunkCount": len(self._keys) if current else 0,
            "retrievalReady": current if vector_enabled else bool(self.corpus.chunks),
            "retrievalError": self._index_error,
            "vectorStore": "memory",
        }
        if self.publisher is not None:
            result.update(self.publisher.info())
        return result
