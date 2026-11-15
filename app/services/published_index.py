"""Build immutable generations, validate storage, then atomically publish one pointer."""
import asyncio
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from app.core.config import Settings
from app.services.index_catalog import IndexCatalog
from app.services.knowledge import Bm25Index, KnowledgeChunk, KnowledgeService, KnowledgeUnavailableError
from app.services.vector_store import VectorStore


@dataclass(frozen=True)
class PublishedIndex:
    version: str
    collection: str
    profile: str
    chunks: tuple[KnowledgeChunk, ...]
    documents: tuple[dict, ...]
    keyword: Bm25Index


class PublishedIndexManager:
    def __init__(
        self, corpus: KnowledgeService, settings: Settings, profile: str,
        store: VectorStore,
        vector_builder: Callable[[tuple[KnowledgeChunk, ...]], Awaitable[tuple]],
    ):
        self.corpus = corpus
        self.settings = settings
        self.profile = profile
        self.store = store
        self.vector_builder = vector_builder
        self.catalog = IndexCatalog(settings.knowledge_index_catalog_path)
        self._lock = asyncio.Lock()
        self._snapshots: dict[str, PublishedIndex] = {}
        self._validated: set[str] = set()
        self._failed_source: str | None = None
        self._update_error: str | None = None

    def _capture(self) -> tuple[str, tuple[KnowledgeChunk, ...], list[dict]]:
        info = self.corpus.info()
        if info["loadError"] and info["loadError"] != "knowledge base is empty":
            raise KnowledgeUnavailableError("知识库源文件解析失败，尚未发布新索引")
        chunks = tuple(self.corpus.chunks)
        documents = info["documents"]
        fingerprint = json.dumps({
            "format": "published-index-v1", "profile": self.profile,
            "chunks": [asdict(chunk) for chunk in chunks],
            "documents": [{k: d[k] for k in ["name", "size", "chunkCount"]} for d in documents],
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(fingerprint.encode()).hexdigest(), chunks, documents

    def _snapshot(self, row: dict) -> PublishedIndex:
        version = row["version"]
        if version not in self._snapshots:
            chunks = tuple(KnowledgeChunk(**chunk) for chunk in row["chunks"])
            self._snapshots[version] = PublishedIndex(
                version, row["collection_name"], row["profile"], chunks,
                tuple(row["documents"]), Bm25Index(chunks),
            )
        return self._snapshots[version]

    def active(self) -> PublishedIndex | None:
        version = self.catalog.state()["active_version"]
        row = self.catalog.get(version) if version else None
        if row is None or row["status"] != "ready" or row["profile"] != self.profile:
            return None
        return self._snapshot(row)

    async def _validate(self, snapshot: PublishedIndex) -> bool:
        if snapshot.version not in self._validated:
            valid = await asyncio.to_thread(
                self.store.valid, snapshot.collection,
                self.settings.knowledge_embedding_dimensions, len(snapshot.chunks),
            )
            if not valid:
                return False
            self._validated.add(snapshot.version)
        return True

    async def prepare(self, *, force: bool = False) -> PublishedIndex:
        async with self._lock:
            version = None
            try:
                version, chunks, documents = self._capture()
                state = self.catalog.state()
                current = self.active()
                if (
                    not force and version == state["source_version"]
                    and current is not None and await self._validate(current)
                ):
                    return current
                row = self.catalog.get(version)
                if row is not None and row["status"] == "ready" and row["profile"] == self.profile:
                    existing = self._snapshot(row)
                    if force:
                        self._validated.discard(version)
                    if await self._validate(existing):
                        self.catalog.activate(version, observe_source=True)
                        self._failed_source = None
                        self._update_error = None
                        return existing
                # Physical repairs also use a fresh collection so in-flight snapshots are untouched.
                repair = row is not None and row["status"] == "ready"
                collection = "fault_kb_" + version + ("_repair_" + uuid.uuid4().hex[:12] if repair else "")
                if not repair:
                    self.catalog.begin(version, collection, self.profile, [asdict(c) for c in chunks], documents)
                vectors, reused, new = await self.vector_builder(chunks)
                rows = [
                    {
                        "id": chunk.chunk_id, "vector": vector,
                        "document_id": chunk.document_id, "entry_id": chunk.entry_id or "",
                        "document": chunk.document, "chapter": chunk.chapter, "section": chunk.section,
                        "line_start": chunk.line_start, "line_end": chunk.line_end,
                        "content_hash": chunk.content_hash, "kb_version": version,
                    }
                    for chunk, vector in zip(chunks, vectors)
                ]
                if len(rows) != len(chunks):
                    raise RuntimeError("Snapshot and vector counts do not match")
                await asyncio.to_thread(
                    self.store.build, collection, self.settings.knowledge_embedding_dimensions, rows,
                )
                if repair:
                    self.catalog.replace_collection(version, collection, reused, new)
                    self._snapshots.pop(version, None)
                else:
                    self.catalog.ready(version, reused, new)
                self.catalog.activate(version, observe_source=True)
                snapshot = self._snapshot(self.catalog.get(version))
                self._validated.add(version)
                self._failed_source = None
                self._update_error = None
                return snapshot
            except Exception as error:
                if version is not None:
                    self.catalog.fail(version, type(error).__name__)
                self._failed_source = version
                self._update_error = "新索引构建失败，已保留上一发布版本；请检查索引状态并重建"
                previous = self.active()
                if previous is not None:
                    await self._validate(previous)
                raise

    async def for_query(self) -> PublishedIndex:
        # A failed new build must not make every query wait on the same failed API again.
        current = self.active()
        try:
            version, _, _ = self._capture()
            if current is not None and self._update_error and version == self._failed_source:
                if await self._validate(current):
                    return current
            return await self.prepare()
        except Exception:
            if self._update_error is None:
                self._update_error = "知识库更新失败，继续使用上一发布版本"
            if current is not None and await self._validate(current):
                return current
            raise

    async def activate(self, version: str) -> PublishedIndex:
        async with self._lock:
            row = self.catalog.get(version)
            if row is None or row["status"] != "ready":
                raise ValueError("索引版本不存在或未完成构建")
            if row["profile"] != self.profile:
                raise ValueError("目标索引的向量模型配置与当前配置不一致")
            snapshot = self._snapshot(row)
            if not await self._validate(snapshot):
                raise ValueError("目标索引的向量存储校验失败")
            self.catalog.activate(version, observe_source=False)
            self._update_error = None
            self._failed_source = None
            return snapshot

    def versions(self) -> list[dict]:
        return self.catalog.list()

    def get_chunk(self, version: str, chunk_id: str) -> dict | None:
        row = self.catalog.get(version)
        if row is None or row["status"] != "ready":
            return None
        chunk = next((c for c in row["chunks"] if c["chunk_id"] == chunk_id), None)
        return dict(chunk, kb_version=version) if chunk else None

    def info(self) -> dict:
        current = self.active()
        ready = current is not None and current.version in self._validated
        result = {
            "vectorStore": "milvus_lite", "indexVersion": current.version if current else None,
            "indexCollection": current.collection if current else None,
            "indexSourceVersion": self.catalog.state()["source_version"],
            "indexUpdateError": self._update_error,
            "indexedChunkCount": len(current.chunks) if ready else 0,
            "retrievalReady": ready and bool(current.chunks),
            "retrievalError": None if ready else "尚无已校验的向量索引版本",
        }
        if ready:
            documents = [dict(d) for d in current.documents]
            result.update({
                "ready": bool(current.chunks), "chunkCount": len(current.chunks),
                "documentCount": len(documents), "documents": documents, "loadError": None,
                "document": documents[0]["name"] if len(documents) == 1 else f"{len(documents)} 个知识库文档",
            })
        return result

    async def close(self):
        await asyncio.to_thread(self.store.close)
