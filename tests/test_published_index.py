import asyncio
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.knowledge import router
from app.core.config import Settings
from app.services.knowledge import KnowledgeService, parse_markdown
from app.services.rag_models import RetrievalModelError
from app.services.rag_pipeline import RagPipeline
from app.services.vector_store import MilvusVectorStore, VectorHit
from test_rag import FakeModels


class FakeVectorStore:
    def __init__(self, collections=None):
        self.collections = collections if collections is not None else {}
        self.fail_build = False
        self.searches = []

    def build(self, collection, dimensions, rows):
        self.collections[collection] = (dimensions, rows)
        if self.fail_build:
            raise RuntimeError("test failed staging build")

    def valid(self, collection, dimensions, count):
        existing = self.collections.get(collection)
        return existing is not None and existing[0] == dimensions and len(existing[1]) == count

    def search(self, collection, vector, top_k):
        self.searches.append(collection)
        return sorted([
            VectorHit(row["id"], sum(a * b for a, b in zip(row["vector"], vector)))
            for row in self.collections[collection][1]
        ], key=lambda hit: -hit.score)[:top_k]

    def close(self):
        pass


class StableChunkIdTest(unittest.TestCase):
    def test_inserting_earlier_text_and_moving_lines_preserves_existing_ids(self):
        text = "# KB\n\n## 道岔\n\n锁闭装置磨损。\n\n接点氧化。"
        old = parse_markdown(text, "kb.md")
        new = parse_markdown(text.replace("## 道岔", "新增前言。\n\n\n## 道岔"), "kb.md")
        by_content = {chunk.content: chunk for chunk in new}
        for chunk in old:
            self.assertEqual(chunk.chunk_id, by_content[chunk.content].chunk_id)
            self.assertEqual(chunk.document_id, by_content[chunk.content].document_id)
            self.assertNotEqual(chunk.line_start, by_content[chunk.content].line_start)

    def test_content_edits_change_only_affected_chunk_id(self):
        text = "## 道岔\n\n锁闭装置磨损。\n\n接点氧化。"
        old = parse_markdown(text, "kb.md")
        new = parse_markdown(text.replace("磨损", "卡阻"), "kb.md")
        self.assertNotEqual(old[0].chunk_id, new[0].chunk_id)
        self.assertNotEqual(old[0].content_hash, new[0].content_hash)
        self.assertEqual(old[1].chunk_id, new[1].chunk_id)

    def test_identical_paragraphs_have_distinct_stable_ids(self):
        text = "## 道岔\n\n重复内容。\n\n重复内容。"
        first = parse_markdown(text, "kb.md")
        second = parse_markdown("前言。\n\n" + text, "kb.md")
        self.assertEqual(len({chunk.chunk_id for chunk in first}), 2)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second[1:]])


class PublishedIndexTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.seed = self.root / "seed.md"
        self.seed.write_text("# KB\n\n## 道岔\n\n锁闭装置磨损。\n\n## 轴承\n\n外圈点蚀。", encoding="utf-8")
        self.corpus = KnowledgeService(self.seed, storage_dir=self.root / "docs")
        self.corpus.reload_if_changed(force=True)
        self.settings = Settings(
            knowledge_retrieval_mode="hybrid", knowledge_vector_store="milvus_lite",
            knowledge_embedding_dimensions=2, knowledge_top_k=5,
            knowledge_embedding_cache_path=self.root / "embeddings.sqlite3",
            knowledge_index_catalog_path=self.root / "catalog.sqlite3",
            knowledge_milvus_uri=self.root / "milvus.db",
        )
        self.models = FakeModels()
        self.store = FakeVectorStore()
        self.pipeline = self.make_pipeline()

    def make_pipeline(self, **changes):
        return RagPipeline(
            self.corpus, self.settings.model_copy(update=changes), self.models, self.models,
            vector_store=self.store,
        )

    async def asyncTearDown(self):
        await self.pipeline.close()
        self.directory.cleanup()

    async def test_publication_and_restart_reuse_vectors_and_milvus_collection(self):
        await self.pipeline.prepare_index()
        before = self.pipeline.info()
        self.assertTrue(before["retrievalReady"])
        self.assertEqual(before["vectorStore"], "milvus_lite")
        batches = len(self.models.document_batches)
        restarted = self.make_pipeline()
        await restarted.prepare_index()
        results = await restarted.search("锁闭故障")
        self.assertEqual(len(self.models.document_batches), batches)
        self.assertEqual(restarted.info()["indexVersion"], before["indexVersion"])
        self.assertEqual(len(self.store.collections), 1)
        self.assertTrue(self.store.searches)
        self.assertTrue(all(r.kb_version == before["indexVersion"] for r in results))

    async def test_failed_build_keeps_previous_lexical_and_vector_snapshot(self):
        await self.pipeline.prepare_index()
        old_version = self.pipeline.info()["indexVersion"]
        self.corpus.upload_document("new.md", "# 新文档\n\n## 道岔\n\n锁闭故障的新检查步骤。".encode())
        self.models.fail_embedding = True
        with self.assertRaises(RetrievalModelError):
            await self.pipeline.prepare_index()
        # Restore query-embedding availability; the failed document generation stays unpublished.
        self.models.fail_embedding = False
        results = await self.pipeline.search("锁闭故障")
        self.assertTrue(self.pipeline.info()["retrievalReady"])
        self.assertIsNotNone(self.pipeline.info()["indexUpdateError"])
        self.assertEqual(self.pipeline.info()["documentCount"], 1)
        self.assertTrue(all(r.kb_version == old_version and r.chunk.document != "new.md" for r in results))
        self.assertEqual(self.pipeline.index_versions()[0]["status"], "failed")

        self.models.fail_embedding = False
        await self.pipeline.rebuild_index()
        self.assertNotEqual(self.pipeline.info()["indexVersion"], old_version)
        self.assertEqual(self.pipeline.info()["documentCount"], 2)
        self.assertIsNone(self.pipeline.info()["indexUpdateError"])

    async def test_staging_storage_failure_never_activates_partial_collection(self):
        await self.pipeline.prepare_index()
        old = self.pipeline.info()["indexVersion"]
        self.corpus.upload_document("new.md", "# 新文档\n\n新锁闭部件。".encode())
        self.store.fail_build = True
        with self.assertRaises(RuntimeError):
            await self.pipeline.prepare_index()
        self.assertEqual(self.pipeline.info()["indexVersion"], old)
        self.store.fail_build = False
        batches = len(self.models.document_batches)
        await self.pipeline.rebuild_index()
        self.assertEqual(len(self.models.document_batches), batches)
        self.assertNotEqual(self.pipeline.info()["indexVersion"], old)

    async def test_rebuild_repairs_missing_storage_in_a_new_collection(self):
        await self.pipeline.prepare_index()
        before = self.pipeline.info()
        del self.store.collections[before["indexCollection"]]
        batches = len(self.models.document_batches)
        await self.pipeline.rebuild_index()
        after = self.pipeline.info()
        self.assertEqual(after["indexVersion"], before["indexVersion"])
        self.assertNotEqual(after["indexCollection"], before["indexCollection"])
        self.assertTrue(after["retrievalReady"])
        self.assertEqual(len(self.models.document_batches), batches)

    async def test_rollback_survives_queries_restart_and_can_fast_forward(self):
        await self.pipeline.prepare_index()
        old = self.pipeline.info()["indexVersion"]
        self.corpus.upload_document("new.md", "# 新文档\n\n新锁闭部件。".encode())
        await self.pipeline.prepare_index()
        head = self.pipeline.info()["indexVersion"]
        await self.pipeline.activate_index(old)
        self.assertEqual(self.pipeline.info()["indexSourceVersion"], head)
        restarted = self.make_pipeline()
        await restarted.prepare_index()
        results = await restarted.search("锁闭故障")
        self.assertEqual(restarted.info()["indexVersion"], old)
        self.assertTrue(all(r.chunk.document != "new.md" for r in results))
        await restarted.rebuild_index()
        self.assertEqual(restarted.info()["indexVersion"], head)

    async def test_query_pins_generation_while_another_generation_is_published(self):
        await self.pipeline.prepare_index()
        old = self.pipeline.info()["indexVersion"]
        entered, release = asyncio.Event(), asyncio.Event()
        original_embed = self.models.embed

        async def blocked_embed(texts, text_type):
            if text_type == "query":
                entered.set()
                await release.wait()
            return await original_embed(texts, text_type)

        self.models.embed = blocked_embed
        query = asyncio.create_task(self.pipeline.search("锁闭故障"))
        await asyncio.wait_for(entered.wait(), 5)
        self.corpus.upload_document("new.md", "# 新文档\n\n新锁闭部件。".encode())
        await self.pipeline.prepare_index()
        self.assertNotEqual(self.pipeline.info()["indexVersion"], old)
        release.set()
        results = await asyncio.wait_for(query, 5)
        self.assertTrue(all(r.kb_version == old and r.chunk.document != "new.md" for r in results))

    async def test_deletion_and_empty_corpus_publish_new_versions(self):
        await self.pipeline.prepare_index()
        old = self.pipeline.info()["indexVersion"]
        old_chunk = self.corpus.chunks[0]
        self.corpus.delete_document("seed.md")
        await self.pipeline.prepare_index()
        self.assertNotEqual(self.pipeline.info()["indexVersion"], old)
        self.assertEqual(await self.pipeline.search("锁闭故障"), [])
        self.assertEqual(self.pipeline.info()["documentCount"], 0)
        evidence = self.pipeline.publisher.get_chunk(old, old_chunk.chunk_id)
        self.assertEqual(evidence["content"], old_chunk.content)

    async def test_invalid_source_encoding_keeps_published_evidence_available(self):
        await self.pipeline.prepare_index()
        old = self.pipeline.info()["indexVersion"]
        (self.root / "docs" / "seed.md").write_bytes(b"\xff\xfe\xff")
        results = await self.pipeline.search("锁闭故障")
        self.assertTrue(all(r.kb_version == old for r in results))
        self.assertIsNotNone(self.pipeline.info()["indexUpdateError"])

    async def test_model_profile_change_prevents_rollback_to_incompatible_vectors(self):
        await self.pipeline.prepare_index()
        old = self.pipeline.info()["indexVersion"]
        changed = self.make_pipeline(knowledge_embedding_model="another-model")
        await changed.prepare_index()
        with self.assertRaisesRegex(ValueError, "配置"):
            await changed.activate_index(old)


@unittest.skipUnless(importlib.util.find_spec("pymilvus"), "Install .[milvus] for real storage tests")
class RealMilvusTest(unittest.TestCase):
    def test_cosine_order_persistence_and_empty_collection(self):
        with TemporaryDirectory() as directory:
            uri = Path(directory) / "milvus.db"
            store = MilvusVectorStore(uri)
            vectors = [[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]]
            try:
                store.build("generation", 2, [{"id": str(i) * 64, "vector": v} for i, v in enumerate(vectors)])
                hits = store.search("generation", [1.0, 0.0], 3)
                self.assertEqual([h.chunk_id for h in hits], [str(i) * 64 for i in range(3)])
                self.assertAlmostEqual(hits[1].score, 0.6, places=5)
                self.assertFalse(store.valid("generation", 3, 3))
                self.assertFalse(store.valid("generation", 2, 4))
                other = subprocess.run([
                    sys.executable, "-c",
                    "import sys; from pymilvus import MilvusClient; MilvusClient(sys.argv[1])",
                    str(uri),
                ], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(other.returncode, 0, "A second process must not own the same Lite directory")
            finally:
                store.close()
            reopened = MilvusVectorStore(uri)
            try:
                self.assertTrue(reopened.valid("generation", 2, 3))
                self.assertEqual(reopened.search("generation", [1.0, 0.0], 1)[0].chunk_id, "0" * 64)
                reopened.build("empty", 2, [])
                self.assertTrue(reopened.valid("empty", 2, 0))
            finally:
                reopened.close()
            fresh = subprocess.run([
                sys.executable, "-c",
                "import sys; from pathlib import Path; "
                "from app.services.vector_store import MilvusVectorStore; "
                "s=MilvusVectorStore(Path(sys.argv[1])); "
                "assert s.valid('generation',2,3); "
                "assert s.search('generation',[1.0,0.0],1)[0].chunk_id=='0'*64; "
                "s.close()",
                str(uri),
            ], capture_output=True, text=True, timeout=20)
            self.assertEqual(fresh.returncode, 0, fresh.stderr[-2000:])


class IndexApiTest(unittest.TestCase):
    def test_rebuild_activate_and_versioned_source_endpoints(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "seed.md"
            seed.write_text("# KB\n\n锁闭装置磨损。", encoding="utf-8")
            corpus = KnowledgeService(seed)
            corpus.reload_if_changed(force=True)
            settings = Settings(
                knowledge_retrieval_mode="vector", knowledge_vector_store="milvus_lite",
                knowledge_embedding_dimensions=2,
                knowledge_embedding_cache_path=root / "cache.db",
                knowledge_index_catalog_path=root / "catalog.db",
            )
            models = FakeModels()
            pipeline = RagPipeline(corpus, settings, models, vector_store=FakeVectorStore())
            app = FastAPI()
            app.state.knowledge_retriever = pipeline
            app.state.knowledge_service = corpus
            app.include_router(router)
            with patch("app.api.routes.knowledge.get_settings", return_value=settings):
                with TestClient(app) as client:
                    response = client.post("/api/knowledge/indexes/rebuild")
                    self.assertEqual(response.status_code, 200)
                    version = response.json()["indexVersion"]
                    self.assertEqual(client.get("/api/knowledge/indexes").json()["activeVersion"], version)
                    chunk_id = corpus.chunks[0].chunk_id
                    evidence = client.get(f"/api/knowledge/indexes/{version}/chunks/{chunk_id}")
                    self.assertEqual(evidence.json()["kb_version"], version)
                    self.assertEqual(client.post(f"/api/knowledge/indexes/{version}/activate").status_code, 200)
                    self.assertEqual(client.post("/api/knowledge/indexes/missing/activate").status_code, 400)
                    models.fail_embedding = True
                    seed.write_text("# KB\n\n新的锁闭故障。", encoding="utf-8")
                    self.assertEqual(client.post("/api/knowledge/indexes/rebuild").status_code, 503)
                    info = client.get("/api/knowledge/info").json()
                    self.assertEqual(info["indexVersion"], version)
                    self.assertTrue(info["ready"])


if __name__ == "__main__":
    unittest.main()
