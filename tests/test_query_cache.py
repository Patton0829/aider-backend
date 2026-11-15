import asyncio
import json
import os
import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.knowledge import router
from app.core.config import Settings
from app.services.knowledge import KnowledgeService
from app.services.query_cache import QueryCache, SingleFlight
from app.services.rag_models import RetrievalModelError
from app.services.rag_pipeline import RagPipeline
from test_published_index import FakeVectorStore
from test_rag import FakeModels


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


class FakeRedis:
    def __init__(self, clock):
        self.data = {}
        self.expires = {}
        self.calls = []
        self.clock = clock
        self.fail = False

    async def get(self, key):
        self.calls.append(("get", key))
        if self.fail:
            raise ConnectionError("secret redis://private-password@host")
        if self.expires.get(key, 0) <= self.clock():
            self.data.pop(key, None)
        return self.data.get(key)

    async def set(self, key, value, ex):
        self.calls.append(("set", key))
        if self.fail:
            raise ConnectionError("private-password")
        self.data[key] = value.encode()
        self.expires[key] = self.clock() + ex
        return True

    async def ping(self):
        if self.fail:
            raise ConnectionError("private-password")
        return True


class CachePipelineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        self.seed = root / "seed.md"
        self.seed.write_text("## 转辙机\n\n锁闭不到位可能是机械磨损。\n\n## 轴承\n\n轴承外圈点蚀。")
        self.corpus = KnowledgeService(self.seed, storage_dir=root / "docs")
        self.corpus.reload_if_changed(force=True)
        self.settings = Settings(
            knowledge_retrieval_mode="hybrid", knowledge_vector_store="milvus_lite",
            knowledge_rerank_enabled=True, knowledge_top_k=1, knowledge_embedding_dimensions=2,
            knowledge_embedding_cache_path=root / "embedding.db",
            knowledge_index_catalog_path=root / "catalog.db", knowledge_cache_enabled=True,
        )
        self.clock = Clock()
        self.redis = FakeRedis(self.clock)
        self.models = FakeModels()
        self.store = FakeVectorStore()
        self.pipelines = []
        self.pipeline = self.make_pipeline()
        await self.pipeline.prepare_index()

    def make_pipeline(self, **changes):
        settings = self.settings.model_copy(update=changes)
        cache = QueryCache(settings, self.redis, clock=self.clock)
        pipeline = RagPipeline(self.corpus, settings, self.models, self.models, self.store, cache)
        self.pipelines.append(pipeline)
        return pipeline

    async def asyncTearDown(self):
        for pipeline in self.pipelines:
            await pipeline.close()
        self.directory.cleanup()

    async def test_hot_request_skips_query_embedding_milvus_and_reranking(self):
        first = await self.pipeline.search_with_trace("锁闭不到位")
        searches = len(self.store.searches)
        second = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertEqual(first.results, second.results)
        self.assertFalse(first.metrics["recallCacheHit"])
        self.assertTrue(second.metrics["recallCacheHit"])
        self.assertTrue(second.metrics["rerankCacheHit"])
        self.assertEqual(len(self.models.queries), 1)
        self.assertEqual(len(self.models.rerank_inputs), 1)
        self.assertEqual(len(self.store.searches), searches)

    async def test_cache_survives_pipeline_restart(self):
        await self.pipeline.search("锁闭不到位")
        restarted = self.make_pipeline()
        await restarted.prepare_index()
        outcome = await restarted.search_with_trace("锁闭不到位")
        self.assertTrue(outcome.metrics["recallCacheHit"])
        self.assertTrue(outcome.metrics["rerankCacheHit"])

    async def test_disabling_cache_does_not_contact_redis(self):
        outcome = await self.make_pipeline(knowledge_cache_enabled=False).search_with_trace("锁闭不到位")
        self.assertTrue(outcome.results)
        self.assertEqual(self.redis.calls, [])
        self.assertEqual(outcome.metrics["queryCacheState"], "disabled")

    async def test_ttl_policy_change_cannot_reuse_longer_lived_entries(self):
        await self.pipeline.search("锁闭不到位")
        outcome = await self.make_pipeline(knowledge_cache_ttl_seconds=1).search_with_trace("锁闭不到位")
        self.assertFalse(outcome.metrics["recallCacheHit"])

    async def test_expiry_is_five_minutes_and_reads_do_not_extend_it(self):
        await self.pipeline.search("锁闭不到位")
        self.assertTrue(all(exp == self.clock.now + 300 for exp in self.redis.expires.values()))
        original_expiry = dict(self.redis.expires)
        self.clock.now += 299
        self.assertTrue((await self.pipeline.search_with_trace("锁闭不到位")).metrics["rerankCacheHit"])
        self.assertEqual(original_expiry, self.redis.expires)
        self.clock.now += 2
        outcome = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertFalse(outcome.metrics["recallCacheHit"])
        self.assertFalse(outcome.metrics["rerankCacheHit"])

    async def test_model_change_reuses_recall_but_not_rerank(self):
        await self.pipeline.search("锁闭不到位")
        outcome = await self.make_pipeline(knowledge_rerank_model="new-reranker").search_with_trace("锁闭不到位")
        self.assertTrue(outcome.metrics["recallCacheHit"])
        self.assertFalse(outcome.metrics["rerankCacheHit"])
        self.assertEqual(len(self.models.queries), 1)
        self.assertEqual(len(self.models.rerank_inputs), 2)

    async def test_retrieval_policy_changes_invalidate_both_layers(self):
        await self.pipeline.search("锁闭不到位")
        for key, value in [("knowledge_rrf_k", 20), ("knowledge_bm25_recall_k", 3), ("knowledge_dense_recall_k", 2), ("knowledge_retrieval_mode", "vector")]:
            outcome = await self.make_pipeline(**{key: value}).search_with_trace("锁闭不到位")
            self.assertFalse(outcome.metrics["recallCacheHit"], key)
            self.assertFalse(outcome.metrics["rerankCacheHit"], key)

    async def test_embedding_model_change_invalidates_cache(self):
        await self.pipeline.search("锁闭不到位")
        outcome = await self.make_pipeline(knowledge_embedding_model="new-embedding").search_with_trace("锁闭不到位")
        self.assertFalse(outcome.metrics["recallCacheHit"])

    async def test_credentials_and_cache_namespace_are_isolated(self):
        await self.pipeline.search("锁闭不到位")
        for changes in ({"dashscope_api_key": "different-api-account"}, {"knowledge_cache_namespace": "other-application"}):
            outcome = await self.make_pipeline(**changes).search_with_trace("锁闭不到位")
            self.assertFalse(outcome.metrics["recallCacheHit"])

    async def test_top_k_change_keeps_recall_if_candidate_limit_unchanged(self):
        await self.pipeline.search("锁闭不到位")
        outcome = await self.make_pipeline(knowledge_top_k=2).search_with_trace("锁闭不到位")
        self.assertTrue(outcome.metrics["recallCacheHit"])
        self.assertFalse(outcome.metrics["rerankCacheHit"])

    async def test_different_questions_and_numbers_never_share_keys(self):
        await self.pipeline.search("转辙机电流12A")
        for question in ("转辙机电流21A", "转辙机电流1 2A"):
            self.assertFalse((await self.pipeline.search_with_trace(question)).metrics["recallCacheHit"])

    async def test_modified_documents_use_new_version_and_rollback_can_reuse_old_cache(self):
        first = await self.pipeline.search_with_trace("锁闭不到位")
        old_version = first.results[0].kb_version
        self.corpus.upload_document("new.md", "## 转辙机\n\n锁闭电流延长。".encode())
        updated = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertFalse(updated.metrics["recallCacheHit"])
        self.assertNotEqual(updated.results[0].kb_version, old_version)
        await self.pipeline.activate_index(old_version)
        rolled_back = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertTrue(rolled_back.metrics["recallCacheHit"])
        self.assertEqual(rolled_back.results, first.results)

    async def test_deleted_documents_cannot_be_resurrected_by_hot_cache(self):
        await self.pipeline.search("锁闭不到位")
        self.corpus.delete_document("seed.md")
        self.assertEqual((await self.pipeline.search_with_trace("锁闭不到位")).results, [])

    async def test_backend_outage_falls_back_and_circuit_recovers(self):
        self.redis.fail = True
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            first = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertTrue(first.results)
        self.assertEqual(first.metrics["queryCacheState"], "degraded")
        self.assertNotIn("private-password", str(logs.output))
        calls = len(self.redis.calls)
        self.redis.fail = False
        await self.pipeline.search("锁闭不到位")
        self.assertEqual(len(self.redis.calls), calls)
        self.clock.now += 31
        await self.pipeline.search("锁闭不到位")
        self.assertTrue((await self.pipeline.search_with_trace("锁闭不到位")).metrics["recallCacheHit"])

    async def test_write_failure_also_keeps_retrieval_available(self):
        async def fail_set(*args, **kwargs):
            raise ConnectionError("private-password")
        self.redis.set = fail_set
        outcome = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertTrue(outcome.results)
        self.assertEqual(outcome.metrics["queryCacheState"], "degraded")

    async def test_empty_dense_cache_is_rejected_but_empty_keyword_result_is_cacheable(self):
        await self.pipeline.search("锁闭不到位")
        key = next(k for k in self.redis.data if ":recall:" in k)
        payload = json.loads(self.redis.data[key]); payload["rows"] = []
        self.redis.data[key] = json.dumps(payload).encode()
        outcome = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertFalse(outcome.metrics["recallCacheHit"])
        keyword = self.make_pipeline(knowledge_retrieval_mode="keyword", knowledge_rerank_enabled=False)
        self.assertEqual((await keyword.search_with_trace("xyzqwerty")).results, [])
        empty = await keyword.search_with_trace("xyzqwerty")
        self.assertEqual(empty.results, [])
        self.assertTrue(empty.metrics["recallCacheHit"])

    async def test_corrupt_schema_hash_score_and_duplicate_rows_are_misses(self):
        await self.pipeline.search("锁闭不到位")
        key = next(k for k in self.redis.data if ":recall:" in k)
        valid = self.redis.data[key]
        variants = [b"not json", b"{" + b"x" * 140000]
        for update in ({"context": "wrong"}, {"stage": "rerank"}):
            payload = json.loads(valid); payload.update(update); variants.append(json.dumps(payload).encode())
        for field, value in [("content_hash", "0" * 64), ("chunk_id", "0" * 64), ("score", float("nan"))]:
            payload = json.loads(valid); payload["rows"][0][field] = value; variants.append(json.dumps(payload).encode())
        payload = json.loads(valid); payload["rows"].append(payload["rows"][0]); variants.append(json.dumps(payload).encode())
        for raw in variants:
            self.redis.data[key] = raw
            outcome = await self.pipeline.search_with_trace("锁闭不到位")
            self.assertFalse(outcome.metrics["recallCacheHit"])
            self.assertTrue(outcome.results)

    async def test_rerank_cache_cannot_reference_non_candidate_chunk(self):
        await self.pipeline.search("锁闭不到位")
        key = next(k for k in self.redis.data if ":rerank:" in k)
        payload = json.loads(self.redis.data[key]); payload["rows"][0]["chunk_id"] = "0" * 64
        self.redis.data[key] = json.dumps(payload).encode()
        outcome = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertTrue(outcome.metrics["recallCacheHit"])
        self.assertFalse(outcome.metrics["rerankCacheHit"])

    async def test_upstream_rerank_failure_does_not_cache_failed_result(self):
        self.models.fail_rerank = True
        with self.assertRaises(RetrievalModelError):
            await self.pipeline.search("锁闭不到位")
        self.assertFalse(any(":rerank:" in key for key in self.redis.data))
        self.models.fail_rerank = False
        outcome = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertTrue(outcome.metrics["recallCacheHit"])
        self.assertFalse(outcome.metrics["rerankCacheHit"])

    async def test_no_raw_question_document_text_or_key_is_stored(self):
        self.pipeline.settings.dashscope_api_key = "private-dashscope-key"
        await self.pipeline.search("隐私设备编号A123锁闭故障")
        data = repr(self.redis.data)
        for forbidden in ("隐私设备编号", "private-dashscope-key", "seed.md", "机械磨损"):
            self.assertNotIn(forbidden, data)

    async def test_concurrent_identical_requests_share_only_in_flight_work(self):
        original = self.models.embed
        async def slow_embed(texts, text_type):
            if text_type == "query":
                await asyncio.sleep(.03)
            return await original(texts, text_type)
        self.models.embed = slow_embed
        outcomes = await asyncio.gather(*(self.pipeline.search_with_trace("锁闭不到位") for _ in range(12)))
        self.assertEqual(len(self.models.queries), 1)
        self.assertEqual(len(self.models.rerank_inputs), 1)
        self.assertEqual(sum(o.metrics["cacheCoalesced"] for o in outcomes), 11)
        self.assertIsNot(outcomes[0].results, outcomes[1].results)

    async def test_in_flight_old_generation_does_not_pollute_new_publication(self):
        original = self.models.embed
        entered, resume = asyncio.Event(), asyncio.Event()
        async def blocked(texts, text_type):
            if text_type == "query":
                entered.set(); await resume.wait()
            return await original(texts, text_type)
        self.models.embed = blocked
        old_version = self.pipeline.info()["indexVersion"]
        task = asyncio.create_task(self.pipeline.search_with_trace("锁闭不到位"))
        await entered.wait()
        self.corpus.upload_document("update.md", "## 补充\n\n转辙机锁闭异常新依据。".encode())
        await self.pipeline.prepare_index()
        resume.set()
        old = await task
        self.assertTrue(all(r.kb_version == old_version for r in old.results))
        current = await self.pipeline.search_with_trace("锁闭不到位")
        self.assertFalse(current.metrics["recallCacheHit"])
        self.assertNotEqual(current.results[0].kb_version, old_version)


class CacheSafetyTest(unittest.IsolatedAsyncioTestCase):
    async def test_slow_backend_is_bounded_and_does_not_log_credentials(self):
        class SlowRedis:
            async def get(self, key):
                await asyncio.sleep(10)
        cache = QueryCache(Settings(knowledge_cache_enabled=True, knowledge_cache_timeout_seconds=.01), SlowRedis())
        start = time.perf_counter()
        self.assertIsNone(await cache.read("recall", {"query": "hash"}))
        self.assertLess(time.perf_counter() - start, .3)
        self.assertEqual(cache.info()["state"], "degraded")

    async def test_waiter_cancellation_does_not_cancel_shared_work(self):
        flight = SingleFlight()
        started, done = asyncio.Event(), asyncio.Event()
        async def work():
            started.set(); await done.wait(); return 42
        first = asyncio.create_task(flight.run("same", work))
        await started.wait()
        second = asyncio.create_task(flight.run("same", work))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        done.set()
        self.assertEqual(await second, (42, True))
        await flight.close()

    async def test_shutdown_cancels_and_drains_shared_work(self):
        flight = SingleFlight()
        entered, cleaned = asyncio.Event(), asyncio.Event()
        async def work():
            entered.set()
            try:
                await asyncio.sleep(30)
            finally:
                cleaned.set()
        waiter = asyncio.create_task(flight.run("same", work))
        await entered.wait()
        await flight.close()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertTrue(cleaned.is_set())

    def test_invalid_url_does_not_echo_password(self):
        with self.assertRaises(ValidationError) as caught:
            Settings(redis_url="redis://:private-password@host:invalid/0")
        self.assertNotIn("private-password", str(caught.exception))


class CacheApiTest(unittest.TestCase):
    def test_probe_requires_auth_and_does_not_expose_url(self):
        with TemporaryDirectory() as directory:
            seed = Path(directory) / "seed.md"; seed.write_text("## 道岔\n\n锁闭机械磨损。")
            settings = Settings(knowledge_cache_enabled=True, agent_api_token="internal-test")
            pipeline = RagPipeline(KnowledgeService(seed), settings, query_cache=QueryCache(settings, FakeRedis(Clock())))
            app = FastAPI(); app.state.knowledge_retriever = pipeline; app.include_router(router)
            with patch("app.core.security.get_settings", return_value=settings), TestClient(app) as client:
                self.assertEqual(client.get("/api/knowledge/cache").status_code, 401)
                headers = {"Authorization": "Bearer internal-test"}
                response = client.post("/api/knowledge/cache/probe", json={"question": "锁闭磨损"}, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["passed"])
                self.assertNotIn("redis://", response.text)
                self.assertEqual(client.post("/api/knowledge/cache/probe", json={"question": " "}, headers=headers).status_code, 400)


@unittest.skipUnless(os.getenv("RAG_TEST_REDIS_URL"), "Set RAG_TEST_REDIS_URL to test a real Redis server")
class RealRedisTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_roundtrip_expiry_and_client_reconnect(self):
        settings = Settings(knowledge_cache_enabled=True, redis_url=os.environ["RAG_TEST_REDIS_URL"], knowledge_cache_ttl_seconds=1,
                            knowledge_cache_namespace="rag-test-" + uuid.uuid4().hex)
        cache = QueryCache(settings)
        row = {"chunk_id": "a" * 64, "content_hash": "b" * 64, "score": .8}
        await cache.write("recall", {"query": "hash"}, [row])
        self.assertEqual((await cache.read("recall", {"query": "hash"}))[0].chunk_id, "a" * 64)
        key, _ = cache.key("recall", {"query": "hash"})
        ttl = await cache.client.ttl(key)
        self.assertIn(ttl, (0, 1))
        await cache.close()
        other = QueryCache(settings)
        self.assertIsNotNone(await other.read("recall", {"query": "hash"}))
        await asyncio.sleep(1.1)
        self.assertIsNone(await other.read("recall", {"query": "hash"}))
        await other.close()
