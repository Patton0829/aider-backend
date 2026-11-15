import asyncio
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from app.core.config import Settings
from app.services.knowledge import KnowledgeSearchResult, KnowledgeService
from app.services.rag_models import DashScopeRetrievalModels, RerankMatch, RetrievalModelError
from app.services.rag_pipeline import RagPipeline, reciprocal_rank_fusion


class FakeModels:
    def __init__(self):
        self.document_batches = []
        self.queries = []
        self.rerank_inputs = []
        self.fail_embedding = False
        self.fail_rerank = False

    async def embed(self, texts, text_type):
        if self.fail_embedding:
            raise RetrievalModelError("test embedding failure")
        if text_type == "document":
            self.document_batches.append(list(texts))
            await asyncio.sleep(0)
            return [[1.0, 0.0] if "锁闭" in text else [0.0, 1.0] for text in texts]
        self.queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    async def rerank(self, query, documents, top_n):
        if self.fail_rerank:
            raise RetrievalModelError("test rerank failure")
        self.rerank_inputs.append((query, list(documents), top_n))
        # Deliberately change the recall order to verify source/index mapping.
        return [RerankMatch(len(documents) - 1, 0.91)][:top_n]


class RagPipelineTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.seed = self.root / "seed.md"
        self.seed.write_text(
            "# 知识库\n\n## 转辙机\n\n锁闭不到位可能是锁闭装置磨损。\n\n"
            "## 轴承\n\n轴承外圈点蚀可能造成周期冲击。", encoding="utf-8",
        )
        self.corpus = KnowledgeService(self.seed, storage_dir=self.root / "docs")
        self.corpus.reload_if_changed(force=True)
        self.settings = Settings(
            knowledge_retrieval_mode="hybrid", knowledge_embedding_dimensions=2,
            knowledge_embedding_cache_path=self.root / "vectors.sqlite3",
            knowledge_candidate_k=2, knowledge_top_k=1, knowledge_embedding_batch_size=1,
        )
        self.models = FakeModels()

    def tearDown(self):
        self.directory.cleanup()

    def pipeline(self, **changes):
        settings = self.settings.model_copy(update=changes)
        return RagPipeline(self.corpus, settings, self.models, self.models)

    async def test_vector_retrieval_finds_semantic_match_without_keyword_overlap(self):
        pipeline = self.pipeline(knowledge_retrieval_mode="vector")
        result = await pipeline.search("机械不能到达目标位置")

        self.assertIn("锁闭", result[0].chunk.content)
        self.assertAlmostEqual(result[0].score, 1.0)
        self.assertEqual(len(self.models.document_batches), 2)

    def test_rrf_uses_rank_and_deduplicates_across_recall_routes(self):
        first, second = self.corpus.snapshot()
        lexical = [KnowledgeSearchResult(first, 9000), KnowledgeSearchResult(second, 0.1)]
        dense = [KnowledgeSearchResult(second, 0.9)]
        results = reciprocal_rank_fusion([lexical, dense], 2, k=10)
        self.assertEqual(results[0].chunk, second)
        self.assertAlmostEqual(results[0].score, 1 / 12 + 1 / 11)
        self.assertEqual(len({r.chunk.chunk_id for r in results}), 2)

    async def test_each_recall_route_has_independent_candidate_count(self):
        pipeline = self.pipeline(
            knowledge_bm25_recall_k=1, knowledge_dense_recall_k=1,
            knowledge_rerank_enabled=True,
        )
        await pipeline.search("轴承周期冲击")
        candidates = self.models.rerank_inputs[0][1]
        # BM25 selects the bearing; the fake semantic model selects the switch machine.
        self.assertEqual(len(candidates), 2)

    async def test_rerank_controls_final_order_and_preserves_source(self):
        pipeline = self.pipeline(knowledge_rerank_enabled=True)
        result = await pipeline.search("锁闭不到位")

        candidates = self.models.rerank_inputs[0][1]
        self.assertIn(result[0].chunk.content, candidates[-1])
        self.assertEqual(result[0].score, 0.91)
        self.assertEqual(result[0].chunk.document, "seed.md")
        self.assertGreater(result[0].chunk.line_start, 0)

    async def test_restart_reuses_cache_but_model_or_dimensions_change_rebuilds(self):
        await self.pipeline().prepare_index()
        original_calls = len(self.models.document_batches)
        await self.pipeline().prepare_index()
        self.assertEqual(len(self.models.document_batches), original_calls)

        await self.pipeline(knowledge_embedding_model="different-model").prepare_index()
        self.assertEqual(len(self.models.document_batches), original_calls * 2)

    async def test_document_upload_only_embeds_new_chunks_and_delete_removes_results(self):
        pipeline = self.pipeline(knowledge_retrieval_mode="vector")
        await pipeline.prepare_index()
        original_calls = len(self.models.document_batches)
        self.corpus.upload_document("new.md", "# 补充\n\n## 道岔\n\n锁闭故障补充检查步骤。".encode())
        await pipeline.prepare_index()
        self.assertEqual(len(self.models.document_batches), original_calls + 1)
        self.assertEqual(pipeline.info()["indexedChunkCount"], 3)

        self.corpus.delete_document("new.md")
        await pipeline.prepare_index()
        self.assertEqual(pipeline.info()["indexedChunkCount"], 2)
        self.assertEqual(len(self.models.document_batches), original_calls + 1)
        self.assertTrue(all(r.chunk.document != "new.md" for r in await pipeline.search("锁闭")))

    async def test_in_place_document_edit_replaces_embedding_even_with_same_filename(self):
        pipeline = self.pipeline(knowledge_retrieval_mode="vector")
        await pipeline.prepare_index()
        path = self.root / "docs" / "seed.md"
        path.write_text("# 知识库\n\n## 新内容\n\n锁闭装置需要检查新型锁钩。", encoding="utf-8")
        results = await pipeline.search("锁闭")

        self.assertIn("新型锁钩", results[0].chunk.content)
        self.assertEqual(pipeline.info()["indexedChunkCount"], 1)
        self.assertEqual(len(self.models.document_batches), 3)

    async def test_concurrent_queries_share_document_index_build(self):
        pipeline = self.pipeline()
        await asyncio.gather(pipeline.search("锁闭"), pipeline.search("道岔故障"))
        self.assertEqual(len(self.models.document_batches), 2)
        self.assertEqual(len(self.models.queries), 2)

    async def test_embedding_failure_is_visible_and_retry_rebuilds(self):
        pipeline = self.pipeline()
        self.models.fail_embedding = True
        with self.assertRaises(RetrievalModelError):
            await pipeline.search("锁闭故障")
        self.assertFalse(pipeline.info()["retrievalReady"])
        self.assertIsNotNone(pipeline.info()["retrievalError"])

        self.models.fail_embedding = False
        await pipeline.search("锁闭故障")
        self.assertTrue(pipeline.info()["retrievalReady"])
        self.assertIsNone(pipeline.info()["retrievalError"])

    async def test_rerank_failure_does_not_silently_return_unreranked_results(self):
        pipeline = self.pipeline(knowledge_rerank_enabled=True)
        self.models.fail_rerank = True
        with self.assertRaisesRegex(RetrievalModelError, "rerank failure"):
            await pipeline.search("锁闭故障")

    async def test_keyword_mode_needs_no_api_or_cache(self):
        settings = self.settings.model_copy(update={
            "knowledge_retrieval_mode": "keyword", "knowledge_rerank_enabled": False,
        })
        pipeline = RagPipeline(self.corpus, settings)
        self.assertIn("锁闭", (await pipeline.search("锁闭故障"))[0].chunk.content)
        self.assertFalse(settings.knowledge_embedding_cache_path.exists())


class RetrievalAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_embedding_payload_and_out_of_order_response(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            self.assertTrue(str(request.url).endswith("/services/embeddings/text-embedding/text-embedding"))
            return httpx.Response(200, json={"output": {"embeddings": [
                {"text_index": 1, "embedding": [0.0, 1.0]},
                {"text_index": 0, "embedding": [1.0, 0.0]},
            ]}})

        settings = Settings(dashscope_api_key="unit-test-key", knowledge_embedding_dimensions=2)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = DashScopeRetrievalModels(settings, client)
            vectors = await adapter.embed(["问题一", "问题二"], "query")
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(requests[0]["parameters"], {"dimension": 2, "text_type": "query"})

    async def test_rejects_invalid_embedding_shape(self):
        for embedding in [[1.0], [math.nan, 0.0]]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
                200, json={"output": {"embeddings": [{"text_index": 0, "embedding": embedding}]}}
            ))) as client:
                adapter = DashScopeRetrievalModels(Settings(
                    dashscope_api_key="unit-test-key", knowledge_embedding_dimensions=2,
                ), client)
                with self.assertRaises(RetrievalModelError):
                    await adapter.embed(["查询"], "query")

    async def test_rejects_invalid_rerank_indices(self):
        for rows in [[], [{"index": 4, "relevance_score": 0.8}], [
            {"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}
        ]]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
                200, json={"output": {"results": rows}}
            ))) as client:
                adapter = DashScopeRetrievalModels(Settings(dashscope_api_key="unit-test-key"), client)
                with self.assertRaises(RetrievalModelError):
                    await adapter.rerank("query", ["one", "two"], 2)

    async def test_api_errors_do_not_expose_response_or_credentials(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            403, json={"message": "unit-test-key private source text"}
        ))) as client:
            adapter = DashScopeRetrievalModels(Settings(dashscope_api_key="unit-test-key"), client)
            with self.assertRaises(RetrievalModelError) as caught:
                await adapter.rerank("query", ["document"], 1)
        self.assertIn("403", str(caught.exception))
        self.assertNotIn("unit-test-key", str(caught.exception))
        self.assertNotIn("private source text", str(caught.exception))

    def test_workspace_specific_native_url_is_derived(self):
        settings = Settings(dashscope_base_url="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/")
        self.assertEqual(settings.retrieval_api_base_url, "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1")


if __name__ == "__main__":
    unittest.main()
