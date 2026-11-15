import asyncio
import unittest
from contextlib import aclosing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.knowledge import router
from app.core.config import Settings
from app.services.citation_guard import CitationGuard
from app.services.knowledge import KnowledgeChunk, KnowledgeSearchResult, KnowledgeService
from app.services.knowledge_llm import EvidenceAssessment
from app.services.knowledge_workflow import KnowledgeWorkflow


def result(entry="SW-021", number=1):
    return KnowledgeSearchResult(KnowledgeChunk(
        str(number) * 64, "故障.md", "第三章 | 转辙机", "3.2 故障模式", entry,
        f"{entry} 锁闭不到位需重复动作，锁闭装置磨损，锁闭阶段电流延长。", 127, 127,
    ), .9, "a" * 64)


class FakeLlm:
    model_label = "千问"

    def __init__(self):
        self.calls = []
        self.fragments = ["可能存在机械磨损", "[资", "料1]。"]
        self.assessment = EvidenceAssessment(answerable=True, evidence_numbers=[1])
        self.assessment_error = False
        self.rewrite_error = False
        self.closed = False

    async def rewrite_question(self, question, topic, history):
        self.calls.append(("rewrite", question, topic))
        if self.rewrite_error:
            raise TimeoutError("private diagnostic text")
        return "转辙机锁闭故障重复动作后能否排除机械磨损？"

    async def assess_evidence(self, question, context, count):
        self.calls.append(("assess", question))
        if self.assessment_error:
            raise TimeoutError("private diagnostic text")
        return self.assessment

    async def stream_answer(self, question, context, history):
        self.calls.append(("generate", question, context, history))
        try:
            for fragment in self.fragments:
                yield fragment
                await asyncio.sleep(0)
        finally:
            self.closed = True

    async def stream_direct_answer(self, question, history):
        self.calls.append(("direct", question, history))
        yield "你好"


class FakeRetriever:
    def __init__(self):
        self.results = [result()]
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.results


class CitationGuardTest(unittest.TestCase):
    def test_split_reference_is_held_until_validated(self):
        guard = CitationGuard([result()], [1])
        self.assertEqual(guard.feed("磨损[资"), "磨损")
        self.assertEqual(guard.feed("料1]"), "[资料1]")
        self.assertEqual(guard.report()["citations"][0]["kb_version"], "a" * 64)

    def test_invalid_reference_never_streams(self):
        guard = CitationGuard([result()], [1])
        streamed = guard.feed("原因[第三章 / 3.2 / SW-") + guard.feed("999]。")
        self.assertNotIn("SW-999", streamed)
        self.assertIn("引用未通过校验", streamed)
        self.assertFalse(guard.report()["passed"])

    def test_valid_entry_and_short_chapter_label_are_canonicalized(self):
        guard = CitationGuard([result()], [1])
        self.assertEqual(guard.feed("[SW-021][第三章 / 3.2 / SW-021]"), "[资料1][资料1]")
        self.assertTrue(guard.report()["passed"])

    def test_wrong_chapter_and_unapproved_source_are_rejected(self):
        guard = CitationGuard([result(), result("SW-022", 2)], [1])
        self.assertNotIn("资料2", guard.feed("[资料2]"))
        self.assertNotIn("第四章", guard.feed("[第四章 / 3.2 / SW-021]"))

    def test_duplicate_entry_id_is_not_resolved_arbitrarily(self):
        guard = CitationGuard([result(), result(number=2)], [1, 2])
        self.assertIn("未通过校验", guard.feed("[SW-021]"))
        self.assertEqual(guard.feed("[资料2]"), "[资料2]")

    def test_incomplete_reference_is_marked_and_buffer_is_bounded(self):
        for text in ("[资料1", "[" + "x" * 300):
            guard = CitationGuard([result()], [1])
            output = guard.feed(text) + guard.feed("", final=True)
            self.assertIn("引用不完整", output)
            self.assertLess(len(guard.pending), 257)

    def test_regular_markdown_brackets_are_preserved(self):
        guard = CitationGuard([result()], [1])
        self.assertEqual(guard.feed("[查看](https://example.com)"), "[查看](https://example.com)")

    def test_unbracketed_unknown_entry_fails_final_identity_check(self):
        guard = CitationGuard([result()], [1])
        guard.feed("根据SW-999和[资料1]判断。")
        self.assertFalse(guard.report()["passed"])

    def test_combined_source_and_entry_must_refer_to_same_chunk(self):
        guard = CitationGuard([result(), result("SW-022", 2)], [1, 2])
        self.assertEqual(guard.feed("[资料1 / SW-021]"), "[资料1]")
        self.assertIn("未通过校验", guard.feed("[资料1 / SW-022]"))
        self.assertEqual(guard.feed("[来源：资料1 / 第三章 / 3.2 / SW-021]"), "[资料1]")
        self.assertIn("未通过校验", guard.feed("[资料2 / 第三章 / 3.2 / SW-021]"))

    def test_explicit_cross_reference_may_be_mentioned_but_not_cited_independently(self):
        from dataclasses import replace
        original = result()
        original = replace(original, chunk=replace(original.chunk, content="SW-021 需与SW-013区分。"))
        guard = CitationGuard([original], [1])
        guard.feed("需与SW-013区分[资料1]。")
        self.assertTrue(guard.report()["passed"])
        guard.feed("[SW-013]")
        self.assertFalse(guard.report()["passed"])


class WorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm, self.retriever = FakeLlm(), FakeRetriever()
        self.knowledge = KnowledgeService(Path("/nonexistent/workflow-test.md"))
        self.settings = Settings(knowledge_provider="qwen")
        self.workflow = KnowledgeWorkflow(self.settings, self.knowledge, self.llm, self.retriever)

    async def run_question(self, question="转辙机锁闭不到位如何排查？", session=None):
        return [event async for event in self.workflow.stream(question, session or {"history": [], "topic": "", "last_mode": None})]

    def data(self, events, kind):
        return next(e["data"] for e in events if e["type"] == kind)

    async def test_grounded_turn_streams_before_completion_and_records_stage_timings(self):
        events = await self.run_question()
        self.assertTrue(self.data(events, "assessment")["answerable"])
        self.assertTrue(self.data(events, "validation")["passed"])
        metrics = self.data(events, "metrics")
        self.assertEqual(metrics["rewrite"], "standalone")
        self.assertLess(metrics["firstContentMs"], metrics["totalMs"])
        self.assertEqual(metrics["indexVersions"], ["a" * 64])
        self.assertFalse(any(call[0] == "rewrite" for call in self.llm.calls))
        self.assertEqual(events[-1]["type"], "_complete")

    async def test_retrieval_cache_trace_is_forwarded_in_per_request_metrics(self):
        from app.services.rag_pipeline import RetrievalOutcome
        async def traced(query):
            results = await self.retriever.search(query)
            return RetrievalOutcome(results, {"recallCacheHit": True, "rerankCacheHit": True, "queryCacheState": "ready", "searchMs": 4.5})
        self.retriever.search_with_trace = traced
        metrics = self.data(await self.run_question(), "metrics")
        self.assertTrue(metrics["recallCacheHit"])
        self.assertTrue(metrics["rerankCacheHit"])
        self.assertEqual(metrics["searchMs"], 4.5)

    async def test_followup_rewrites_and_retains_topic_after_history_trim(self):
        events = await self.run_question()
        session = self.data(events, "_complete")
        for _ in range(8):
            events = await self.run_question("那电流会怎样？", session)
            session = self.data(events, "_complete")
        self.assertIn("锁闭不到位", self.retriever.queries[-1])
        self.assertEqual(session["topic"], "转辙机锁闭不到位如何排查？")
        self.assertEqual(len(session["history"]), 12)
        self.assertEqual(self.data(events, "metrics")["rewrite"], "model")

    async def test_rewrite_failure_falls_back_to_anchor_without_leaking_error(self):
        initial = self.data(await self.run_question(), "_complete")
        self.llm.rewrite_error = True
        events = await self.run_question("如果重复动作后恢复呢？", initial)
        self.assertIn("转辙机", self.retriever.queries[-1])
        self.assertEqual(self.data(events, "metrics")["rewrite"], "anchored_fallback")
        self.assertNotIn("private diagnostic", str(events))

    async def test_explicit_new_equipment_resets_old_topic(self):
        session = self.data(await self.run_question(), "_complete")
        events = await self.run_question("那车门系统无法关闭如何排查？", session)
        self.assertEqual(self.retriever.queries[-1], "那车门系统无法关闭如何排查？")
        self.assertFalse(self.data(events, "metrics")["followUp"])
        generation = [c for c in self.llm.calls if c[0] == "generate"][-1]
        self.assertEqual(generation[-1], [])

    async def test_insufficient_evidence_clarifies_without_answer_generation(self):
        self.llm.assessment = EvidenceAssessment(answerable=False, missing_information=["动作阶段电流曲线"])
        events = await self.run_question()
        self.assertFalse(self.data(events, "assessment")["answerable"])
        answer = "".join(e["content"] for e in events if e["type"] == "delta")
        self.assertIn("动作阶段电流曲线", answer)
        self.assertFalse(any(c[0] == "generate" for c in self.llm.calls))

    async def test_partial_evidence_generates_with_explicit_missing_scope(self):
        self.llm.assessment = EvidenceAssessment(answerable=True, coverage="partial", evidence_numbers=[1], missing_information=["厂家规定的紧固扭矩"])
        events = await self.run_question()
        self.assertEqual(self.data(events, "assessment")["reason"], "model_partial")
        answer = "".join(e["content"] for e in events if e["type"] == "delta")
        self.assertIn("资料范围", answer)
        self.assertIn("紧固扭矩", answer)

    async def test_explicit_diagnostic_question_after_direct_topic_routes_to_knowledge(self):
        session = self.data(await self.run_question("你好"), "_complete")
        events = await self.run_question("那转辙机故障怎么排查？", session)
        self.assertEqual(self.data(events, "metrics")["answerMode"], "knowledge")
        self.assertEqual(self.retriever.queries[-1], "那转辙机故障怎么排查？")

    async def test_switch_alias_does_not_reset_equipment_topic(self):
        session = self.data(await self.run_question(), "_complete")
        events = await self.run_question("那道岔的电流有什么特征？", session)
        self.assertTrue(self.data(events, "metrics")["followUp"])

    async def test_evidence_check_failure_is_distinct_from_no_knowledge(self):
        self.llm.assessment_error = True
        events = await self.run_question()
        self.assertEqual(self.data(events, "assessment")["reason"], "check_unavailable")
        self.assertFalse(any(c[0] == "generate" for c in self.llm.calls))

    async def test_nonexistent_explicit_entry_never_uses_nearby_entries(self):
        events = await self.run_question("请解释SW-999的故障原因")
        self.assertEqual(self.data(events, "assessment")["reason"], "requested_entry_missing")
        self.assertEqual(self.llm.calls, [])

    async def test_empty_results_skip_control_and_generation_calls(self):
        self.retriever.results = []
        events = await self.run_question()
        self.assertFalse(self.data(events, "assessment")["answerable"])
        self.assertEqual(self.llm.calls, [])

    async def test_direct_route_skips_retrieval_and_evidence_calls(self):
        events = await self.run_question("你好")
        self.assertEqual(self.retriever.queries, [])
        self.assertEqual(self.llm.calls, [("direct", "你好", [])])
        self.assertEqual(self.data(events, "metrics")["answerMode"], "direct")

    async def test_missing_citation_warns_in_answer_and_validation(self):
        self.llm.fragments = ["机械磨损。"]
        events = await self.run_question()
        self.assertTrue(self.data(events, "validation")["missingCitation"])
        self.assertIn("缺少可核验", "".join(e["content"] for e in events))

    async def test_only_approved_evidence_enters_generation(self):
        self.retriever.results = [result(), result("SW-022", 2)]
        self.llm.assessment = EvidenceAssessment(answerable=True, evidence_numbers=[2])
        self.llm.fragments = ["接点异常[资料2]。"]
        events = await self.run_question()
        generation = next(c for c in self.llm.calls if c[0] == "generate")
        self.assertNotIn("[资料1]", generation[2])
        self.assertTrue(self.data(events, "validation")["passed"])

    async def test_error_does_not_mutate_callers_session(self):
        session = {"history": [], "topic": "", "last_mode": None}
        self.llm.fragments = []
        with self.assertRaises(RuntimeError):
            await self.run_question(session=session)
        self.assertEqual(session["history"], [])

    async def test_cancelled_stream_closes_upstream(self):
        async with aclosing(self.workflow.stream("转辙机异常", {})) as events:
            async for event in events:
                if event["type"] == "delta":
                    break
        await asyncio.sleep(.01)
        self.assertTrue(self.llm.closed)


class WorkflowSocketTest(unittest.TestCase):
    def test_clear_and_new_connections_have_no_previous_context(self):
        with TemporaryDirectory() as directory:
            seed = Path(directory) / "kb.md"
            seed.write_text("# 知识\n\n## 转辙机\n\n锁闭不到位需重复动作。")
            llm = FakeLlm()
            app = FastAPI()
            app.state.knowledge_service = KnowledgeService(seed)
            app.state.knowledge_llm_service = llm
            app.state.knowledge_retriever = FakeRetriever()
            app.include_router(router)
            with patch("app.api.routes.knowledge.get_settings", return_value=Settings()):
                with TestClient(app) as client:
                    for _ in range(2):
                        with client.websocket_connect("/ws/knowledge") as socket:
                            socket.receive_json(); socket.receive_json()
                            socket.send_json({"question": "转辙机异常", "requestId": "first"})
                            while socket.receive_json()["type"] != "ready":
                                pass
                            socket.send_json({"type": "clear"})
                            self.assertEqual(socket.receive_json()["type"], "cleared")
                            socket.receive_json()
                            socket.send_json({"question": "那应该怎么办？", "requestId": "second"})
                            events = []
                            while True:
                                event = socket.receive_json(); events.append(event)
                                if event["type"] == "ready":
                                    break
                            self.assertEqual(next(e["content"] for e in events if e["type"] == "route"), "direct")
                            self.assertTrue(all(e["requestId"] == "second" for e in events))
