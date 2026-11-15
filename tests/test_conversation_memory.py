import asyncio
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import BadRequestError

from app.core.config import Settings
from app.services.context_budget import ContextBudget, ContextBudgetExceeded, clip_text, message_tokens, text_tokens
from app.services.conversation_memory import ConversationMemory, recall_plan, relevance
from app.services.knowledge import is_conversation_recall
from app.services.knowledge_llm import KnowledgeLlmService
from test_glm import FakeClient, chunk


def pair(question="转辙机电压偏低", answer="模型建议测量电压，不代表已核实。"):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(knowledge_provider="qwen", knowledge_context_window_tokens=4096,
                                 knowledge_max_input_tokens=2048, knowledge_qwen_max_tokens=512,
                                 knowledge_context_safety_tokens=512)

    def test_budget_accounts_for_every_message_and_preserves_question(self):
        messages = [{"role": "system", "content": "system"}]
        messages += pair("旧问题" * 200, "长回答" * 200) + pair("最近问题", "最近回答")
        messages += [{"role": "user", "content": "当前完整问题"}]
        packed = ContextBudget(self.settings, 512).pack(messages)
        self.assertLessEqual(message_tokens(packed.messages), 2048)
        self.assertEqual(packed.messages[0], messages[0])
        self.assertEqual(packed.messages[-1], messages[-1])
        self.assertEqual(packed.messages[1]["role"], "user")
        self.assertGreater(packed.metrics["contextMessagesDropped"], 0)

    def test_mandatory_evidence_is_never_silently_truncated(self):
        with self.assertRaises(ContextBudgetExceeded):
            ContextBudget(self.settings, 512).pack([
                {"role": "system", "content": "rules"}, {"role": "user", "content": "资料" * 1000}])

    def test_recalled_episode_is_retained_while_old_dialogue_is_removed(self):
        memory = {"role": "user", "content": "第一条问题的原始记录", "_context_required": True}
        messages = [{"role": "system", "content": "rules"}, *pair("旧" * 800, "答" * 800),
                    memory, {"role": "user", "content": "我第一条问了什么"}]
        packed = ContextBudget(self.settings, 512).pack(messages)
        self.assertEqual(packed.messages[1]["content"], memory["content"])
        self.assertNotIn("_context_required", packed.messages[1])

    def test_single_message_and_utf8_clipping(self):
        message = {"role": "user", "content": "问题"}
        self.assertEqual(ContextBudget(self.settings, 512).pack([message]).messages, [message])
        self.assertEqual(clip_text("测量🚆电压", 8), "测量")
        self.assertLessEqual(text_tokens(clip_text("测量🚆电压", 10)), 10)
        with self.assertRaises(ContextBudgetExceeded):
            ContextBudget(self.settings, 512).pack([])


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_evicted_turns_are_summarized_and_source_labels_are_preserved(self):
        settings = Settings(knowledge_history_turns=1)
        llm = SimpleNamespace(select_memory=AsyncMock(return_value=[0, 1]))
        memory = ConversationMemory(settings, llm)
        history = pair() + pair("那电流呢", "需实测")
        recent, notes, metrics = await memory.compact(history, [])
        self.assertEqual(recent, history[-2:])
        self.assertEqual(notes[0]["quote"], history[0]["content"])
        self.assertEqual(notes[0]["sourceHash"], hashlib.sha256(history[0]["content"].encode()).hexdigest())
        self.assertEqual(notes[1]["role"], "assistant")
        self.assertEqual(metrics["memorySummaryMethod"], "model_selection")
        context = memory.assemble(recent, notes)[-1]["content"]
        self.assertIn("不代表已核实", context)
        await memory.compact(recent, notes)
        llm.select_memory.assert_awaited_once()

    async def test_summary_timeout_and_invalid_selection_fall_back_to_original_quotes(self):
        for effect in (TimeoutError(), [999], []):
            with self.subTest(effect=type(effect).__name__):
                select = AsyncMock(side_effect=effect) if isinstance(effect, Exception) else AsyncMock(return_value=effect)
                memory = ConversationMemory(Settings(knowledge_history_turns=1), SimpleNamespace(select_memory=select))
                history = pair() + pair("继续", "答")
                _, notes, metrics = await memory.compact(history, [])
                self.assertEqual(notes[0]["quote"], history[0]["content"])
                self.assertEqual(metrics["memorySummaryMethod"], "extractive_fallback")

    async def test_long_turn_compacts_even_when_turn_count_is_small(self):
        memory = ConversationMemory(Settings(knowledge_memory_history_tokens=256), object())
        recent, notes, _ = await memory.compact(pair("电压" * 1000, "答" * 1000), [])
        self.assertEqual(recent, [])
        self.assertTrue(notes)
        self.assertLessEqual(text_tokens(json.dumps(notes, ensure_ascii=False)), 2048)

    async def test_compaction_leaves_room_and_summary_wait_is_bounded(self):
        llm = SimpleNamespace(select_memory=AsyncMock(return_value=[0]))
        memory = ConversationMemory(Settings(knowledge_history_turns=6), llm)
        recent, notes, _ = await memory.compact(pair() * 7, [])
        self.assertEqual(len(recent), 6)
        await memory.compact(recent + pair("继续", "收到"), notes)
        llm.select_memory.assert_awaited_once()
        async def slow(_):
            await asyncio.sleep(10)
        memory = ConversationMemory(Settings(knowledge_history_turns=1, knowledge_memory_summary_timeout_seconds=.01),
                                    SimpleNamespace(select_memory=slow))
        async with asyncio.timeout(.5):
            _, notes, metrics = await memory.compact(pair() * 2, [])
        self.assertTrue(notes)
        self.assertEqual(metrics["memorySummaryMethod"], "extractive_fallback")

    async def test_disabled_memory_and_zero_history_do_not_add_summaries(self):
        for settings in (Settings(knowledge_memory_enabled=False, knowledge_history_turns=1),
                         Settings(knowledge_history_turns=0)):
            llm = SimpleNamespace(select_memory=AsyncMock())
            recent, notes, _ = await ConversationMemory(settings, llm).compact(pair() * 3, [])
            self.assertEqual(notes, [])
            self.assertEqual(len(recent), settings.knowledge_history_turns * 2)
            llm.select_memory.assert_not_awaited()

    async def test_recall_payload_is_bounded_and_truncation_is_disclosed(self):
        memory = ConversationMemory(Settings(knowledge_memory_recall_tokens=1024), object())
        records = [{"sequence": i, "question": "故障" * 100, "answerExcerpt": "建议" * 500,
                    "answerMayBeTruncated": False} for i in range(10)]
        context = memory.assemble([], [], {"mode": "list", "records": records, "hasMore": True})[-1]
        payload = json.loads(context["content"].split("\n", 1)[1])["ownedConversationRecords"]
        self.assertTrue(payload["recallBudgetTruncated"])
        self.assertLess(len(payload["records"]), 10)
        self.assertTrue(context["_context_required"])

    def test_ordinal_and_search_plans(self):
        for question, expected in (("我第一条问题是什么", 1), ("第十二次问了什么", 12),
                                   ("第101个问题是什么", 101)):
            self.assertTrue(is_conversation_recall(question))
            self.assertEqual(recall_plan(question), ("ordinal", expected))
        self.assertEqual(recall_plan("回顾之前关于轴承温度的对话")[0], "search")
        self.assertEqual(recall_plan("列出之前关于轴承温度的对话")[0], "search")
        self.assertGreater(relevance("回顾之前关于轴承温度的对话", "轴承温度报警"),
                           relevance("回顾之前关于轴承温度的对话", "车门无法关闭"))


class BudgetApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_request_fails_before_calling_provider(self):
        client = FakeClient([chunk(content="答")])
        settings = Settings(knowledge_provider="qwen", knowledge_max_input_tokens=1024)
        service = KnowledgeLlmService(settings, client)
        with self.assertRaises(ContextBudgetExceeded):
            _ = [part async for part in service.stream_direct_answer("问题" * 1000, [])]
        self.assertIsNone(client.chat.completions.request)

    async def test_context_length_error_retries_once_before_streaming(self):
        client = FakeClient([chunk(content="答")])
        original = client.chat.completions.create
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        error = BadRequestError("maximum context length exceeded", response=httpx.Response(400, request=request), body=None)
        count = 0
        payloads = []
        async def create(**kwargs):
            nonlocal count
            count += 1
            payloads.append(kwargs["messages"])
            if count == 1:
                raise error
            return await original(**kwargs)
        client.chat.completions.create = create
        service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client)
        answer = [part async for part in service.stream_direct_answer("现在的问题", pair("旧" * 3000, "答" * 3000))]
        self.assertEqual(answer, ["答"])
        self.assertEqual(count, 2)
        self.assertLess(message_tokens(payloads[1]), message_tokens(payloads[0]))
        self.assertTrue(service.context_metrics()["contextRetried"])

    async def test_non_context_provider_errors_are_not_retried(self):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        create = AsyncMock(side_effect=BadRequestError("invalid model", response=httpx.Response(400, request=request), body=None))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client)
        with self.assertRaises(BadRequestError):
            _ = [part async for part in service.stream_direct_answer("问题", [])]
        create.assert_awaited_once()

    async def test_stream_failure_after_partial_output_does_not_retry(self):
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        class BrokenStream:
            closed = False
            async def __aiter__(self):
                yield chunk(content="部分回答")
                raise BadRequestError("maximum context length", response=httpx.Response(400, request=request), body=None)
            async def close(self):
                self.closed = True
        stream = BrokenStream()
        create = AsyncMock(return_value=stream)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client)
        answer = []
        with self.assertRaises(BadRequestError):
            async for part in service.stream_direct_answer("问题", []):
                answer.append(part)
        self.assertEqual(answer, ["部分回答"])
        create.assert_awaited_once()
        self.assertTrue(stream.closed)

    async def test_context_metrics_are_isolated_between_concurrent_requests(self):
        from test_glm import FakeStream
        async def create(**kwargs):
            await asyncio.sleep(0)
            return FakeStream([chunk(content="答")])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client)
        async def run(question):
            _ = [part async for part in service.stream_direct_answer(question, [])]
            await asyncio.sleep(0)
            return service.context_metrics()["estimatedInputTokens"]
        small, large = await asyncio.gather(run("短"), run("长" * 100))
        self.assertEqual(large - small, text_tokens("长" * 100) - text_tokens("短"))
