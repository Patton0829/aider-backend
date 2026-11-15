import unittest
from types import SimpleNamespace

from app.core.config import Settings
from app.services.glm import GlmService


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk

    async def close(self):
        self.closed = True


class FakeCompletions:
    def __init__(self, stream):
        self.stream = stream
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        return self.stream


class FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=FakeCompletions(FakeStream(chunks)))


def chunk(content=None, reasoning_content=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class GlmServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_streams_final_content_and_ignores_reasoning(self):
        client = FakeClient(
            [
                chunk(reasoning_content="内部推理"),
                chunk(content="锁闭装置"),
                chunk(content="可能磨损"),
            ]
        )
        service = GlmService(Settings(), client=client)

        output = [
            item
            async for item in service.stream_answer(
                "为什么锁闭不到位？",
                "[资料1] SW-021 锁闭装置磨损",
                [],
            )
        ]

        self.assertEqual(output, ["锁闭装置", "可能磨损"])
        self.assertTrue(client.chat.completions.request["stream"])
        self.assertEqual(client.chat.completions.request["max_tokens"], 4096)
        self.assertEqual(
            client.chat.completions.request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertTrue(client.chat.completions.stream.closed)

    async def test_can_enable_thinking_explicitly(self):
        client = FakeClient([chunk(content="诊断依据")])
        service = GlmService(Settings(glm_enable_thinking=True), client=client)

        _ = [item async for item in service.stream_direct_answer("问题", [])]

        self.assertTrue(
            client.chat.completions.request["extra_body"]["chat_template_kwargs"][
                "enable_thinking"
            ]
        )

    async def test_closing_generator_closes_upstream_stream_and_releases_slot(self):
        client = FakeClient([chunk(content="第一段"), chunk(content="第二段")])
        service = GlmService(Settings(glm_max_concurrency=1), client=client)
        stream = service.stream_direct_answer("问题", [])

        self.assertEqual(await anext(stream), "第一段")
        await stream.aclose()

        self.assertTrue(client.chat.completions.stream.closed)
        self.assertFalse(service._semaphore.locked())

    async def test_raises_when_model_returns_only_reasoning(self):
        service = GlmService(
            Settings(),
            client=FakeClient([chunk(reasoning_content="只有推理")]),
        )

        with self.assertRaisesRegex(RuntimeError, "no final content"):
            _ = [
                item
                async for item in service.stream_answer("问题", "资料", [])
            ]

    async def test_direct_answer_does_not_inject_knowledge_context(self):
        client = FakeClient([chunk(content="你好，我可以帮助你。")])
        service = GlmService(Settings(), client=client)

        output = [
            item
            async for item in service.stream_direct_answer("你好", [])
        ]

        self.assertEqual(output, ["你好，我可以帮助你。"])
        messages = client.chat.completions.request["messages"]
        self.assertEqual(messages[-1]["content"], "你好")
        self.assertNotIn("知识库资料", messages[-1]["content"])
