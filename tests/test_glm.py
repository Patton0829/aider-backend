import unittest
from types import SimpleNamespace

from app.core.config import Settings
from app.services.glm import GlmService


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


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
