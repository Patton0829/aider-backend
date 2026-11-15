import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.api.routes.knowledge import router
from app.core.config import Settings
from app.main import lifespan
from app.services.glm import GlmService
from app.services.knowledge import KnowledgeService
from app.services.knowledge_llm import KnowledgeLlmService
from test_glm import FakeClient, chunk


class KnowledgeLlmTest(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_uses_grounded_prompt_and_qwen_thinking_parameter(self):
        client = FakeClient([chunk(content="锁闭装置"), chunk(content="可能磨损。")])
        service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client=client)

        output = [
            item async for item in service.stream_answer(
                "为什么锁闭不到位？", "[资料1] SW-021 锁闭装置磨损", []
            )
        ]

        self.assertEqual(output, ["锁闭装置", "可能磨损。"])
        request = client.chat.completions.request
        self.assertEqual(request["model"], "qwen3.8-flash")
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        self.assertIn("SW-021", request["messages"][-1]["content"])
        self.assertIn("标注来源", request["messages"][-1]["content"])
        self.assertTrue(request["stream"])
        self.assertTrue(client.chat.completions.stream.closed)

    async def test_qwen_serializes_thinking_at_body_top_level(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            event = {
                "id": "qwen-test", "object": "chat.completion.chunk",
                "created": 1, "model": "qwen3.8-flash",
                "choices": [{"index": 0, "delta": {"content": "测试回答"}}],
            }
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async with AsyncOpenAI(
                api_key="unit-test-key", base_url="https://qwen.test/v1", http_client=http
            ) as client:
                service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client)
                output = [item async for item in service.stream_direct_answer("你好", [])]

        self.assertEqual(output, ["测试回答"])
        self.assertIs(requests[0]["enable_thinking"], False)
        self.assertNotIn("chat_template_kwargs", requests[0])

    async def test_glm_client_stays_glm_when_knowledge_uses_qwen(self):
        client = FakeClient([chunk(content="回答")])
        service = GlmService(Settings(knowledge_provider="qwen"), client=client)
        _ = [item async for item in service.stream_direct_answer("问题", [])]

        self.assertEqual(service.provider, "glm")
        self.assertEqual(client.chat.completions.request["model"], "glm")
        self.assertEqual(
            client.chat.completions.request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def test_qwen_requires_credentials_without_silent_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY is required"):
            KnowledgeLlmService(Settings(knowledge_provider="qwen", dashscope_api_key=""))


class KnowledgeProviderApiTest(unittest.TestCase):
    def test_qwen_model_info_and_streaming_knowledge_route(self):
        with TemporaryDirectory() as directory:
            seed = Path(directory) / "knowledge.md"
            seed.write_text(
                "# 故障诊断\n\n## 转辙机锁闭不到位\n\n锁闭装置磨损可能造成锁闭异常。",
                encoding="utf-8",
            )
            settings = Settings(knowledge_provider="qwen", knowledge_evidence_check="off")
            fake = FakeClient([chunk(content="可能原因："), chunk(content="锁闭装置磨损。")])
            app = FastAPI()
            app.state.knowledge_service = KnowledgeService(seed)
            app.state.knowledge_llm_service = KnowledgeLlmService(settings, fake)
            app.include_router(router)

            with patch("app.api.routes.knowledge.get_settings", return_value=settings):
                with TestClient(app) as client:
                    info = client.get("/api/knowledge/info").json()
                    self.assertEqual(info["model"], "qwen3.8-flash")
                    self.assertEqual(info["provider"], "qwen")
                    with client.websocket_connect("/ws/knowledge") as socket:
                        self.assertEqual(socket.receive_json()["type"], "connected")
                        self.assertEqual(socket.receive_json()["type"], "ready")
                        socket.send_json({
                            "type": "ask", "question": "转辙机锁闭不到位是什么原因？",
                            "requestId": "qwen-rag-test",
                        })
                        events = []
                        while True:
                            event = socket.receive_json()
                            events.append(event)
                            if event["type"] == "ready":
                                break

            self.assertFalse(any(e["type"] == "error" for e in events))
            self.assertTrue(any(e["type"] == "sources" and e["sources"] for e in events))
            self.assertTrue(any("千问" in e.get("content", "") for e in events))
            deltas = [e["content"] for e in events if e["type"] == "delta"]
            self.assertEqual(deltas[:2], ["可能原因：", "锁闭装置磨损。"])
            validation = next(e["data"] for e in events if e["type"] == "validation")
            self.assertTrue(validation["missingCitation"])
            self.assertTrue(all(e.get("requestId") == "qwen-rag-test" for e in events))

    def test_lifespan_uses_separate_clients_for_qwen_and_code_conversion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                knowledge_provider="qwen", dashscope_api_key="unit-test-key",
                knowledge_base_path=root / "seed.md", knowledge_base_dir=root / "kb",
                code_conversion_storage_dir=root / "conversion",
            )
            app = FastAPI(lifespan=lifespan)
            with patch("app.main.get_settings", return_value=settings):
                with TestClient(app):
                    self.assertEqual(app.state.knowledge_llm_service.provider, "qwen")
                    self.assertEqual(app.state.glm_service.provider, "glm")
                    self.assertIsNot(
                        app.state.knowledge_llm_service.client, app.state.glm_service.client
                    )


if __name__ == "__main__":
    unittest.main()
