import asyncio
import json
import unittest

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.services.knowledge_llm import KnowledgeLlmService


class ControlRequestsTest(unittest.IsolatedAsyncioTestCase):
    async def call_service(self, payload, invoke):
        bodies = []
        def respond(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "control", "object": "chat.completion", "created": 1,
                "model": "qwen3.8-flash", "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(payload)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async with AsyncOpenAI(api_key="unit-test", base_url="https://qwen.test/v1", http_client=http) as client:
                service = KnowledgeLlmService(Settings(knowledge_provider="qwen"), client)
                result = await invoke(service)
        return result, bodies

    async def test_evidence_uses_json_and_non_thinking_request(self):
        result, bodies = await self.call_service(
            {"answerable": True, "evidence_numbers": [1], "missing_information": []},
            lambda service: service.assess_evidence("转辙机异常", "[资料1] 故障", 1),
        )
        self.assertTrue(result.answerable)
        self.assertFalse(bodies[0]["stream"])
        self.assertFalse(bodies[0]["enable_thinking"])
        self.assertEqual(bodies[0]["response_format"], {"type": "json_object"})

    async def test_rejects_invalid_boolean_and_unknown_evidence_number(self):
        for payload, error in [
            ({"answerable": "false", "evidence_numbers": []}, ValidationError),
            ({"answerable": True, "evidence_numbers": [2]}, ValueError),
            ({"answerable": True, "evidence_numbers": []}, ValueError),
            ({"answerable": True, "evidence_numbers": [True]}, ValidationError),
        ]:
            with self.assertRaises(error):
                await self.call_service(payload, lambda service: service.assess_evidence("问题", "资料", 1))

    async def test_rewrite_cannot_invent_entry_ids(self):
        with self.assertRaises(ValueError):
            await self.call_service({"question": "转辙机SW-999电流如何？"},
                lambda service: service.rewrite_question("那电流呢？", "转辙机SW-021故障", []))

    async def test_rewrite_accepts_case_insensitive_existing_entry(self):
        result, _ = await self.call_service({"question": "转辙机SW-021电流如何？"},
            lambda service: service.rewrite_question("那电流呢？", "转辙机sw-021故障", []))
        self.assertIn("SW-021", result)

    async def test_control_timeout_also_bounds_semaphore_wait(self):
        service = KnowledgeLlmService(Settings(knowledge_provider="qwen", knowledge_control_timeout_seconds=.01), client=object())
        service._semaphore = asyncio.Semaphore(0)
        with self.assertRaises(TimeoutError):
            await service.assess_evidence("问题", "资料", 1)
