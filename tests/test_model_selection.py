"""Request-scoped generation selection, shared retrieval and persisted model identity."""
import asyncio
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine

from app.api.routes.conversations import router as history_router
from app.api.routes.knowledge import router
from app.core.config import Settings
from app.services.conversation_store import ConversationStore
from app.services.knowledge import KnowledgeService
from app.services.knowledge_workflow import KnowledgeWorkflow
from test_knowledge_workflow import FakeLlm, FakeRetriever


@contextmanager
def application(*, persistent=False, qwen=True):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        settings = Settings(knowledge_provider="qwen" if qwen else "glm")
        knowledge = KnowledgeService(root / "seed.md")
        retriever = FakeRetriever()
        llms = {name: FakeLlm() for name in (("qwen", "glm") if qwen else ("glm",))}
        llms["glm"].model_label = "GLM"
        for provider, llm in llms.items():
            llm.provider = provider
            llm.model = "qwen3.8-flash" if provider == "qwen" else "glm"
        app = FastAPI()
        app.state.knowledge_service = knowledge
        app.state.knowledge_llm_service = llms[settings.knowledge_provider]
        app.state.knowledge_workflows = {
            provider: KnowledgeWorkflow(settings.model_copy(update={"knowledge_provider": provider}),
                                        knowledge, llm, retriever, evidence_llm=llms.get("qwen"))
            for provider, llm in llms.items()
        }
        app.state.knowledge_workflow = app.state.knowledge_workflows[settings.knowledge_provider]
        store = None
        if persistent:
            engine = create_async_engine("sqlite+aiosqlite:///" + str(root / "history.db"))
            store = ConversationStore(settings, engine)
            asyncio.run(store.initialize())
        app.state.conversation_store = store
        app.include_router(router)
        app.include_router(history_router)
        try:
            with patch("app.api.routes.knowledge.get_settings", return_value=settings), \
                 patch("app.core.security.get_settings", return_value=settings), TestClient(app) as client:
                yield client, app, llms, retriever
        finally:
            if store:
                asyncio.run(store.close())


def connect(client):
    return client.websocket_connect("/ws/knowledge")


def ready(socket):
    assert socket.receive_json()["type"] == "connected"
    assert socket.receive_json()["type"] == "ready"


def ask(socket, **payload):
    socket.send_json(payload)
    events = []
    while True:
        event = socket.receive_json()
        events.append(event)
        if event["type"] == "ready":
            return events


def data(events, kind):
    return next(event["data"] for event in events if event["type"] == kind)


class ModelSelectionTests(unittest.TestCase):
    def test_switch_same_socket_keeps_context_and_does_not_change_other_users(self):
        with application() as (client, app, llms, retriever):
            info = client.get("/api/knowledge/info").json()
            self.assertEqual([m["provider"] for m in info["generationModels"]], ["qwen", "glm"])
            self.assertTrue(all(m["enabled"] for m in info["generationModels"]))
            with connect(client) as first, connect(client) as second:
                ready(first); ready(second)
                qwen = ask(first, question="转辙机锁闭故障", provider="qwen", requestId="qwen-1")
                glm = ask(first, question="那电流呢？", provider="glm", requestId="glm-1")
                other = ask(second, question="你好", requestId="other-1")
            self.assertEqual(data(qwen, "model"), {"provider": "qwen", "model": "qwen3.8-flash"})
            self.assertEqual(data(glm, "model"), {"provider": "glm", "model": "glm"})
            self.assertEqual(data(other, "model")["provider"], "qwen")
            self.assertTrue(data(glm, "metrics")["followUp"])
            self.assertEqual(data(glm, "assessment")["provider"], "qwen")
            self.assertEqual(data(glm, "assessment")["model"], "qwen3.8-flash")
            self.assertFalse(any(call[0] == "assess" for call in llms["glm"].calls))
            self.assertEqual(sum(call[0] == "assess" for call in llms["qwen"].calls), 2)
            history = next(call[-1] for call in llms["glm"].calls if call[0] == "generate")
            self.assertIn("转辙机锁闭故障", [message["content"] for message in history])
            self.assertEqual(len(retriever.queries), 2)
            self.assertEqual(app.state.knowledge_workflow.settings.knowledge_provider, "qwen")
            self.assertEqual(app.state.knowledge_workflows["glm"].settings.knowledge_provider, "glm")

    def test_qwen_review_failure_never_uses_glm_as_reviewer_or_generates_unchecked_answer(self):
        with application() as (client, _, llms, _):
            llms["qwen"].assessment_error = True
            with connect(client) as socket:
                ready(socket)
                events = ask(socket, question="转辙机锁闭故障", provider="glm", requestId="audit-failure")
            self.assertEqual(data(events, "assessment")["reason"], "check_unavailable")
            self.assertEqual(data(events, "assessment")["provider"], "qwen")
            self.assertEqual(llms["glm"].calls, [])
            self.assertIn("证据检查暂未完成", "".join(e.get("content", "") for e in events if e["type"] == "delta"))

    def test_unconfigured_or_invalid_provider_fails_without_fallback_then_socket_recovers(self):
        with application(qwen=False) as (client, _, llms, _):
            models = client.get("/api/knowledge/info").json()["generationModels"]
            self.assertFalse(models[0]["enabled"])
            with connect(client) as socket:
                ready(socket)
                unavailable = ask(socket, question="你好", provider="qwen", requestId="missing")
                invalid = ask(socket, question="你好", provider="untrusted", requestId="invalid")
                self.assertEqual(llms["glm"].calls, [])
                valid = ask(socket, question="你好", provider="glm", requestId="valid")
            self.assertTrue(any(e["type"] == "error" for e in unavailable))
            self.assertTrue(any(e["type"] == "error" for e in invalid))
            self.assertFalse(any(e["type"] == "error" for e in valid))

    def test_mixed_models_are_saved_and_request_replay_cannot_change_model(self):
        with application(persistent=True) as (client, _, llms, _):
            token = "a" * 64
            headers = {"X-Conversation-Token": token}
            identity = client.post("/api/knowledge/conversations", headers=headers).json()["id"]
            credentials = {"conversationId": identity, "ownerToken": token}
            with connect(client) as socket:
                ready(socket)
                first = ask(socket, question="转辙机锁闭故障", provider="qwen", requestId="one", **credentials)
                second = ask(socket, question="那电流呢？", provider="glm", requestId="two", **credentials)
                counts = {provider: len(llm.calls) for provider, llm in llms.items()}
                mismatch = ask(socket, question="那电流呢？", provider="qwen", requestId="two", **credentials)
                replay = ask(socket, question="那电流呢？", provider="glm", requestId="two", **credentials)
            self.assertTrue(data(first, "done")["persisted"])
            self.assertTrue(data(second, "metrics")["followUp"])
            self.assertTrue(any(e["type"] == "error" for e in mismatch))
            self.assertTrue(data(replay, "metrics")["replayed"])
            self.assertEqual(data(replay, "model")["provider"], "glm")
            self.assertEqual(counts, {provider: len(llm.calls) for provider, llm in llms.items()})
            saved = client.get(f"/api/knowledge/conversations/{identity}", headers=headers).json()["turns"]
            self.assertEqual([(t["provider"], t["model"]) for t in saved],
                             [("qwen", "qwen3.8-flash"), ("glm", "glm")])
