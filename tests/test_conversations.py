import asyncio
import json
import os
import unittest
import uuid
from contextlib import aclosing
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine

from app.api.routes.conversations import router as history_router
from app.api.routes.knowledge import router as knowledge_router
from app.core.config import Settings
from app.schemas.knowledge import KnowledgeRequest
from app.services.conversation_session import ConversationSession
from app.services.conversation_store import (
    ConversationStore, HistoryConflict, HistoryNotFound, HistoryUnavailable,
    conversations, empty_state, owner_hash, turns,
)
from app.services.knowledge import KnowledgeService
from app.services.knowledge_workflow import KnowledgeWorkflow
from test_knowledge_workflow import FakeLlm, FakeRetriever


class StoreTests(unittest.IsolatedAsyncioTestCase):
    real_mysql = False

    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.settings = Settings(knowledge_provider="qwen", knowledge_history_store="mysql",
            knowledge_database_url="mysql+asyncmy://user:password@localhost/aider_rag")
        url = os.environ["RAG_TEST_MYSQL_URL"] if self.real_mysql else "sqlite+aiosqlite:///" + str(Path(self.temp.name) / "history.db")
        self.engine = create_async_engine(url, hide_parameters=True)
        self.now = datetime(2026, 11, 15, 12)
        self.store = ConversationStore(self.settings, self.engine, clock=lambda: self.now)
        await self.store.initialize()
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        self.owner = owner_hash(self.token)
        self.owners = {self.owner}
        self.conversation = await self.store.create(self.owner)
        self.identity = self.conversation["id"]

    async def asyncTearDown(self):
        async with self.engine.begin() as connection:
            ids = (await connection.execute(select(conversations.c.id).where(conversations.c.owner_hash.in_(self.owners)))).scalars().all()
            await connection.execute(delete(turns).where(turns.c.conversation_id.in_(ids)))
            await connection.execute(delete(conversations).where(conversations.c.id.in_(ids)))
        await self.store.close()
        self.temp.cleanup()

    async def begin(self, request="request-1", question="转辙机锁闭异常"):
        return await self.store.begin(self.owner, self.identity, request, question)

    async def complete(self, lease, answer="机械磨损[资料1]。", state=None):
        record = {"answer": answer, "answer_mode": "knowledge", "sources": [{"chunkId": "a" * 64, "kbVersion": "b" * 64}],
                  "assessment": {"answerable": True}, "validation": {"passed": True}, "metrics": {"firstContentMs": 12.3}}
        state = state or {"history": [{"role": "user", "content": "转辙机锁闭异常"}, {"role": "assistant", "content": answer}], "topic": "转辙机锁闭异常", "last_mode": "knowledge"}
        return await self.store.complete(self.owner, self.identity, lease["lease"], lease["revision"], record, state)

    async def test_completed_turn_and_checkpoint_survive_new_store_instance(self):
        lease = await self.begin()
        await self.complete(lease, "故障诊断 🚆 [资料1]")
        restarted = ConversationStore(self.settings, self.engine, clock=lambda: self.now)
        result = await restarted.restore(self.owner, self.identity)
        self.assertEqual(result["conversation"]["revision"], 1)
        self.assertEqual(result["turns"][0]["status"], "completed")
        self.assertIn("🚆", result["turns"][0]["answer"])
        self.assertEqual(result["turns"][0]["sources"][0]["kbVersion"], "b" * 64)
        self.assertEqual(result["state"]["topic"], "转辙机锁闭异常")

    async def test_conversation_recall_restores_completed_turns_from_database(self):
        llm, retriever = FakeLlm(), FakeRetriever()
        workflow = KnowledgeWorkflow(self.settings, KnowledgeService(Path(self.temp.name) / "unused.md"), llm, retriever)
        questions = ["转辙机锁闭故障", "那电流呢？", "我刚刚问了你什么问题"]
        for number, question in enumerate(questions):
            # A new session per request forces checkpoint recovery from the DB.
            session = ConversationSession(self.store)
            request = KnowledgeRequest(question=question, requestId=f"recall-{number}",
                                       conversationId=self.identity, ownerToken=self.token)
            events = [event async for event in session.stream(workflow, request)]
            self.assertTrue(events[-1]["data"]["persisted"])
        self.assertEqual(llm.calls[-1][0], "direct")
        history = llm.calls[-1][-1]
        self.assertEqual([m["content"] for m in history if m["role"] == "user"], questions[:2])
        restored = await self.store.restore(self.owner, self.identity)
        self.assertEqual(len(restored["turns"]), 3)
        self.assertEqual(restored["state"]["topic"], questions[0])
        self.assertEqual(restored["state"]["last_mode"], "knowledge")

    async def test_owner_isolation_for_read_list_clear_delete_and_begin(self):
        other = owner_hash(uuid.uuid4().hex)
        self.assertEqual((await self.store.list(other))["conversations"], [])
        for operation in [self.store.restore(other, self.identity), self.store.clear(other, self.identity),
                          self.store.delete(other, self.identity), self.store.begin(other, self.identity, "wrong", "问题")]:
            with self.assertRaises(HistoryNotFound):
                await operation

    async def test_completed_request_replays_without_rolling_back_newer_checkpoint(self):
        await self.complete(await self.begin())
        newer = {"history": [], "topic": "车门故障", "last_mode": "knowledge"}
        await self.complete(await self.begin("request-2", "车门故障"), state=newer)
        replay = await self.begin()
        self.assertIn("replay", replay)
        self.assertEqual(replay["state"]["topic"], "车门故障")
        self.assertEqual(len((await self.store.restore(self.owner, self.identity))["turns"]), 2)

    async def test_same_request_id_cannot_change_question(self):
        await self.complete(await self.begin())
        with self.assertRaises(HistoryConflict):
            await self.begin(question="另一问题")

    async def test_request_id_whitespace_cannot_alias_mysql_padded_strings(self):
        await self.complete(await self.begin("same"))
        with self.assertRaises(HistoryConflict):await self.begin("same ")

    async def test_active_turn_prevents_concurrent_ask_clear_and_delete(self):
        await self.begin()
        for operation in [self.begin("request-2"), self.store.clear(self.owner, self.identity), self.store.delete(self.owner, self.identity)]:
            with self.assertRaises(HistoryConflict):
                await operation

    async def test_expired_turn_is_interrupted_and_cannot_commit_over_new_turn(self):
        old = await self.begin()
        self.now += timedelta(seconds=301)
        new = await self.begin("request-2")
        with self.assertRaises(HistoryConflict):
            await self.complete(old)
        await self.complete(new)
        records = (await self.store.restore(self.owner, self.identity))["turns"]
        self.assertEqual([r["status"] for r in records], ["interrupted", "completed"])

    async def test_heartbeat_renews_lease(self):
        lease = await self.begin()
        self.now += timedelta(seconds=250)
        await self.store.renew(self.owner, self.identity, lease["lease"])
        self.now += timedelta(seconds=100)
        await self.complete(lease)

    async def test_failed_partial_answer_does_not_enter_future_context(self):
        lease = await self.begin()
        await self.store.fail(self.owner, self.identity, lease["lease"], {"answer": "未完成"}, interrupted=True)
        resumed = await self.store.restore(self.owner, self.identity)
        self.assertEqual(resumed["turns"][0]["status"], "interrupted")
        self.assertEqual(resumed["turns"][0]["answer"], "未完成")
        self.assertEqual(resumed["state"], empty_state())
        self.assertFalse((await self.begin("retry"))["state"]["history"])

    async def test_save_failure_never_emits_persisted_success(self):
        workflow = KnowledgeWorkflow(self.settings, KnowledgeService(Path(self.temp.name) / "unused.md"), FakeLlm(), FakeRetriever())
        request = KnowledgeRequest.model_validate({"question": "转辙机锁闭异常", "requestId": "failed-save", "conversationId": self.identity, "ownerToken": self.token})
        events = []
        with patch.object(self.store, "complete", side_effect=HistoryUnavailable()):
            with self.assertRaises(HistoryUnavailable):
                async for event in ConversationSession(self.store).stream(workflow, request):events.append(event)
        self.assertFalse(any(e.get("data", {}).get("persisted") for e in events))
        restored = await self.store.restore(self.owner, self.identity)
        self.assertEqual(restored["state"], empty_state())
        self.assertEqual(restored["turns"][0]["status"], "failed")

    async def test_failed_turn_with_same_request_id_is_not_silently_regenerated(self):
        lease = await self.begin()
        await self.store.fail(self.owner, self.identity, lease["lease"], {}, interrupted=True)
        with self.assertRaises(HistoryConflict):await self.begin()

    async def test_clear_removes_turns_and_context_but_keeps_owned_conversation(self):
        await self.complete(await self.begin())
        await self.store.clear(self.owner, self.identity)
        restored = await self.store.restore(self.owner, self.identity)
        self.assertEqual(restored["state"], empty_state())
        self.assertEqual(restored["turns"], [])
        self.assertEqual(restored["conversation"]["revision"], 2)

    async def test_history_pages_are_ordered_and_do_not_repeat_boundary(self):
        for n in range(5):
            await self.complete(await self.begin(str(n)))
        page = await self.store.restore(self.owner, self.identity, limit=2)
        self.assertEqual([r["sequence"] for r in page["turns"]], [4, 5])
        older = await self.store.restore(self.owner, self.identity, limit=2, before=page["nextBefore"])
        self.assertEqual([r["sequence"] for r in older["turns"]], [2, 3])

    async def test_owner_token_is_not_in_public_results(self):
        payload = json.dumps(await self.store.restore(self.owner, self.identity))
        self.assertNotIn(self.token, payload)
        self.assertNotIn(self.owner, payload)

    async def test_persistent_session_replays_without_another_model_call(self):
        llm = FakeLlm()
        workflow = KnowledgeWorkflow(self.settings, KnowledgeService(Path(self.temp.name) / "unused.md"), llm, FakeRetriever())
        request = KnowledgeRequest.model_validate({"question": "转辙机锁闭异常", "requestId": "session-1", "conversationId": self.identity, "ownerToken": self.token})
        session = ConversationSession(self.store)
        first = [event async for event in session.stream(workflow, request)]
        self.assertTrue(first[-1]["data"]["persisted"])
        calls = len(llm.calls)
        second = [event async for event in ConversationSession(self.store).stream(workflow, request)]
        self.assertEqual(len(llm.calls), calls)
        self.assertTrue(next(e for e in second if e["type"] == "metrics")["data"]["replayed"])

    async def test_disconnect_immediately_after_session_event_releases_lease(self):
        workflow = KnowledgeWorkflow(self.settings, KnowledgeService(Path(self.temp.name) / "unused.md"), FakeLlm(), FakeRetriever())
        request = KnowledgeRequest.model_validate({"question": "转辙机锁闭异常", "requestId": "disconnect", "conversationId": self.identity, "ownerToken": self.token})
        async with aclosing(ConversationSession(self.store).stream(workflow, request)) as events:
            self.assertEqual((await anext(events))["type"], "session")
        history = await self.store.restore(self.owner, self.identity)
        self.assertEqual(history["turns"][0]["status"], "interrupted")
        self.assertFalse(history["conversation"]["busy"])


@unittest.skipUnless(os.getenv("RAG_TEST_MYSQL_URL"), "Set RAG_TEST_MYSQL_URL for native MySQL tests")
class MySqlStoreTests(StoreTests):
    real_mysql = True

    async def test_real_row_lock_allows_only_one_concurrent_turn(self):
        result = await asyncio.gather(self.begin("a"), self.begin("b"), return_exceptions=True)
        self.assertEqual(sum(isinstance(value, HistoryConflict) for value in result), 1)
        self.assertEqual(sum(isinstance(value, dict) for value in result), 1)

    async def test_request_id_is_case_sensitive_on_mysql(self):
        await self.complete(await self.begin("ABC"))
        await self.complete(await self.begin("abc"))
        self.assertEqual(len((await self.store.restore(self.owner, self.identity))["turns"]), 2)


class HistoryApiTests(unittest.TestCase):
    def test_rest_and_ws_resume_preserve_context_and_enforce_owner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "seed.md"; seed.write_text("## 道岔\n\n机械磨损。")
            settings = Settings(knowledge_provider="qwen", agent_api_token="test-service")
            engine = create_async_engine("sqlite+aiosqlite:///" + str(root / "history.db"))
            store = ConversationStore(settings, engine)
            async def initialize(): await store.initialize()
            asyncio.run(initialize())
            app = FastAPI()
            app.state.knowledge_service = KnowledgeService(seed)
            app.state.knowledge_llm_service = FakeLlm()
            app.state.knowledge_retriever = FakeRetriever()
            app.state.conversation_store = store
            app.include_router(history_router); app.include_router(knowledge_router)
            token = "a" * 64
            headers = {"Authorization": "Bearer test-service", "X-Conversation-Token": token}
            with patch("app.core.security.get_settings", return_value=settings), patch("app.api.routes.knowledge.get_settings", return_value=settings), TestClient(app) as client:
                self.assertEqual(client.get("/api/knowledge/conversations").status_code, 401)
                identity = client.post("/api/knowledge/conversations", headers=headers).json()["id"]
                self.assertEqual(client.get(f"/api/knowledge/conversations/{identity}", headers={**headers, "X-Conversation-Token": "b" * 64}).status_code, 404)
                for question, request_id in [("转辙机锁闭故障", "first"), ("那电流呢？", "second"), ("我刚刚问了你什么问题", "recall")]:
                    with client.websocket_connect("/ws/knowledge?token=test-service") as socket:
                        socket.receive_json(); socket.receive_json()
                        socket.send_json({"type": "resume", "conversationId": identity, "ownerToken": token, "requestId": "restore"})
                        self.assertEqual(socket.receive_json()["type"], "history"); socket.receive_json()
                        socket.send_json({"question": question, "requestId": request_id, "conversationId": identity, "ownerToken": token})
                        events=[]
                        while True:
                            event=socket.receive_json(); events.append(event)
                            if event["type"] == "ready":break
                        self.assertFalse(any(e["type"] == "error" for e in events))
                        if request_id == "first":
                            socket.send_json({"question": question, "requestId": request_id, "conversationId": identity, "ownerToken": token})
                            replay = []
                            while True:
                                event=socket.receive_json(); replay.append(event)
                                if event["type"] == "ready":break
                            self.assertFalse(any(e["type"] == "error" for e in replay))
                            self.assertTrue(next(e["data"] for e in replay if e["type"] == "metrics")["replayed"])
                            self.assertEqual(next(e["sources"] for e in replay if e["type"] == "sources")[0]["entryId"], "SW-021")
                        if request_id == "second":
                            self.assertTrue(next(e["data"] for e in events if e["type"] == "metrics")["followUp"])
                        if request_id == "recall":
                            metrics = next(e["data"] for e in events if e["type"] == "metrics")
                            self.assertEqual(metrics["answerMode"], "direct")
                            self.assertTrue(metrics["conversationRecall"])
                            history = app.state.knowledge_llm_service.calls[-1][-1]
                            self.assertEqual([m["content"] for m in history if m["role"] == "user"], ["转辙机锁闭故障", "那电流呢？"])
                            self.assertTrue(next(e["data"] for e in events if e["type"] == "done")["persisted"])
                saved = client.get(f"/api/knowledge/conversations/{identity}", headers=headers).json()
                self.assertEqual(len(saved["turns"]), 3)
                self.assertNotIn("state", saved)
                self.assertEqual(client.delete(f"/api/knowledge/conversations/{identity}", headers=headers).status_code, 200)
                self.assertEqual(client.get(f"/api/knowledge/conversations/{identity}", headers=headers).status_code, 404)
            asyncio.run(store.close())
