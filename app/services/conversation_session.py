"""Per-socket state with explicit persistence, idempotent replay and completed checkpoints."""
import asyncio
import logging
from contextlib import aclosing

from app.schemas.knowledge import KnowledgeSource
from app.services.conversation_store import HistoryConflict, HistoryError, TurnHeartbeat, empty_state, owner_hash


class ConversationSession:
    def __init__(self, store=None):
        self.store = store
        self.state = empty_state()
        self.identity = None
        self.owner = None

    def credentials(self, request):
        if not request.conversation_id or not request.owner_token:
            raise HistoryConflict("恢复或保存历史会话需要完整的会话编号和访问凭据。")
        if self.store is None:
            raise HistoryConflict("当前服务尚未启用 MySQL 历史会话。")
        return owner_hash(request.owner_token), request.conversation_id

    async def resume(self, request):
        owner, identity = self.credentials(request)
        result = await self.store.restore(owner, identity)
        self.state = result.pop("state")
        self.owner, self.identity = owner, identity
        return result

    async def clear(self, request):
        if self.identity is not None or request.conversation_id is not None:
            owner, identity = self.credentials(request)
            await self.store.clear(owner, identity)
            self.owner, self.identity = owner, identity
        self.state = empty_state()

    async def stream(self, workflow, request):
        persistent = request.conversation_id is not None or request.owner_token is not None
        if not persistent:
            if self.identity is not None:
                raise HistoryConflict("当前是持久会话，请携带会话编号和访问凭据。")
            async with aclosing(workflow.stream(request.question.strip(), self.state)) as events:
                async for event in events:
                    if event["type"] == "_complete":
                        self.state = event["data"]
                    else:
                        yield event
            return

        owner, identity = self.credentials(request)
        if not request.request_id or not request.request_id.strip():
            raise HistoryConflict("保存问答需要非空 requestId，以避免重复生成。")
        lease = await self.store.begin(owner, identity, request.request_id, request.question.strip())
        self.owner, self.identity, self.state = owner, identity, lease["state"]
        if "replay" in lease:
            yield {"type": "session", "data": lease["conversation"]}
            record = lease["replay"]
            yield {"type": "route", "content": record["answerMode"] or "knowledge"}
            yield {"type": "sources", "sources": record["sources"]}
            for key in ("assessment", "validation", "metrics"):
                yield {"type": key, "data": {**record[key], "replayed": True}}
            yield {"type": "delta", "content": record["answer"]}
            return

        record = {"answer": "", "sources": [], "assessment": {}, "validation": {}, "metrics": {}}
        heartbeat = TurnHeartbeat(self.store, owner, identity, lease["lease"])
        committed = False
        try:
            yield {"type": "session", "data": lease["conversation"]}
            async with aclosing(workflow.stream(request.question.strip(), self.state)) as events:
                checkpoint = None
                async for event in events:
                    if heartbeat.lost:
                        raise HistoryConflict("历史会话连接已中断，本轮回答未保存，请恢复会话后重试。")
                    kind = event["type"]
                    if kind == "_complete":
                        checkpoint = event["data"]
                        continue
                    if kind == "delta":
                        record["answer"] += event["content"]
                    elif kind == "sources":
                        record["sources"] = [KnowledgeSource.model_validate(s).model_dump(by_alias=True, exclude_none=True) for s in event["sources"]]
                    elif kind == "route":
                        record["answer_mode"] = event["content"]
                    elif kind in ("assessment", "validation", "metrics"):
                        record[kind] = event["data"]
                    yield event
            await heartbeat.close()
            if checkpoint is None:
                raise HistoryConflict("本轮问答未正常结束，未保存为完整回答。")
            saved = await self.store.complete(owner, identity, lease["lease"], lease["revision"], record, checkpoint)
            committed = True
            self.state = checkpoint
            yield {"type": "session", "data": {**saved, "persisted": True}}
        except BaseException as error:
            if not committed:
                try:
                    # Preserve partial output for audit, never as context for a later answer.
                    await self.store.fail(owner, identity, lease["lease"], record,
                        interrupted=isinstance(error, (asyncio.CancelledError, GeneratorExit)) or type(error).__name__ == "WebSocketDisconnect")
                except HistoryError:
                    logging.getLogger("uvicorn.error").warning("Incomplete conversation turn could not be recorded")
            raise
        finally:
            await heartbeat.close()
