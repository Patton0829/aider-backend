"""Transactional conversation checkpoints and audit records in a dedicated MySQL schema."""
import asyncio
import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    JSON, Column, DateTime, ForeignKey, Index, Integer, MetaData, String, Table,
    Text, UniqueConstraint, delete, func, insert, select, update,
)
from sqlalchemy.dialects.mysql import DATETIME, MEDIUMTEXT, VARCHAR
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.services.conversation_memory import recall_plan, relevance


logger = logging.getLogger("uvicorn.error")
metadata = MetaData()
DB_OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "mysql_collate": "utf8mb4_unicode_ci"}
clock_type = DateTime().with_variant(DATETIME(fsp=6), "mysql")
text_type = Text().with_variant(MEDIUMTEXT(), "mysql")
schema = Table("rag_schema_version", metadata, Column("id", Integer, primary_key=True), Column("version", Integer, nullable=False), **DB_OPTIONS)
conversations = Table(
    "rag_conversations", metadata,
    Column("id", String(36), primary_key=True), Column("owner_hash", String(64), nullable=False),
    Column("title", String(200), nullable=False), Column("state", JSON, nullable=False),
    Column("revision", Integer, nullable=False), Column("next_sequence", Integer, nullable=False),
    Column("lease_token", String(36)), Column("lease_until", clock_type),
    Column("created_at", clock_type, nullable=False), Column("updated_at", clock_type, nullable=False),
    Index("ix_rag_conversations_owner_updated", "owner_hash", "updated_at", "id"), **DB_OPTIONS,
)
turns = Table(
    "rag_turns", metadata,
    Column("id", String(36), primary_key=True),
    Column("conversation_id", String(36), ForeignKey("rag_conversations.id", ondelete="CASCADE"), nullable=False),
    Column("request_id", String(128).with_variant(VARCHAR(128, collation="utf8mb4_bin"), "mysql"), nullable=False),
    Column("question_hash", String(64), nullable=False), Column("sequence", Integer, nullable=False),
    Column("question", text_type, nullable=False), Column("answer", text_type, nullable=False),
    Column("status", String(16), nullable=False), Column("provider", String(16), nullable=False),
    Column("model", String(100), nullable=False), Column("answer_mode", String(16)),
    Column("sources", JSON, nullable=False), Column("assessment", JSON, nullable=False),
    Column("validation", JSON, nullable=False), Column("metrics", JSON, nullable=False),
    Column("error_code", String(64)), Column("created_at", clock_type, nullable=False), Column("finished_at", clock_type),
    UniqueConstraint("conversation_id", "request_id", name="uq_rag_turn_request"),
    UniqueConstraint("conversation_id", "sequence", name="uq_rag_turn_sequence"), **DB_OPTIONS,
)


def empty_state():
    return {"history": [], "topic": "", "last_mode": None}


def owner_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(value):
    return value.replace(tzinfo=timezone.utc).isoformat() if value else None


class HistoryError(RuntimeError):
    status_code = 503


class HistoryUnavailable(HistoryError):
    def __init__(self):
        super().__init__("历史会话服务暂不可用，请检查数据库连接和表结构后重试。")


class HistoryNotFound(HistoryError):
    status_code = 404

    def __init__(self):
        super().__init__("会话不存在或当前凭据无权访问。")


class HistoryConflict(HistoryError):
    status_code = 409


class ConversationStore:
    def __init__(self, settings: Settings, engine=None, clock=utcnow):
        self.settings, self.clock = settings, clock
        if engine is None:
            if not settings.knowledge_database_url:
                raise ValueError("Set DB_HOST/DB_USERNAME/DB_DATABASE or KNOWLEDGE_DATABASE_URL before enabling MySQL history")
            engine = create_async_engine(
                settings.knowledge_database_url, echo=False, hide_parameters=True,
                pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=5,
                pool_timeout=settings.knowledge_history_db_timeout_seconds,
                connect_args={"connect_timeout": max(1, int(settings.knowledge_history_db_timeout_seconds)), "charset": "utf8mb4"},
            )
        self.engine = engine
        self._state, self._error = "unchecked", None

    @asynccontextmanager
    async def operation(self):
        try:
            async with asyncio.timeout(self.settings.knowledge_history_db_timeout_seconds):
                async with self.engine.begin() as connection:
                    yield connection
            self._state, self._error = "ready", None
        except HistoryUnavailable:
            self._state, self._error = "unavailable", self._error or "SchemaUnavailable"
            raise
        except HistoryError:
            raise
        except Exception as error:
            self._state, self._error = "unavailable", type(error).__name__
            logger.warning("Conversation database unavailable (%s)", type(error).__name__)
            raise HistoryUnavailable() from None

    async def initialize(self):
        # Explicit admin command only. Never DROP tables or alter an existing business schema.
        async with self.operation() as connection:
            await connection.run_sync(metadata.create_all)
            version = await connection.scalar(select(schema.c.version).where(schema.c.id == 1))
            if version is None:
                await connection.execute(insert(schema).values(id=1, version=1))
            elif version != 1:
                raise HistoryUnavailable()

    async def check(self):
        async with self.operation() as connection:
            if await connection.scalar(select(schema.c.version).where(schema.c.id == 1)) != 1:
                raise HistoryUnavailable()
            await connection.execute(select(conversations.c.id).limit(1))
            await connection.execute(select(turns.c.id).limit(1))
        return self.info()

    def info(self):
        return {"backend": "mysql", "state": self._state, "lastErrorType": self._error, "schemaVersion": 1}

    def summary(self, row):
        return {"id": row["id"], "title": row["title"], "revision": row["revision"],
                "createdAt": iso(row["created_at"]), "updatedAt": iso(row["updated_at"]),
                "busy": bool(row["lease_token"] and row["lease_until"] and row["lease_until"] > self.clock())}

    def turn_record(self, row):
        return {"id": row["id"], "requestId": row["request_id"], "sequence": row["sequence"],
                "question": row["question"], "answer": row["answer"], "status": row["status"],
                "provider": row["provider"], "model": row["model"], "answerMode": row["answer_mode"],
                "sources": row["sources"], "assessment": row["assessment"], "validation": row["validation"],
                "metrics": row["metrics"], "errorCode": row["error_code"],
                "createdAt": iso(row["created_at"]), "finishedAt": iso(row["finished_at"])}

    async def create(self, owner: str):
        now, identity = self.clock(), str(uuid.uuid4())
        row = dict(id=identity, owner_hash=owner, title="新建诊断会话", state=empty_state(), revision=0,
                   next_sequence=1, lease_token=None, lease_until=None, created_at=now, updated_at=now)
        async with self.operation() as connection:
            await connection.execute(insert(conversations).values(**row))
        return self.summary(row)

    async def list(self, owner: str, limit: int = 20, offset: int = 0):
        async with self.operation() as connection:
            rows = (await connection.execute(select(conversations).where(conversations.c.owner_hash == owner)
                .order_by(conversations.c.updated_at.desc(), conversations.c.id.desc()).limit(limit + 1).offset(offset))).mappings().all()
        return {"conversations": [self.summary(row) for row in rows[:limit]], "hasMore": len(rows) > limit,
                "nextOffset": offset + limit if len(rows) > limit else None}

    async def _locked(self, connection, owner, identity):
        row = (await connection.execute(select(conversations).where(
            conversations.c.id == identity, conversations.c.owner_hash == owner,
        ).with_for_update())).mappings().first()
        if row is None:
            raise HistoryNotFound()
        row = dict(row)
        if row["lease_token"] and row["lease_until"] <= self.clock():
            await connection.execute(update(turns).where(turns.c.id == row["lease_token"], turns.c.status == "running")
                .values(status="interrupted", error_code="lease_expired", finished_at=self.clock()))
            await connection.execute(update(conversations).where(conversations.c.id == identity).values(lease_token=None, lease_until=None))
            row.update(lease_token=None, lease_until=None)
        return row

    async def restore(self, owner, identity, limit=50, before=None):
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            query = select(turns).where(turns.c.conversation_id == identity)
            if before is not None:
                query = query.where(turns.c.sequence < before)
            rows = (await connection.execute(query.order_by(turns.c.sequence.desc()).limit(limit + 1))).mappings().all()
        selected = list(reversed(rows[:limit]))
        return {"conversation": self.summary(conversation), "state": conversation["state"],
                "turns": [self.turn_record(row) for row in selected], "hasMore": len(rows) > limit,
                "nextBefore": selected[0]["sequence"] if len(rows) > limit and selected else None}

    async def recall(self, owner, identity, question):
        """Read completed episodes from this owned conversation only, never other users.

        Ordered/ordinal reads work beyond the scan window. Subject matching is a
        deliberately bounded lexical scan, not an all-history semantic index.
        """
        mode, ordinal = recall_plan(question)
        limit = self.settings.knowledge_memory_recall_limit
        async with self.operation() as connection:
            await self._locked(connection, owner, identity)
            condition = (turns.c.conversation_id == identity) & (turns.c.status == "completed")
            total = await connection.scalar(select(func.count()).select_from(turns).where(condition))
            query = select(turns.c.sequence, turns.c.question,
                           func.substr(turns.c.answer, 1, 1200).label("answer")).where(condition)
            if mode == "ordinal":
                if not ordinal or not 1 <= ordinal <= 2147483647:
                    rows = []
                else:
                    rows = (await connection.execute(query.where(turns.c.sequence == ordinal).limit(1))).mappings().all()
            elif mode == "list":
                rows = (await connection.execute(query.order_by(turns.c.sequence).limit(limit))).mappings().all()
            else:
                scan = self.settings.knowledge_memory_scan_limit if mode == "search" else limit
                rows = (await connection.execute(query.order_by(turns.c.sequence.desc()).limit(scan))).mappings().all()
                if mode == "search":
                    ranked = [(relevance(question, row["question"] + row["answer"]), row) for row in rows]
                    rows = [row for score, row in sorted(ranked, key=lambda item: item[0], reverse=True) if score > 0][:limit]
                rows = sorted(rows, key=lambda row: row["sequence"])
        records = [{"sequence": row["sequence"], "question": row["question"],
                    "answerExcerpt": row["answer"], "answerMayBeTruncated": len(row["answer"]) == 1200} for row in rows]
        return {"mode": mode, "requestedOrdinal": ordinal, "completedCount": total,
                "records": records, "hasMore": total > len(records),
                "scannedLimit": self.settings.knowledge_memory_scan_limit if mode == "search" else None}

    async def begin(self, owner, identity, request_id, question):
        if not request_id or request_id != request_id.strip():
            raise HistoryConflict("请求编号不可为空或包含首尾空白。")
        hashed = hashlib.sha256(question.encode()).hexdigest()
        now, token = self.clock(), str(uuid.uuid4())
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            existing = (await connection.execute(select(turns).where(
                turns.c.conversation_id == identity, turns.c.request_id == request_id,
            ))).mappings().first()
            if existing is not None:
                if existing["question_hash"] != hashed:
                    raise HistoryConflict("相同请求编号不能对应不同问题。")
                if existing["status"] == "completed":
                    return {"replay": self.turn_record(existing), "state": conversation["state"], "conversation": self.summary(conversation)}
                raise HistoryConflict("该请求正在处理或未完整结束，请恢复会话后使用新请求编号重试。")
            if conversation["lease_token"]:
                raise HistoryConflict("该会话正在生成回答，请等待完成后再发送问题。")
            await connection.execute(insert(turns).values(
                id=token, conversation_id=identity, request_id=request_id, question_hash=hashed,
                sequence=conversation["next_sequence"], question=question, answer="", status="running",
                provider=self.settings.knowledge_provider, model=self.settings.knowledge_model,
                sources=[], assessment={}, validation={}, metrics={}, created_at=now,
            ))
            await connection.execute(update(conversations).where(conversations.c.id == identity).values(
                lease_token=token, lease_until=now + timedelta(seconds=self.settings.knowledge_history_lease_seconds),
                next_sequence=conversation["next_sequence"] + 1, updated_at=now,
                title=question[:200] if conversation["next_sequence"] == 1 else conversation["title"],
            ))
        return {"lease": token, "revision": conversation["revision"], "state": conversation["state"],
                "conversation": {**self.summary(conversation), "busy": True, "title": question[:200] if conversation["next_sequence"] == 1 else conversation["title"]}}

    async def renew(self, owner, identity, token):
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            if conversation["lease_token"] != token:
                raise HistoryConflict("会话写入租约已失效。")
            await connection.execute(update(conversations).where(conversations.c.id == identity)
                .values(lease_until=self.clock() + timedelta(seconds=self.settings.knowledge_history_lease_seconds)))

    async def complete(self, owner, identity, token, revision, record, state):
        now = self.clock()
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            if conversation["lease_token"] != token or conversation["revision"] != revision:
                raise HistoryConflict("会话版本已改变，本轮回答未保存，请恢复会话后重试。")
            await connection.execute(update(turns).where(turns.c.id == token, turns.c.status == "running").values(
                answer=record.get("answer", ""), answer_mode=record.get("answer_mode"),
                sources=record.get("sources", []), assessment=record.get("assessment", {}),
                validation=record.get("validation", {}), metrics=record.get("metrics", {}),
                status="completed", error_code=None, finished_at=now,
            ))
            await connection.execute(update(conversations).where(conversations.c.id == identity).values(
                state=state, revision=revision + 1, lease_token=None, lease_until=None, updated_at=now,
            ))
        return {**self.summary(conversation), "revision": revision + 1, "busy": False, "updatedAt": iso(now)}

    async def fail(self, owner, identity, token, record, interrupted=False):
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            if conversation["lease_token"] != token:
                return
            await connection.execute(update(turns).where(turns.c.id == token, turns.c.status == "running").values(
                answer=record.get("answer", ""), answer_mode=record.get("answer_mode"), sources=record.get("sources", []),
                assessment=record.get("assessment", {}), validation=record.get("validation", {}), metrics=record.get("metrics", {}),
                status="interrupted" if interrupted else "failed", error_code="interrupted" if interrupted else "request_failed",
                finished_at=self.clock(),
            ))
            await connection.execute(update(conversations).where(conversations.c.id == identity).values(lease_token=None, lease_until=None))

    async def clear(self, owner, identity):
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            if conversation["lease_token"]:
                raise HistoryConflict("请等待当前回答完成后再清空会话。")
            await connection.execute(delete(turns).where(turns.c.conversation_id == identity))
            await connection.execute(update(conversations).where(conversations.c.id == identity).values(
                state=empty_state(), revision=conversation["revision"] + 1, next_sequence=1,
                title="新建诊断会话", updated_at=self.clock(),
            ))

    async def delete(self, owner, identity):
        async with self.operation() as connection:
            conversation = await self._locked(connection, owner, identity)
            if conversation["lease_token"]:
                raise HistoryConflict("请等待当前回答完成后再删除会话。")
            await connection.execute(delete(turns).where(turns.c.conversation_id == identity))
            await connection.execute(delete(conversations).where(conversations.c.id == identity))

    async def close(self):
        await self.engine.dispose()


class TurnHeartbeat:
    def __init__(self, store, owner, identity, token):
        self.store, self.owner, self.identity, self.token = store, owner, identity, token
        self.lost = False
        self.task = asyncio.create_task(self._run())

    async def _run(self):
        try:
            while True:
                await asyncio.sleep(min(30, self.store.settings.knowledge_history_lease_seconds / 3))
                await self.store.renew(self.owner, self.identity, self.token)
        except asyncio.CancelledError:
            raise
        except HistoryError:
            self.lost = True

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
