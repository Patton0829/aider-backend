"""Optional Redis rankings cache. Only hashes, chunk IDs and scores leave this process."""
import asyncio
import hashlib
import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from app.core.config import Settings


logger = logging.getLogger("uvicorn.error")
T = TypeVar("T")
MAX_PAYLOAD_BYTES = 131072


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class CachedRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float = Field(allow_inf_nan=False)


class CachedRanking(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    format: Literal[1] = 1
    context: str
    stage: Literal["recall", "rerank"]
    rows: list[CachedRow] = Field(max_length=500)


class QueryCache:
    def __init__(self, settings: Settings, client=None, clock=time.monotonic):
        self.settings = settings
        self.enabled = settings.knowledge_cache_enabled
        self._clock = clock
        self._retry_at = 0.0
        self._state = "unchecked" if self.enabled else "disabled"
        self._failure: str | None = None
        self._counters = Counter()
        self._owns_client = client is None
        self.client = client
        if self.enabled and client is None:
            self.client = Redis.from_url(
                settings.redis_url, decode_responses=False, protocol=2,
                socket_connect_timeout=settings.knowledge_cache_timeout_seconds,
                socket_timeout=settings.knowledge_cache_timeout_seconds,
                retry=Retry(NoBackoff(), 0), max_connections=20,
            )

    def key(self, stage: str, context: dict) -> tuple[str, str]:
        hashed = digest({"format": "rankings-cache-v1", "ttl": self.settings.knowledge_cache_ttl_seconds, "context": context})
        return f"{self.settings.knowledge_cache_namespace}:v1:{stage}:{hashed}", hashed

    async def _command(self, operation: Callable[[], Awaitable]):
        if not self.enabled or self._clock() < self._retry_at:
            if self.enabled:
                self._counters["circuitSkips"] += 1
            return False, None
        try:
            async with asyncio.timeout(self.settings.knowledge_cache_timeout_seconds):
                value = await operation()
            self._state, self._failure, self._retry_at = "ready", None, 0
            return True, value
        except Exception as error:
            self._state = "degraded"
            self._failure = type(error).__name__
            self._retry_at = self._clock() + self.settings.knowledge_cache_retry_seconds
            self._counters["backendErrors"] += 1
            # Never log exception text: Redis errors may include URLs/passwords.
            logger.warning("RAG Redis cache unavailable (%s); continuing without cache", self._failure)
            return False, None

    async def read(self, stage: Literal["recall", "rerank"], context: dict) -> list[CachedRow] | None:
        if not self.enabled:
            return None
        key, hashed = self.key(stage, context)
        ok, raw = await self._command(lambda: self.client.get(key))
        if not ok:
            return None
        if raw is None:
            self._counters[f"{stage}Misses"] += 1
            return None
        try:
            if not isinstance(raw, (bytes, str)) or len(raw) > MAX_PAYLOAD_BYTES:
                raise ValueError("Invalid cache size")
            parsed = CachedRanking.model_validate_json(raw)
            if parsed.context != hashed or parsed.stage != stage:
                raise ValueError("Wrong cache context")
            return parsed.rows
        except (ValueError, TypeError):
            self.reject(stage)
            return None

    def reject(self, stage: str):
        self._counters[f"{stage}Invalid"] += 1

    def hit(self, stage: str):
        # Count a hit only after validating rows against the pinned corpus snapshot.
        self._counters[f"{stage}Hits"] += 1

    async def write(self, stage: Literal["recall", "rerank"], context: dict, rows: list[dict]):
        if not self.enabled:
            return
        key, hashed = self.key(stage, context)
        try:
            payload = CachedRanking(context=hashed, stage=stage, rows=[CachedRow(**row) for row in rows]).model_dump_json()
        except (ValueError, TypeError):
            self.reject(stage)
            return
        if len(payload.encode()) > MAX_PAYLOAD_BYTES:
            return
        ok, _ = await self._command(lambda: self.client.set(key, payload, ex=self.settings.knowledge_cache_ttl_seconds))
        if ok:
            self._counters[f"{stage}Writes"] += 1

    async def ping(self) -> dict:
        await self._command(lambda: self.client.ping())
        return self.info()

    def info(self) -> dict:
        return {"enabled": self.enabled, "state": self._state,
                "ttlSeconds": self.settings.knowledge_cache_ttl_seconds,
                "retryAfterSeconds": round(max(0, self._retry_at - self._clock()), 2),
                "lastErrorType": self._failure, "counters": dict(self._counters)}

    async def close(self):
        if self.client is not None and self._owns_client:
            await self.client.aclose()


class SingleFlight:
    """Share in-flight work within the single Milvus Lite owner process, never retain results."""
    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}

    async def run(self, key: str, factory: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        task = self._tasks.get(key)
        joined = task is not None
        if task is None:
            task = asyncio.create_task(factory())
            self._tasks[key] = task

            def finished(done):
                if self._tasks.get(key) is done:
                    self._tasks.pop(key, None)
                if not done.cancelled():
                    done.exception()  # Consume errors even when every waiter disconnected.

            task.add_done_callback(finished)
        # A disconnected waiter must not cancel a request shared by other users.
        return await asyncio.shield(task), joined

    async def close(self):
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
