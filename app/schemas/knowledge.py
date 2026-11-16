from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeRequest(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    type: Literal["ask", "clear", "close", "resume"] = "ask"
    question: str = Field(default="", max_length=2000)
    provider: Literal["qwen", "glm"] | None = None
    conversation_id: str | None = Field(default=None, min_length=36, max_length=36,
        pattern=r"^[0-9a-fA-F-]{36}$", validation_alias="conversationId", serialization_alias="conversationId")
    owner_token: str | None = Field(default=None, min_length=32, max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$", validation_alias="ownerToken", repr=False, exclude=True)
    request_id: str | None = Field(
        default=None,
        max_length=128,
        validation_alias="requestId",
        serialization_alias="requestId",
    )


class KnowledgeSource(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    document: str
    chapter: str
    section: str
    entry_id: str | None = Field(default=None, alias="entryId")
    line_start: int = Field(alias="lineStart")
    line_end: int = Field(alias="lineEnd")
    score: float
    chunk_id: str | None = Field(default=None, alias="chunkId")
    document_id: str | None = Field(default=None, alias="documentId")
    content_hash: str | None = Field(default=None, alias="contentHash")
    kb_version: str | None = Field(default=None, alias="kbVersion")


class KnowledgeEvent(BaseModel):
    type: Literal[
        "connected",
        "status",
        "route",
        "sources",
        "delta",
        "done",
        "ready",
        "cleared",
        "error",
        "closed",
        "assessment",
        "validation",
        "metrics",
        "model",
        "session",
        "history",
    ]
    content: str = ""
    sources: list[KnowledgeSource] = Field(default_factory=list)
    data: dict | None = None
    conversation_id: str | None = Field(default=None, serialization_alias="conversationId")
    request_id: str | None = Field(
        default=None,
        serialization_alias="requestId",
    )
