from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeRequest(BaseModel):
    type: Literal["ask", "clear", "close"] = "ask"
    question: str = Field(default="", max_length=2000)
    request_id: str | None = Field(
        default=None,
        max_length=128,
        validation_alias="requestId",
        serialization_alias="requestId",
    )


class KnowledgeSource(BaseModel):
    document: str
    chapter: str
    section: str
    entry_id: str | None = Field(default=None, serialization_alias="entryId")
    line_start: int = Field(serialization_alias="lineStart")
    line_end: int = Field(serialization_alias="lineEnd")
    score: float


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
    ]
    content: str = ""
    sources: list[KnowledgeSource] = Field(default_factory=list)
    request_id: str | None = Field(
        default=None,
        serialization_alias="requestId",
    )
