from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeRequest(BaseModel):
    type: Literal["ask", "clear", "close"] = "ask"
    question: str = Field(default="", max_length=2000)


class KnowledgeSource(BaseModel):
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
