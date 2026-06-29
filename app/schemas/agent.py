from typing import Literal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    type: Literal["chat", "edit", "interrupt", "close"]
    content: str = ""
    model: str = "flash"
    workspace: str = "control-center-fe"
    files: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    type: Literal[
        "connected",
        "status",
        "delta",
        "output",
        "ready",
        "error",
        "closed",
    ]
    content: str = ""
    request_type: str | None = None


class LogEvent(BaseModel):
    type: Literal["connected", "snapshot", "append", "status", "error"]
    content: str = ""
    path: str | None = None

