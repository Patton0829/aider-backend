from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


ConversionStatus = Literal[
    "queued",
    "extracting",
    "detecting",
    "converting",
    "validating",
    "packaging",
    "completed",
    "failed",
]

ConversionLogLevel = Literal["info", "success", "error"]


class ConversionLogEntry(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    sequence: int = Field(ge=1)
    timestamp: str
    phase: str
    level: ConversionLogLevel = "info"
    message: str


class ConversionJob(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    job_id: str
    status: ConversionStatus = "queued"
    phase: str = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    message: str = "等待语言检测"
    conversion_required: bool | None = None
    source_language: str | None = None
    target_language: str = "python"
    source_files: list[str] = Field(default_factory=list)
    generated_files: list[str] = Field(default_factory=list)
    logs: list[ConversionLogEntry] = Field(default_factory=list)
    artifact_available: bool = False
    error: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def public_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)


class ConversionOutcome(BaseModel):
    conversion_required: bool
    source_language: str
    source_files: list[str]
    generated_files: list[str] = Field(default_factory=list)
    artifact_available: bool = False


class ConversionFile(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    name: str
    path: str
    size: int = Field(ge=0)
    type: Literal["code", "manifest", "metric"]
    language: str

    def public_dict(self) -> dict:
        return self.model_dump(by_alias=True)


class ConversionFileContent(ConversionFile):
    content: str
