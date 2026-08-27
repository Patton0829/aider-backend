import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_plus_model: str = "qwen3-coder-plus"
    qwen_flash_model: str = "qwen3-coder-flash"

    glm_base_url: str = "http://115.190.191.103:8081/v1"
    glm_api_key: str = "not-needed"
    glm_model: str = "glm"
    glm_max_tokens: int = Field(default=4096, ge=512)
    glm_timeout_seconds: float = Field(default=120.0, gt=0)
    glm_max_concurrency: int = Field(default=2, ge=1)
    glm_temperature: float = Field(default=0.2, ge=0, le=2)

    code_conversion_storage_dir: Path = PROJECT_ROOT / "storage" / "code-conversion"
    code_conversion_max_upload_bytes: int = Field(
        default=512 * 1024 * 1024, ge=1024
    )
    code_conversion_max_source_bytes: int = Field(
        default=2 * 1024 * 1024, ge=1024
    )
    code_conversion_max_source_files: int = Field(default=32, ge=1, le=256)
    code_conversion_model: str = "glm"
    code_conversion_max_tokens: int = Field(default=12000, ge=1024)
    code_conversion_timeout_seconds: float = Field(default=180.0, gt=0)
    code_conversion_max_concurrency: int = Field(default=1, ge=1, le=8)
    code_conversion_temperature: float = Field(default=0.1, ge=0, le=2)

    knowledge_base_path: Path = PROJECT_ROOT / "列车故障诊断经验知识库.md"
    knowledge_base_dir: Path = PROJECT_ROOT / "knowledge-bases"
    knowledge_max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    knowledge_top_k: int = Field(default=5, ge=1, le=20)
    knowledge_history_turns: int = Field(default=6, ge=0, le=20)

    aider_source_root: Path = Path("/Users/patton/system/aider")
    aider_model: str = "openai/qwen3-coder-plus"
    aider_edit_format: str = "diff"
    aider_auto_commits: bool = False

    workspace_root: Path = Path("/Users/patton/system")
    allowed_workspaces: str = (
        "control-center-fe,control-center-be,transfer,communicate,aider-backend,code"
    )

    job_workspace_root: Path = Path("/home/mec-share-dc")
    job_log_poll_interval: float = Field(default=1.0, gt=0)
    job_log_max_initial_bytes: int = Field(default=65536, ge=0)

    agent_api_token: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: Path = PROJECT_ROOT / ".env") -> "Settings":
        values: dict[str, str] = {}
        if env_file.is_file():
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip().strip("\"'")

        for field_name in cls.model_fields:
            env_name = field_name.upper()
            if env_name in os.environ:
                values[field_name] = os.environ[env_name]
        return cls.model_validate(values)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def workspace_names(self) -> set[str]:
        return {
            item.strip()
            for item in self.allowed_workspaces.split(",")
            if item.strip()
        }

    @property
    def model_aliases(self) -> dict[str, str]:
        return {
            "plus": self.qwen_plus_model,
            "flash": self.qwen_flash_model,
            self.qwen_plus_model: self.qwen_plus_model,
            self.qwen_flash_model: self.qwen_flash_model,
        }

    def resolve_workspace(self, name: str) -> Path:
        if name not in self.workspace_names:
            raise ValueError(f"Workspace is not allowed: {name}")
        root = self.workspace_root.resolve()
        workspace = (root / name).resolve()
        if root not in workspace.parents or not workspace.is_dir():
            raise ValueError(f"Workspace does not exist: {name}")
        return workspace


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
