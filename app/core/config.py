import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_plus_model: str = "qwen3-coder-plus"
    qwen_flash_model: str = "qwen3-coder-flash"

    knowledge_provider: Literal["glm", "qwen"] = "glm"
    knowledge_qwen_model: str = "qwen3.8-flash"
    knowledge_qwen_enable_thinking: bool = False
    knowledge_qwen_max_tokens: int = Field(default=2048, ge=128)
    knowledge_qwen_timeout_seconds: float = Field(default=60.0, gt=0)
    knowledge_qwen_max_concurrency: int = Field(default=2, ge=1)
    knowledge_qwen_temperature: float = Field(default=0.2, ge=0, le=2)

    glm_base_url: str = "http://115.190.191.103:8081/v1"
    glm_api_key: str = "not-needed"
    glm_model: str = "glm"
    glm_enable_thinking: bool = False
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
    # Application budgets, not claims about any provider's maximum window.
    knowledge_context_window_tokens: int = Field(default=65536, ge=2048)
    knowledge_max_input_tokens: int = Field(default=49152, ge=1024)
    knowledge_context_safety_tokens: int = Field(default=2048, ge=128)
    knowledge_evidence_budget_tokens: int = Field(default=24000, ge=256)
    knowledge_memory_enabled: bool = True
    knowledge_memory_history_tokens: int = Field(default=12000, ge=256)
    knowledge_memory_summary_tokens: int = Field(default=2048, ge=128)
    knowledge_memory_summary_timeout_seconds: float = Field(default=4.0, gt=0, le=30)
    knowledge_memory_recall_tokens: int = Field(default=8000, ge=1024)
    knowledge_memory_recall_limit: int = Field(default=6, ge=1, le=20)
    knowledge_memory_scan_limit: int = Field(default=200, ge=20, le=1000)
    knowledge_rewrite_followups: bool = True
    knowledge_evidence_check: Literal["model", "off"] = "model"
    knowledge_control_timeout_seconds: float = Field(default=12.0, gt=0)
    knowledge_retrieval_mode: Literal["keyword", "vector", "hybrid"] = "keyword"
    knowledge_candidate_k: int = Field(default=20, ge=1, le=500)
    knowledge_bm25_recall_k: int = Field(default=20, ge=1, le=500)
    knowledge_dense_recall_k: int = Field(default=20, ge=1, le=500)
    knowledge_rrf_k: int = Field(default=60, ge=1)
    knowledge_embedding_model: str = "qwen3.7-text-embedding-flash"
    knowledge_embedding_dimensions: int = Field(default=1024, ge=1)
    knowledge_embedding_batch_size: int = Field(default=20, ge=1, le=20)
    knowledge_embedding_cache_path: Path = PROJECT_ROOT / "storage" / "rag" / "embeddings.sqlite3"
    knowledge_vector_store: Literal["memory", "milvus_lite"] = "memory"
    knowledge_milvus_uri: Path = PROJECT_ROOT / "storage" / "rag" / "milvus.db"
    knowledge_index_catalog_path: Path = PROJECT_ROOT / "storage" / "rag" / "index_catalog.sqlite3"
    knowledge_rerank_enabled: bool = False
    knowledge_rerank_model: str = "qwen3.7-text-rerank"
    knowledge_retrieval_timeout_seconds: float = Field(default=30.0, gt=0)
    dashscope_http_base_url: str = ""
    knowledge_cache_enabled: bool = False
    knowledge_cache_ttl_seconds: int = Field(default=300, ge=1, le=86400)
    knowledge_cache_namespace: str = Field(default="control-center-rag", pattern=r"^[a-zA-Z0-9:_-]{1,64}$")
    knowledge_cache_timeout_seconds: float = Field(default=0.2, gt=0, le=5)
    knowledge_cache_retry_seconds: float = Field(default=30, gt=0, le=300)
    redis_url: str = Field(default="redis://127.0.0.1:6380/0", repr=False)
    knowledge_history_store: Literal["off", "mysql"] = "off"
    knowledge_database_url: str = Field(default="", repr=False)
    db_host: str = ""
    db_port: int = Field(default=3306, ge=1, le=65535)
    db_database: str = "aider_rag"
    db_username: str = ""
    db_password: str = Field(default="", repr=False)
    knowledge_history_db_timeout_seconds: float = Field(default=5, gt=0, le=30)
    knowledge_history_lease_seconds: int = Field(default=300, ge=30, le=3600)

    @model_validator(mode="after")
    def build_database_url(self):
        # Keep existing deployments compatible: an explicit URL takes precedence.
        if self.knowledge_database_url or not self.db_host.strip():
            return self
        if not self.db_username.strip() or not self.db_database.strip():
            raise ValueError("DB_USERNAME and DB_DATABASE are required when DB_HOST is set")
        from sqlalchemy.engine import URL
        self.knowledge_database_url = URL.create(
            "mysql+asyncmy", host=self.db_host.strip(), port=self.db_port,
            database=self.db_database.strip(), username=self.db_username.strip(),
            password=self.db_password,
        ).render_as_string(hide_password=False)
        return self

    @model_validator(mode="after")
    def validate_context_budget(self):
        output = self.knowledge_qwen_max_tokens if self.knowledge_provider == "qwen" else self.glm_max_tokens
        if self.knowledge_context_window_tokens - output - self.knowledge_context_safety_tokens < 512:
            raise ValueError("Context window must reserve space for output, safety margin and at least 512 input tokens")
        return self

    @field_validator("knowledge_database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value:
            return value
        from sqlalchemy.engine import make_url
        try:
            url = make_url(value)
            if url.drivername != "mysql+asyncmy" or not url.host or not url.database:
                raise ValueError
            if url.port is not None and not 1 <= url.port <= 65535:
                raise ValueError
        except Exception:
            raise ValueError("KNOWLEDGE_DATABASE_URL must be mysql+asyncmy:// with a host and database") from None
        return value

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        from urllib.parse import urlsplit
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in ("redis", "rediss") or not parsed.hostname or parsed.netloc.endswith(":"):
                raise ValueError
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError
            if parsed.path not in ("", "/") and not parsed.path.removeprefix("/").isdigit():
                raise ValueError
        except ValueError:
            raise ValueError("REDIS_URL must be a valid redis:// or rediss:// address") from None
        return value

    aider_source_root: Path = PROJECT_ROOT.parent / "aider"
    aider_model: str = "openai/qwen3-coder-plus"
    aider_edit_format: str = "diff"
    aider_auto_commits: bool = False

    workspace_root: Path = PROJECT_ROOT.parent
    allowed_workspaces: str = (
        "control-center-fe,control-center-be,transfer,communicate,aider-backend,code"
    )

    job_workspace_root: Path = Path("/home/mec-share-dc")
    job_log_poll_interval: float = Field(default=1.0, gt=0)
    job_log_max_initial_bytes: int = Field(default=65536, ge=0)

    agent_api_token: str = ""
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8088,http://127.0.0.1:8088"
    )
    log_level: str = "INFO"

    @field_validator(
        "knowledge_base_path", "knowledge_base_dir", "knowledge_embedding_cache_path",
        "knowledge_milvus_uri", "knowledge_index_catalog_path", "code_conversion_storage_dir",
        "aider_source_root", "workspace_root", "job_workspace_root",
        mode="after",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @classmethod
    def from_env(cls, env_file: Path = PROJECT_ROOT / ".env") -> "Settings":
        values: dict[str, str] = {}
        if env_file.is_file():
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip()
                if len(value) >= 2 and value[0] in ("\"", "'") and value[-1] == value[0]:
                    value = value[1:-1]
                values[key.strip().lower()] = value

        for field_name in cls.model_fields:
            env_name = field_name.upper()
            if env_name in os.environ:
                values[field_name] = os.environ[env_name]
        return cls.model_validate(values)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def knowledge_model(self) -> str:
        return self.knowledge_qwen_model if self.knowledge_provider == "qwen" else self.glm_model

    @property
    def retrieval_api_base_url(self) -> str:
        if self.dashscope_http_base_url:
            return self.dashscope_http_base_url.rstrip("/")
        base = self.dashscope_base_url.rstrip("/")
        suffix = "/compatible-mode/v1"
        if not base.endswith(suffix):
            raise ValueError("Set DASHSCOPE_HTTP_BASE_URL for a custom embedding/rerank endpoint")
        return base[:-len(suffix)] + "/api/v1"

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
