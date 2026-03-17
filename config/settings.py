from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .nim import NimSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")


    nvidia_nim_api_key: str = ""

    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )


    model: str = ""

    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )

    http_read_timeout: float = Field(
        default=300.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=2.0, validation_alias="HTTP_CONNECT_TIMEOUT"
    )

    fast_prefix_detection: bool = True

    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    context_management_enabled: bool = Field(
        default=True, validation_alias="CONTEXT_MANAGEMENT_ENABLED"
    )
    context_compact_instructions: str = Field(
        default="Focus on preserving code snippets, variable names, and technical decisions.",
        validation_alias="CONTEXT_COMPACT_INSTRUCTIONS",
    )
    thinking_keep_last_n: int = Field(
        default=2, validation_alias="THINKING_KEEP_LAST_N"
    )
    tool_filter_enabled: bool = Field(
        default=True, validation_alias="TOOL_FILTER_ENABLED"
    )
    tool_filter_lookback_turns: int = Field(
        default=4, validation_alias="TOOL_FILTER_LOOKBACK_TURNS"
    )

    nim: NimSettings = Field(default_factory=NimSettings)


    memory_enabled: bool = Field(default=True, validation_alias="MEMORY_ENABLED")
    memory_token_budget: int = Field(default=400, validation_alias="MEMORY_TOKEN_BUDGET")
    memory_top_k: int = Field(default=8, validation_alias="MEMORY_TOP_K")
    memory_dedup_threshold: float = Field(default=0.98, validation_alias="MEMORY_DEDUP_THRESHOLD")
    memory_auto_reduce_threshold: int = Field(default=500, validation_alias="MEMORY_AUTO_REDUCE_THRESHOLD")
    memory_db_path: str = Field(default="./data/memory.db", validation_alias="MEMORY_DB_PATH")
    memory_chroma_path: str = Field(default="./data/chroma", validation_alias="MEMORY_CHROMA_PATH")
    memory_embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        validation_alias="MEMORY_EMBEDDING_MODEL",
    )

    allow_dangerous_permissions: bool = Field(
        default=False, validation_alias="ALLOW_DANGEROUS_PERMISSIONS"
    )

    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "server.log"


    @field_validator("model", mode="before")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        valid_providers = (
            "nvidia_nim",
            "open_router",
            "openrouter",
            "lmstudio",
            "anthropic",
            "gemini",
            "groq",
            "deepseek",
            "grok",
        )
        if not v or not v.strip():
            return ""
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(valid_providers)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: '{provider}'. "
                f"Supported: {', '.join(valid_providers)}"
            )
        return v


    @property
    def provider_type(self) -> str:
        """Extract provider type from the model string."""
        if not self.model or "/" not in self.model:
            return "passthrough"
        return self.model.split("/", 1)[0]

    @property
    def model_name(self) -> str:
        """Extract the actual model name from the model string."""
        if not self.model or "/" not in self.model:
            return ""
        return self.model.split("/", 1)[1]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
