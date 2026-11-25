from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    llm_model: Literal["dummy", "gemini", "ollama", "transformers"] = "dummy"
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    gemini_request_timeout: float = 30
    gemini_retries: int = 1
    ollama_model: str = "llama3"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.7
    ollama_request_timeout: float = 120
    transformers_model: str | None = None

    db_option: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_db_path: str = "sqlite.db"
    db_name: str = "semantic_search_db"
    db_user: str = "vector_admin"
    db_password: SecretStr | None = None
    db_host: str = "localhost"
    db_port: int = 5433
    postgres_pool_min: int = 1
    postgres_pool_max: int = 8
    job_upload_dir: str = str(PROJECT_ROOT / "data" / "uploads")
    admin_api_key: SecretStr | None = None
    cors_allowed_origins: str = "http://localhost:5500,http://127.0.0.1:5500"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    ocr_enabled: bool = False
    ocr_lang: str = "eng"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.llm_model == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required when LLM_MODEL=gemini")
            if not self.gemini_model:
                raise ValueError("GEMINI_MODEL is required when LLM_MODEL=gemini")
        if self.llm_model == "transformers" and not self.transformers_model:
            raise ValueError(
                "TRANSFORMERS_MODEL is required when LLM_MODEL=transformers"
            )
        if self.db_option == "postgres" and not self.db_password:
            raise ValueError("DB_PASSWORD is required when DB_OPTION=postgres")
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be greater than zero")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "CHUNK_SIZE must be positive and CHUNK_OVERLAP must be "
                "non-negative and smaller than CHUNK_SIZE"
            )
        if (
            self.postgres_pool_min < 1
            or self.postgres_pool_max < self.postgres_pool_min
        ):
            raise ValueError(
                "POSTGRES_POOL_MAX must be greater than or equal to "
                "POSTGRES_POOL_MIN, and both must be positive"
            )
        if (
            self.gemini_request_timeout <= 0
            or self.gemini_retries < 0
            or self.ollama_request_timeout <= 0
        ):
            raise ValueError(
                "Provider timeouts must be positive and retries non-negative"
            )
        if not 0 <= self.ollama_temperature <= 2:
            raise ValueError("OLLAMA_TEMPERATURE must be between 0 and 2")
        if self.db_port < 1 or self.db_port > 65535:
            raise ValueError("DB_PORT must be between 1 and 65535")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
