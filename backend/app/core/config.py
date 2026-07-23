from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medical Document Intelligence"
    app_env: str = "development"
    debug: bool = True

    model_id: str = "google/medgemma-1.5-4b-it"
    model_provider: Literal["mock", "http"] = "mock"
    model_api_url: str | None = None
    model_api_token: str | None = None
    model_request_timeout_seconds: float = Field(
        default=180,
        gt=0,
    )
    model_max_new_tokens: int = Field(
        default=700,
        ge=100,
        le=2000,
    )

    max_upload_mb: int = Field(default=20, ge=1)
    max_pdf_pages: int = Field(default=100, ge=1)
    min_page_text_chars: int = Field(default=20, ge=0)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    chunk_size_chars: int = Field(default=800, ge=100)
    chunk_overlap_chars: int = Field(default=120, ge=0)

    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    min_retrieval_similarity: float = Field(
        default=0.20,
        ge=0,
        le=1,
    )

    chroma_path: str = "./chroma_db"
    chroma_collection: str = "medical_document_chunks"
    upload_path: str = "./data/raw"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    if settings.chunk_overlap_chars >= settings.chunk_size_chars:
        raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS.")

    if settings.model_provider == "http" and not settings.model_api_url:
        raise ValueError("MODEL_API_URL is required when MODEL_PROVIDER=http.")

    return settings
