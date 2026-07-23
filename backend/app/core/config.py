from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medical Document Intelligence"
    app_env: str = "development"
    debug: bool = True

    model_id: str = "google/medgemma-1.5-4b-it"

    max_upload_mb: int = Field(default=20, ge=1)
    max_pdf_pages: int = Field(default=100, ge=1)
    min_page_text_chars: int = Field(default=20, ge=0)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    chunk_size_chars: int = Field(default=800, ge=100)
    chunk_overlap_chars: int = Field(default=120, ge=0)

    retrieval_top_k: int = Field(default=5, ge=1, le=20)

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

    return settings
