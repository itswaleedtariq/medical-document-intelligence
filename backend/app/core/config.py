from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medical Document Intelligence"
    app_env: str = "development"
    debug: bool = True

    model_id: str = "google/medgemma-1.5-4b-it"

    max_upload_mb: int = 20
    max_pdf_pages: int = 100
    min_page_text_chars: int = 20

    chroma_path: str = "./chroma_db"
    upload_path: str = "./data/raw"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()