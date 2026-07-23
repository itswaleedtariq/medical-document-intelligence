from fastapi import FastAPI

from backend.app.api.documents import router as documents_router
from backend.app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Medical Document Intelligence API",
    description=(
        "Educational medical document extraction and explanation API. "
        "This application does not provide medical diagnoses or treatment."
    ),
    version="0.2.0",
    debug=settings.debug,
)

app.include_router(documents_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "running",
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "version": "0.2.0",
    }