from fastapi import FastAPI

app = FastAPI(
    title="Medical Document Intelligence API",
    description=(
        "Educational medical document extraction and explanation API. "
        "This application does not provide medical diagnoses or treatment."
    ),
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Medical Document Intelligence API",
        "status": "running",
    }


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "version": "0.1.0",
    }
