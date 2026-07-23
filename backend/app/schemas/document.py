from uuid import UUID

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    has_images: bool
    needs_ocr: bool


class PDFExtractionResponse(BaseModel):
    document_id: UUID
    filename: str
    file_size_bytes: int = Field(ge=1)

    page_count: int = Field(ge=1)
    text_page_count: int = Field(ge=0)
    ocr_candidate_page_count: int = Field(ge=0)
    total_characters: int = Field(ge=0)

    pages: list[ExtractedPage]


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: UUID
    filename: str

    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)

    text: str = Field(min_length=1)
    start_character: int = Field(ge=0)
    end_character: int = Field(ge=1)


class DocumentIndexResponse(BaseModel):
    document_id: UUID
    filename: str

    page_count: int = Field(ge=1)
    indexed_page_count: int = Field(ge=0)
    ocr_candidate_page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=1)

    embedding_model: str
    collection_name: str


class DocumentSearchRequest(BaseModel):
    document_id: UUID
    question: str = Field(
        min_length=3,
        max_length=2000,
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    filename: str

    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)

    text: str
    distance: float
    similarity_score: float = Field(ge=0, le=1)

    citation: str


class DocumentSearchResponse(BaseModel):
    document_id: UUID
    question: str
    result_count: int = Field(ge=0)
    results: list[RetrievedChunk]
