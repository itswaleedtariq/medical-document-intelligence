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