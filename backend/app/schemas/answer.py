from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSED = "refused"


class AnswerFact(BaseModel):
    statement: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)


class ModelGroundedAnswer(BaseModel):
    status: AnswerStatus
    answer: str = Field(min_length=1)
    facts: list[AnswerFact] = Field(default_factory=list)
    safety_note: str | None = None
    limitations: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    evidence_id: str
    chunk_id: str
    filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    text: str
    similarity_score: float = Field(ge=0, le=1)
    citation: str


class DocumentAskRequest(BaseModel):
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


class DocumentAskResponse(BaseModel):
    document_id: UUID
    question: str

    status: AnswerStatus
    answer: str
    facts: list[AnswerFact]

    evidence: list[EvidenceItem]
    safety_note: str | None
    limitations: list[str]

    model_id: str
    provider: str
