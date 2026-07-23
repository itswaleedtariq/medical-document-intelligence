import json
from uuid import UUID, uuid4

import pytest

from backend.app.inference.exceptions import (
    ModelOutputError,
)
from backend.app.inference.output_parser import (
    parse_grounded_model_output,
)
from backend.app.inference.qa_service import (
    GroundedQAService,
)
from backend.app.schemas.answer import AnswerStatus
from backend.app.schemas.document import (
    DocumentSearchResponse,
    RetrievedChunk,
)


class FakeRetrievalService:
    def __init__(
        self,
        results: list[RetrievedChunk],
    ) -> None:
        self.results = results

    def search_document(
        self,
        document_id: UUID,
        question: str,
        top_k: int | None = None,
    ) -> DocumentSearchResponse:
        del top_k

        return DocumentSearchResponse(
            document_id=document_id,
            question=question,
            result_count=len(self.results),
            results=self.results,
        )


class FakeModelProvider:
    name = "fake"
    model_id = "fake-medgemma"

    def __init__(self) -> None:
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        del max_new_tokens

        self.call_count += 1

        assert "[E1]" in prompt
        assert "Hemoglobin" in prompt

        return json.dumps(
            {
                "status": "answered",
                "answer": ("The hemoglobin result was 10.2 g/dL."),
                "facts": [
                    {
                        "statement": ("Hemoglobin was reported as 10.2 g/dL."),
                        "citations": ["E1"],
                    }
                ],
                "safety_note": None,
                "limitations": [],
            }
        )


def create_retrieved_chunk(
    similarity_score: float = 0.90,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-one",
        filename="report.pdf",
        page_number=1,
        chunk_index=1,
        text="Hemoglobin result: 10.2 g/dL.",
        distance=1.0 - similarity_score,
        similarity_score=similarity_score,
        citation="report.pdf, page 1",
    )


def test_grounded_answer_contains_evidence() -> None:
    provider = FakeModelProvider()

    service = GroundedQAService(
        retrieval_service=FakeRetrievalService([create_retrieved_chunk()]),
        model_provider=provider,
        minimum_similarity=0.20,
        max_new_tokens=500,
    )

    response = service.answer_question(
        document_id=uuid4(),
        question="What was the hemoglobin result?",
    )

    assert response.status == AnswerStatus.ANSWERED
    assert response.provider == "fake"
    assert response.evidence[0].evidence_id == "E1"
    assert response.evidence[0].page_number == 1
    assert response.facts[0].citations == ["E1"]
    assert provider.call_count == 1


def test_unsafe_question_is_refused_before_model() -> None:
    provider = FakeModelProvider()

    service = GroundedQAService(
        retrieval_service=FakeRetrievalService([create_retrieved_chunk()]),
        model_provider=provider,
        minimum_similarity=0.20,
        max_new_tokens=500,
    )

    response = service.answer_question(
        document_id=uuid4(),
        question=("Diagnose me and prescribe medication."),
    )

    assert response.status == AnswerStatus.REFUSED
    assert response.evidence == []
    assert provider.call_count == 0


def test_weak_evidence_skips_model() -> None:
    provider = FakeModelProvider()

    service = GroundedQAService(
        retrieval_service=FakeRetrievalService(
            [
                create_retrieved_chunk(
                    similarity_score=0.10,
                )
            ]
        ),
        model_provider=provider,
        minimum_similarity=0.20,
        max_new_tokens=500,
    )

    response = service.answer_question(
        document_id=uuid4(),
        question="What was the glucose result?",
    )

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE
    assert provider.call_count == 0


def test_unknown_model_citation_is_rejected() -> None:
    raw_output = json.dumps(
        {
            "status": "answered",
            "answer": "A result was found.",
            "facts": [
                {
                    "statement": "A result was found.",
                    "citations": ["E9"],
                }
            ],
            "safety_note": None,
            "limitations": [],
        }
    )

    with pytest.raises(ModelOutputError):
        parse_grounded_model_output(
            raw_output=raw_output,
            allowed_evidence_ids={"E1"},
        )
