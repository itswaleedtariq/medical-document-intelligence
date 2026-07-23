from functools import lru_cache
from uuid import UUID

from backend.app.core.config import get_settings
from backend.app.inference.output_parser import (
    parse_grounded_model_output,
)
from backend.app.inference.prompts import (
    build_grounded_medical_prompt,
)
from backend.app.inference.providers import (
    ModelProvider,
    get_model_provider,
)
from backend.app.retrieval.service import (
    RetrievalService,
    get_retrieval_service,
)
from backend.app.safety.request_guard import (
    get_medical_request_refusal,
)
from backend.app.schemas.answer import (
    AnswerStatus,
    DocumentAskResponse,
    EvidenceItem,
)


class GroundedQAService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        model_provider: ModelProvider,
        minimum_similarity: float,
        max_new_tokens: int,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.model_provider = model_provider
        self.minimum_similarity = minimum_similarity
        self.max_new_tokens = max_new_tokens

    def answer_question(
        self,
        document_id: UUID,
        question: str,
        top_k: int | None = None,
    ) -> DocumentAskResponse:
        cleaned_question = question.strip()

        refusal = get_medical_request_refusal(cleaned_question)

        if refusal:
            return DocumentAskResponse(
                document_id=document_id,
                question=cleaned_question,
                status=AnswerStatus.REFUSED,
                answer=refusal,
                facts=[],
                evidence=[],
                safety_note=(
                    "Personalized diagnosis and treatment requests "
                    "are outside this system's scope."
                ),
                limitations=[
                    "The assistant only explains information "
                    "explicitly present in uploaded documents."
                ],
                model_id=self.model_provider.model_id,
                provider=self.model_provider.name,
            )

        search_response = self.retrieval_service.search_document(
            document_id=document_id,
            question=cleaned_question,
            top_k=top_k,
        )

        qualified_evidence = [
            result
            for result in search_response.results
            if (result.similarity_score >= self.minimum_similarity)
        ]

        if not qualified_evidence:
            return DocumentAskResponse(
                document_id=document_id,
                question=cleaned_question,
                status=(AnswerStatus.INSUFFICIENT_EVIDENCE),
                answer=(
                    "The indexed document does not contain enough "
                    "relevant evidence to answer this question."
                ),
                facts=[],
                evidence=[],
                safety_note=None,
                limitations=[
                    "No retrieved passage met the configured similarity threshold."
                ],
                model_id=self.model_provider.model_id,
                provider=self.model_provider.name,
            )

        prompt = build_grounded_medical_prompt(
            question=cleaned_question,
            evidence=qualified_evidence,
        )

        raw_output = self.model_provider.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
        )

        evidence_ids = {
            f"E{index}"
            for index in range(
                1,
                len(qualified_evidence) + 1,
            )
        }

        parsed_answer = parse_grounded_model_output(
            raw_output=raw_output,
            allowed_evidence_ids=evidence_ids,
        )

        evidence_items = [
            EvidenceItem(
                evidence_id=f"E{index}",
                chunk_id=result.chunk_id,
                filename=result.filename,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
                text=result.text,
                similarity_score=(result.similarity_score),
                citation=result.citation,
            )
            for index, result in enumerate(
                qualified_evidence,
                start=1,
            )
        ]

        return DocumentAskResponse(
            document_id=document_id,
            question=cleaned_question,
            status=parsed_answer.status,
            answer=parsed_answer.answer,
            facts=parsed_answer.facts,
            evidence=evidence_items,
            safety_note=parsed_answer.safety_note,
            limitations=parsed_answer.limitations,
            model_id=self.model_provider.model_id,
            provider=self.model_provider.name,
        )


@lru_cache(maxsize=1)
def get_grounded_qa_service() -> GroundedQAService:
    settings = get_settings()

    return GroundedQAService(
        retrieval_service=get_retrieval_service(),
        model_provider=get_model_provider(),
        minimum_similarity=(settings.min_retrieval_similarity),
        max_new_tokens=settings.model_max_new_tokens,
    )
