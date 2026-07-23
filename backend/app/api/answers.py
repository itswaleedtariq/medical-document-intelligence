from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from backend.app.inference.exceptions import (
    ModelOutputError,
    ModelProviderError,
)
from backend.app.inference.qa_service import (
    GroundedQAService,
    get_grounded_qa_service,
)
from backend.app.retrieval.exceptions import (
    EmbeddingError,
    VectorStoreError,
)
from backend.app.schemas.answer import (
    DocumentAskRequest,
    DocumentAskResponse,
)

router = APIRouter(
    prefix="/api/documents",
    tags=["Document Answers"],
)

GroundedQAServiceDependency = Annotated[
    GroundedQAService,
    Depends(get_grounded_qa_service),
]


@router.post(
    "/ask",
    response_model=DocumentAskResponse,
    summary="Ask a grounded question about an indexed PDF",
)
def ask_medical_document(
    request: DocumentAskRequest,
    qa_service: GroundedQAServiceDependency,
) -> DocumentAskResponse:
    try:
        return qa_service.answer_question(
            document_id=request.document_id,
            question=request.question,
            top_k=request.top_k,
        )

    except (
        ModelProviderError,
        ModelOutputError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    except (
        EmbeddingError,
        VectorStoreError,
    ) as error:
        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=str(error),
        ) from error
