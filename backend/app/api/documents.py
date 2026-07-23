from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from backend.app.core.config import get_settings
from backend.app.ingestion.exceptions import (
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFPageLimitError,
    PDFTooLargeError,
    UnsupportedFileTypeError,
)
from backend.app.ingestion.pdf_extractor import (
    extract_pdf_bytes,
)
from backend.app.ingestion.pdf_validator import (
    read_and_validate_pdf_upload,
)
from backend.app.retrieval.exceptions import (
    EmbeddingError,
    NoExtractableTextError,
    VectorStoreError,
)
from backend.app.retrieval.service import (
    RetrievalService,
    get_retrieval_service,
)
from backend.app.schemas.document import (
    DocumentIndexResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    PDFExtractionResponse,
)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)

RetrievalServiceDependency = Annotated[
    RetrievalService,
    Depends(get_retrieval_service),
]


def _extract_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> PDFExtractionResponse:
    settings = get_settings()

    return extract_pdf_bytes(
        pdf_bytes=pdf_bytes,
        filename=filename,
        max_pages=settings.max_pdf_pages,
        min_page_text_chars=(settings.min_page_text_chars),
    )


def _raise_pdf_http_error(
    error: Exception,
) -> None:
    if isinstance(error, PDFTooLargeError):
        raise HTTPException(
            status_code=(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE),
            detail=str(error),
        ) from error

    if isinstance(error, UnsupportedFileTypeError):
        raise HTTPException(
            status_code=(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
            detail=str(error),
        ) from error

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(error),
    ) from error


@router.post(
    "/extract",
    response_model=PDFExtractionResponse,
    summary="Extract page-aware text from a medical PDF",
)
async def extract_medical_pdf(
    file: Annotated[
        UploadFile,
        File(description="A medical PDF document"),
    ],
) -> PDFExtractionResponse:
    settings = get_settings()

    try:
        pdf_bytes, safe_filename = await read_and_validate_pdf_upload(
            upload=file,
            max_size_mb=settings.max_upload_mb,
        )

        return _extract_pdf(
            pdf_bytes=pdf_bytes,
            filename=safe_filename,
        )

    except (
        PDFTooLargeError,
        UnsupportedFileTypeError,
        EmptyPDFError,
        EncryptedPDFError,
        InvalidPDFError,
        PDFPageLimitError,
    ) as error:
        _raise_pdf_http_error(error)
        raise

    finally:
        await file.close()


@router.post(
    "/index",
    response_model=DocumentIndexResponse,
    summary="Extract and index a medical PDF",
)
async def index_medical_pdf(
    file: Annotated[
        UploadFile,
        File(description="A text-based medical PDF"),
    ],
    retrieval_service: RetrievalServiceDependency,
) -> DocumentIndexResponse:
    settings = get_settings()

    try:
        pdf_bytes, safe_filename = await read_and_validate_pdf_upload(
            upload=file,
            max_size_mb=settings.max_upload_mb,
        )

        extraction = _extract_pdf(
            pdf_bytes=pdf_bytes,
            filename=safe_filename,
        )

        return retrieval_service.index_document(
            extraction=extraction,
        )

    except (
        PDFTooLargeError,
        UnsupportedFileTypeError,
        EmptyPDFError,
        EncryptedPDFError,
        InvalidPDFError,
        PDFPageLimitError,
    ) as error:
        _raise_pdf_http_error(error)
        raise

    except NoExtractableTextError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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

    finally:
        await file.close()


@router.post(
    "/search",
    response_model=DocumentSearchResponse,
    summary="Search an indexed medical document",
)
def search_medical_document(
    request: DocumentSearchRequest,
    retrieval_service: RetrievalServiceDependency,
) -> DocumentSearchResponse:
    try:
        return retrieval_service.search_document(
            document_id=request.document_id,
            question=request.question,
            top_k=request.top_k,
        )

    except (
        EmbeddingError,
        VectorStoreError,
    ) as error:
        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail=str(error),
        ) from error
