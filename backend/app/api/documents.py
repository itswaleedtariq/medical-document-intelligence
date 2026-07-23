from typing import Annotated

from fastapi import (
    APIRouter,
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
from backend.app.ingestion.pdf_extractor import extract_pdf_bytes
from backend.app.ingestion.pdf_validator import (
    read_and_validate_pdf_upload,
)
from backend.app.schemas.document import PDFExtractionResponse

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


@router.post(
    "/extract",
    response_model=PDFExtractionResponse,
    summary="Extract text from a medical PDF",
)
async def extract_medical_pdf(
    file: Annotated[
        UploadFile,
        File(description="A text-based medical PDF document"),
    ],
) -> PDFExtractionResponse:
    settings = get_settings()

    try:
        pdf_bytes, safe_filename = await read_and_validate_pdf_upload(
            upload=file,
            max_size_mb=settings.max_upload_mb,
        )

        return extract_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename=safe_filename,
            max_pages=settings.max_pdf_pages,
            min_page_text_chars=settings.min_page_text_chars,
        )

    except PDFTooLargeError as error:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(error),
        ) from error

    except UnsupportedFileTypeError as error:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(error),
        ) from error

    except (
        EmptyPDFError,
        EncryptedPDFError,
        InvalidPDFError,
        PDFPageLimitError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    finally:
        await file.close()