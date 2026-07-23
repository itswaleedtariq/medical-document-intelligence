from uuid import uuid4

import pymupdf

from backend.app.ingestion.exceptions import (
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFPageLimitError,
)
from backend.app.schemas.document import (
    ExtractedPage,
    PDFExtractionResponse,
)


def extract_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    max_pages: int,
    min_page_text_chars: int,
) -> PDFExtractionResponse:
    """
    Extract page-aware text and metadata from PDF bytes.

    OCR is not executed here. Pages containing images but very little
    extractable text are marked as candidates for OCR.
    """
    try:
        document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        )
    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
        RuntimeError,
    ) as error:
        raise InvalidPDFError(
            "The uploaded PDF is corrupted or cannot be opened."
        ) from error

    try:
        if document.needs_pass or document.is_encrypted:
            raise EncryptedPDFError(
                "Password-protected PDFs are not supported."
            )

        if document.page_count == 0:
            raise EmptyPDFError("The PDF does not contain any pages.")

        if document.page_count > max_pages:
            raise PDFPageLimitError(
                f"The PDF contains {document.page_count} pages. "
                f"The maximum allowed is {max_pages}."
            )

        extracted_pages: list[ExtractedPage] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            try:
                text = page.get_text(
                    "text",
                    sort=True,
                ).strip()

                has_images = bool(page.get_images(full=True))
            except RuntimeError as error:
                raise InvalidPDFError(
                    f"Text extraction failed on page {page_index + 1}."
                ) from error

            character_count = len(text)

            # This is a heuristic, not a definite scanned-page diagnosis.
            needs_ocr = (
                has_images
                and character_count < min_page_text_chars
            )

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    text=text,
                    character_count=character_count,
                    has_images=has_images,
                    needs_ocr=needs_ocr,
                )
            )

        text_page_count = sum(
            page.character_count > 0
            for page in extracted_pages
        )

        ocr_candidate_page_count = sum(
            page.needs_ocr
            for page in extracted_pages
        )

        total_characters = sum(
            page.character_count
            for page in extracted_pages
        )

        return PDFExtractionResponse(
            document_id=uuid4(),
            filename=filename,
            file_size_bytes=len(pdf_bytes),
            page_count=document.page_count,
            text_page_count=text_page_count,
            ocr_candidate_page_count=ocr_candidate_page_count,
            total_characters=total_characters,
            pages=extracted_pages,
        )

    finally:
        document.close()