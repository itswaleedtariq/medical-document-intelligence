from pathlib import Path

from fastapi import UploadFile

from backend.app.ingestion.exceptions import (
    EmptyPDFError,
    PDFTooLargeError,
    UnsupportedFileTypeError,
)

ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}

READ_CHUNK_SIZE = 1024 * 1024  # 1 MB


def get_safe_filename(filename: str | None) -> str:
    """
    Remove directory components from a user-provided filename.

    Both Windows and Unix separators are normalized before taking
    the final filename component.
    """
    normalized = (filename or "uploaded.pdf").replace("\\", "/")
    safe_filename = Path(normalized).name.strip()

    return safe_filename or "uploaded.pdf"


async def read_and_validate_pdf_upload(
    upload: UploadFile,
    max_size_mb: int,
) -> tuple[bytes, str]:
    """
    Read an uploaded PDF in chunks while enforcing basic validation.

    Returns:
        A tuple containing the PDF bytes and sanitized display filename.
    """
    safe_filename = get_safe_filename(upload.filename)

    if Path(safe_filename).suffix.lower() != ".pdf":
        raise UnsupportedFileTypeError(
            "Only files with the .pdf extension are supported."
        )

    content_type = (upload.content_type or "").lower()

    if content_type and content_type not in ALLOWED_PDF_CONTENT_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported content type: {content_type}"
        )

    max_size_bytes = max_size_mb * 1024 * 1024
    total_size = 0
    chunks: list[bytes] = []

    await upload.seek(0)

    while True:
        chunk = await upload.read(READ_CHUNK_SIZE)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > max_size_bytes:
            raise PDFTooLargeError(
                f"PDF exceeds the {max_size_mb} MB upload limit."
            )

        chunks.append(chunk)

    pdf_bytes = b"".join(chunks)

    if not pdf_bytes:
        raise EmptyPDFError("The uploaded PDF is empty.")

    # A valid PDF header should normally appear near the start of the file.
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise UnsupportedFileTypeError(
            "The uploaded file does not contain a valid PDF header."
        )

    return pdf_bytes, safe_filename