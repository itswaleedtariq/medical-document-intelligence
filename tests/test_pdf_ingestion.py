from io import BytesIO

import pymupdf
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.ingestion.exceptions import PDFPageLimitError
from backend.app.ingestion.pdf_extractor import extract_pdf_bytes
from backend.app.main import app

client = TestClient(app)


def create_text_pdf(page_texts: list[str]) -> bytes:
    document = pymupdf.open()

    try:
        for text in page_texts:
            page = document.new_page()
            page.insert_text(
                (72, 72),
                text,
                fontsize=11,
            )

        return document.tobytes()
    finally:
        document.close()


def create_image_only_pdf() -> bytes:
    image_buffer = BytesIO()

    image = Image.new(
        "RGB",
        (400, 200),
        "white",
    )
    image.save(
        image_buffer,
        format="PNG",
    )

    document = pymupdf.open()

    try:
        page = document.new_page()
        page.insert_image(
            page.rect,
            stream=image_buffer.getvalue(),
        )

        return document.tobytes()
    finally:
        document.close()


def test_extract_text_pdf() -> None:
    pdf_bytes = create_text_pdf(
        [
            "Hemoglobin: 10.2 g/dL",
            "Platelets: 250 x10^9/L",
        ]
    )

    response = client.post(
        "/api/documents/extract",
        files={
            "file": (
                "synthetic-report.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["filename"] == "synthetic-report.pdf"
    assert payload["page_count"] == 2
    assert payload["text_page_count"] == 2
    assert payload["ocr_candidate_page_count"] == 0
    assert "Hemoglobin" in payload["pages"][0]["text"]
    assert payload["pages"][0]["page_number"] == 1


def test_reject_non_pdf_content() -> None:
    response = client.post(
        "/api/documents/extract",
        files={
            "file": (
                "fake.pdf",
                b"This is not a PDF file.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415


def test_reject_wrong_extension() -> None:
    pdf_bytes = create_text_pdf(["Synthetic report"])

    response = client.post(
        "/api/documents/extract",
        files={
            "file": (
                "report.txt",
                pdf_bytes,
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_image_only_page_is_marked_for_ocr() -> None:
    pdf_bytes = create_image_only_pdf()

    result = extract_pdf_bytes(
        pdf_bytes=pdf_bytes,
        filename="scanned-report.pdf",
        max_pages=10,
        min_page_text_chars=20,
    )

    assert result.page_count == 1
    assert result.pages[0].has_images is True
    assert result.pages[0].needs_ocr is True
    assert result.ocr_candidate_page_count == 1


def test_page_limit_is_enforced() -> None:
    pdf_bytes = create_text_pdf(
        [
            "Page one",
            "Page two",
        ]
    )

    try:
        extract_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename="two-pages.pdf",
            max_pages=1,
            min_page_text_chars=20,
        )
    except PDFPageLimitError:
        return

    raise AssertionError("PDFPageLimitError was not raised.")
