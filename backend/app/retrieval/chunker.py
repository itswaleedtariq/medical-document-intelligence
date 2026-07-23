import re
from uuid import UUID

from backend.app.schemas.document import (
    DocumentChunk,
    ExtractedPage,
)


def normalize_page_text(text: str) -> str:
    """
    Normalize whitespace while retaining useful line boundaries.

    Empty lines are removed, but separate report rows remain on
    separate lines.
    """
    normalized_lines: list[str] = []

    for line in text.replace("\r\n", "\n").split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()

        if cleaned_line:
            normalized_lines.append(cleaned_line)

    return "\n".join(normalized_lines)


def _find_chunk_end(
    text: str,
    start: int,
    maximum_end: int,
    minimum_end: int,
) -> int:
    """
    Prefer ending a chunk at a natural text boundary.
    """
    separators = (
        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " ",
    )

    for separator in separators:
        boundary = text.rfind(
            separator,
            minimum_end,
            maximum_end,
        )

        if boundary != -1:
            return boundary + len(separator)

    return maximum_end


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping character-based chunks.

    Returns:
        Tuples containing chunk text, start character and
        end character.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    normalized_text = normalize_page_text(text)

    if not normalized_text:
        return []

    chunks: list[tuple[str, int, int]] = []
    text_length = len(normalized_text)
    start = 0

    while start < text_length:
        maximum_end = min(
            start + chunk_size,
            text_length,
        )

        if maximum_end == text_length:
            end = text_length
        else:
            minimum_end = min(
                start + max(chunk_size // 2, 1),
                maximum_end,
            )

            end = _find_chunk_end(
                text=normalized_text,
                start=start,
                maximum_end=maximum_end,
                minimum_end=minimum_end,
            )

        if end <= start:
            end = maximum_end

        chunk_text = normalized_text[start:end].strip()

        if chunk_text:
            chunks.append(
                (
                    chunk_text,
                    start,
                    end,
                )
            )

        if end >= text_length:
            break

        next_start = max(
            end - overlap,
            start + 1,
        )

        # Avoid beginning the next chunk inside a word.
        if (
            next_start > 0
            and next_start < text_length
            and not normalized_text[next_start - 1].isspace()
            and not normalized_text[next_start].isspace()
        ):
            next_space = normalized_text.find(
                " ",
                next_start,
                min(next_start + 50, text_length),
            )

            if next_space != -1:
                next_start = next_space + 1

        while next_start < text_length and normalized_text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


def chunk_document_pages(
    document_id: UUID,
    filename: str,
    pages: list[ExtractedPage],
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """
    Create page-aware chunks.

    Chunks never cross page boundaries, allowing every retrieved
    result to have a reliable source-page citation.
    """
    document_chunks: list[DocumentChunk] = []

    for page in pages:
        page_chunks = split_text_into_chunks(
            text=page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, (
            chunk_text,
            start_character,
            end_character,
        ) in enumerate(page_chunks, start=1):
            chunk_id = f"{document_id}-page-{page.page_number}-chunk-{chunk_index}"

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_character=start_character,
                    end_character=end_character,
                )
            )

    return document_chunks
