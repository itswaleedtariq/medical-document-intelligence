from uuid import uuid4

import chromadb

from backend.app.retrieval.chunker import (
    chunk_document_pages,
    split_text_into_chunks,
)
from backend.app.retrieval.vector_store import (
    ChromaVectorStore,
)
from backend.app.schemas.document import (
    DocumentChunk,
    ExtractedPage,
)


def test_text_is_split_with_overlap() -> None:
    text = " ".join(f"word-{index}" for index in range(200))

    chunks = split_text_into_chunks(
        text=text,
        chunk_size=200,
        overlap=40,
    )

    assert len(chunks) > 1

    for chunk_text, start, end in chunks:
        assert chunk_text
        assert start >= 0
        assert end > start
        assert len(chunk_text) <= 200


def test_chunking_preserves_page_number() -> None:
    document_id = uuid4()

    pages = [
        ExtractedPage(
            page_number=1,
            text="Hemoglobin result is 10.2 g/dL.",
            character_count=32,
            has_images=False,
            needs_ocr=False,
        ),
        ExtractedPage(
            page_number=2,
            text="Platelet result is 250 x10^9/L.",
            character_count=34,
            has_images=False,
            needs_ocr=False,
        ),
    ]

    chunks = chunk_document_pages(
        document_id=document_id,
        filename="report.pdf",
        pages=pages,
        chunk_size=800,
        overlap=120,
    )

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].document_id == document_id
    assert chunks[0].chunk_index == 1


def test_chroma_search_filters_by_document() -> None:
    client = chromadb.EphemeralClient()

    store = ChromaVectorStore(
        path="unused",
        collection_name="test-medical-chunks",
        client=client,
    )

    first_document_id = uuid4()
    second_document_id = uuid4()

    chunks = [
        DocumentChunk(
            chunk_id="chunk-one",
            document_id=first_document_id,
            filename="first.pdf",
            page_number=1,
            chunk_index=1,
            text="Hemoglobin is 10.2 g/dL.",
            start_character=0,
            end_character=24,
        ),
        DocumentChunk(
            chunk_id="chunk-two",
            document_id=first_document_id,
            filename="first.pdf",
            page_number=2,
            chunk_index=1,
            text="Platelets are 250 x10^9/L.",
            start_character=0,
            end_character=27,
        ),
        DocumentChunk(
            chunk_id="chunk-three",
            document_id=second_document_id,
            filename="second.pdf",
            page_number=1,
            chunk_index=1,
            text="Hemoglobin is 14.0 g/dL.",
            start_character=0,
            end_character=24,
        ),
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    indexed_count = store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    assert indexed_count == 3

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        document_id=str(first_document_id),
        top_k=3,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "chunk-one"
    assert results[0].page_number == 1

    returned_filenames = {result.filename for result in results}

    assert returned_filenames == {"first.pdf"}
