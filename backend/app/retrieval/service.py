from functools import lru_cache
from uuid import UUID

from backend.app.core.config import get_settings
from backend.app.retrieval.chunker import (
    chunk_document_pages,
)
from backend.app.retrieval.embeddings import (
    EmbeddingService,
    get_embedding_service,
)
from backend.app.retrieval.exceptions import (
    NoExtractableTextError,
)
from backend.app.retrieval.vector_store import (
    ChromaVectorStore,
    get_vector_store,
)
from backend.app.schemas.document import (
    DocumentIndexResponse,
    DocumentSearchResponse,
    PDFExtractionResponse,
)


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        chunk_size: int,
        chunk_overlap: int,
        default_top_k: int,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_top_k = default_top_k

    def index_document(
        self,
        extraction: PDFExtractionResponse,
    ) -> DocumentIndexResponse:
        chunks = chunk_document_pages(
            document_id=extraction.document_id,
            filename=extraction.filename,
            pages=extraction.pages,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        if not chunks:
            raise NoExtractableTextError(
                "The PDF contains no extractable text. It may require OCR."
            )

        embeddings = self.embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )

        indexed_count = self.vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        indexed_pages = {chunk.page_number for chunk in chunks}

        return DocumentIndexResponse(
            document_id=extraction.document_id,
            filename=extraction.filename,
            page_count=extraction.page_count,
            indexed_page_count=len(indexed_pages),
            ocr_candidate_page_count=(extraction.ocr_candidate_page_count),
            chunk_count=indexed_count,
            embedding_model=(self.embedding_service.model_name),
            collection_name=(self.vector_store.collection_name),
        )

    def search_document(
        self,
        document_id: UUID,
        question: str,
        top_k: int | None = None,
    ) -> DocumentSearchResponse:
        cleaned_question = question.strip()

        effective_top_k = top_k if top_k is not None else self.default_top_k

        query_embedding = self.embedding_service.embed_query(cleaned_question)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            document_id=str(document_id),
            top_k=effective_top_k,
        )

        return DocumentSearchResponse(
            document_id=document_id,
            question=cleaned_question,
            result_count=len(results),
            results=results,
        )


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    settings = get_settings()

    return RetrievalService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        chunk_size=settings.chunk_size_chars,
        chunk_overlap=settings.chunk_overlap_chars,
        default_top_k=settings.retrieval_top_k,
    )
