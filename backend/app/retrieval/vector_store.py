from pathlib import Path
from typing import Any

import chromadb

from backend.app.core.config import get_settings
from backend.app.retrieval.exceptions import VectorStoreError
from backend.app.schemas.document import (
    DocumentChunk,
    RetrievedChunk,
)


class ChromaVectorStore:
    def __init__(
        self,
        path: str,
        collection_name: str,
        client: Any | None = None,
    ) -> None:
        self.path = path
        self.collection_name = collection_name

        try:
            if client is None:
                Path(path).mkdir(
                    parents=True,
                    exist_ok=True,
                )

                self.client = chromadb.PersistentClient(
                    path=path,
                )
            else:
                self.client = client

            self.collection = self._get_collection()

        except Exception as error:
            raise VectorStoreError("Could not initialize ChromaDB.") from error

    def _get_collection(self) -> Any:
        """
        Create a cosine-distance collection.

        The fallback supports older Chroma installations that used
        metadata rather than the newer configuration argument.
        """
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                configuration={
                    "hnsw": {
                        "space": "cosine",
                    }
                },
            )

        except TypeError:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={
                    "hnsw:space": "cosine",
                },
            )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise VectorStoreError("The number of chunks and embeddings must match.")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            metadatas.append(
                {
                    "document_id": str(chunk.document_id),
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "start_character": chunk.start_character,
                    "end_character": chunk.end_character,
                }
            )

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        except Exception as error:
            raise VectorStoreError(
                "Could not save document chunks to ChromaDB."
            ) from error

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        document_id: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not query_embedding:
            raise VectorStoreError("Query embedding cannot be empty.")

        try:
            query_result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={
                    "document_id": document_id,
                },
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )
        except Exception as error:
            raise VectorStoreError("ChromaDB search failed.") from error

        ids_groups = query_result.get("ids") or []
        document_groups = query_result.get("documents") or []
        metadata_groups = query_result.get("metadatas") or []
        distance_groups = query_result.get("distances") or []

        if not ids_groups:
            return []

        ids = ids_groups[0] or []
        documents = document_groups[0] if document_groups else []
        metadatas = metadata_groups[0] if metadata_groups else []
        distances = distance_groups[0] if distance_groups else []

        retrieved_chunks: list[RetrievedChunk] = []

        for index, chunk_id in enumerate(ids):
            document_text = documents[index] if index < len(documents) else ""

            metadata = (metadatas[index] if index < len(metadatas) else {}) or {}

            distance = float(distances[index] if index < len(distances) else 1.0)

            # For cosine distance:
            # similarity = 1 - distance.
            similarity_score = max(
                0.0,
                min(1.0, 1.0 - distance),
            )

            filename = str(
                metadata.get(
                    "filename",
                    "unknown.pdf",
                )
            )

            page_number = int(
                metadata.get(
                    "page_number",
                    1,
                )
            )

            chunk_index = int(
                metadata.get(
                    "chunk_index",
                    1,
                )
            )

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    filename=filename,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=document_text,
                    distance=round(distance, 6),
                    similarity_score=round(
                        similarity_score,
                        6,
                    ),
                    citation=(f"{filename}, page {page_number}"),
                )
            )

        return retrieved_chunks

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        try:
            self.collection.delete(
                where={
                    "document_id": document_id,
                }
            )
        except Exception as error:
            raise VectorStoreError("Could not delete document vectors.") from error


def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()

    return ChromaVectorStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )
