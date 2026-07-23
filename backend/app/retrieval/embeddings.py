from functools import lru_cache
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.core.config import get_settings
from backend.app.retrieval.exceptions import EmbeddingError


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """
        Load the model only when the first embedding is requested.
        """
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            except Exception as error:
                raise EmbeddingError(
                    f"Could not load embedding model: {self.model_name}"
                ) from error

        return self._model

    def _encode(
        self,
        texts: list[str],
        task: Literal["document", "query"],
    ) -> list[list[float]]:
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise EmbeddingError("Empty text cannot be embedded.")

        model = self._get_model()

        try:
            if task == "document" and hasattr(
                model,
                "encode_document",
            ):
                embeddings = model.encode_document(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

            elif task == "query" and hasattr(
                model,
                "encode_query",
            ):
                embeddings = model.encode_query(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

            else:
                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

        except Exception as error:
            raise EmbeddingError("Embedding generation failed.") from error

        embedding_array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        return embedding_array.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._encode(
            texts=texts,
            task="document",
        )

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        if not query.strip():
            raise EmbeddingError("Search query cannot be empty.")

        embeddings = self._encode(
            texts=[query],
            task="query",
        )

        return embeddings[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()

    return EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
