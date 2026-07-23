class RetrievalError(Exception):
    """Base exception for document retrieval failures."""


class NoExtractableTextError(RetrievalError):
    """Raised when a document has no text suitable for indexing."""


class EmbeddingError(RetrievalError):
    """Raised when text embedding generation fails."""


class VectorStoreError(RetrievalError):
    """Raised when Chroma storage or retrieval fails."""
