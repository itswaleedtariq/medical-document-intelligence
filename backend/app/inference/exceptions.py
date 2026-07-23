class InferenceError(Exception):
    """Base exception for model inference failures."""


class ModelProviderError(InferenceError):
    """Raised when the configured model provider fails."""


class ModelOutputError(InferenceError):
    """Raised when model output is invalid or unsafe."""
