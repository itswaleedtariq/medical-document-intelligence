import json
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from backend.app.core.config import get_settings
from backend.app.inference.exceptions import ModelProviderError


class ModelProvider(ABC):
    name: str
    model_id: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Generate one structured response."""


class MockModelProvider(ModelProvider):
    """
    Development-only provider.

    It verifies the complete backend flow without loading MedGemma.
    It does not produce a real medical-document answer.
    """

    name = "mock"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        del prompt
        del max_new_tokens

        return json.dumps(
            {
                "status": "answered",
                "answer": (
                    "The development model provider is active. "
                    "Configure MODEL_PROVIDER=http to receive a "
                    "real MedGemma-generated answer."
                ),
                "facts": [
                    {
                        "statement": (
                            "Document evidence was retrieved and "
                            "provided to the inference layer."
                        ),
                        "citations": ["E1"],
                    }
                ],
                "safety_note": (
                    "This is a development response, not medical analysis."
                ),
                "limitations": ["The mock provider does not run MedGemma."],
            }
        )


class HTTPModelProvider(ModelProvider):
    name = "http"

    def __init__(
        self,
        model_id: str,
        api_url: str,
        api_token: str | None,
        timeout_seconds: float,
    ) -> None:
        self.model_id = model_id
        self.api_url = api_url
        self.api_token = api_token
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
            ) as client:
                response = client.post(
                    self.api_url,
                    headers=headers,
                    json={
                        "prompt": prompt,
                        "max_new_tokens": max_new_tokens,
                    },
                )

                response.raise_for_status()
                payload = response.json()

        except (
            httpx.HTTPError,
            ValueError,
        ) as error:
            raise ModelProviderError(
                "The remote MedGemma service could not be reached."
            ) from error

        generated_text = payload.get("generated_text")

        if not isinstance(generated_text, str):
            raise ModelProviderError(
                "The remote model response did not contain 'generated_text'."
            )

        if not generated_text.strip():
            raise ModelProviderError("The remote model returned an empty response.")

        return generated_text.strip()


@lru_cache(maxsize=1)
def get_model_provider() -> ModelProvider:
    settings = get_settings()

    if settings.model_provider == "mock":
        return MockModelProvider(
            model_id=settings.model_id,
        )

    if settings.model_provider == "http":
        if not settings.model_api_url:
            raise ModelProviderError("MODEL_API_URL is not configured.")

        return HTTPModelProvider(
            model_id=settings.model_id,
            api_url=settings.model_api_url,
            api_token=settings.model_api_token,
            timeout_seconds=(settings.model_request_timeout_seconds),
        )

    raise ModelProviderError(f"Unsupported model provider: {settings.model_provider}")
