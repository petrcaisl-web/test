"""Azure OpenAI wrapper providing structured LLM completions.

Credentials are NEVER stored in code — loaded from environment via
Settings.  Retry logic with exponential backoff is built in.
"""

from __future__ import annotations

import asyncio
import json
from typing import TypeVar

import structlog
from openai import AsyncAzureOpenAI, APIError, APITimeoutError
from pydantic import BaseModel

from app.core.config import get_settings

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Retry configuration
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.5  # seconds


class LlmServiceError(Exception):
    """Raised when the LLM call fails after all retry attempts."""


class LlmService:
    """Async wrapper around Azure OpenAI for structured JSON completions.

    Instantiate once on app startup (singleton) and inject via FastAPI
    dependency injection.
    """

    def __init__(self) -> None:
        self._client: AsyncAzureOpenAI | None = None

    def _get_client(self) -> AsyncAzureOpenAI:
        """Return (or lazily create) the Azure OpenAI async client."""
        if self._client is not None:
            return self._client

        settings = get_settings()
        self._client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key or None,  # None → uses env/identity
            api_version=settings.azure_openai_api_version,
        )
        logger.info("llm_client_initialized", endpoint=settings.azure_openai_endpoint)
        return self._client

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        response_format: type[T],
    ) -> T:
        """Request a structured completion from Azure OpenAI.

        Uses the ``response_format`` Pydantic model to instruct the LLM
        to return valid JSON matching that schema (JSON mode).

        Args:
            system_prompt: Instructions for the LLM.
            user_content: The document text or data to process.
            response_format: Pydantic model class defining the expected
                JSON schema.  The method deserialises the LLM output
                into an instance of this class.

        Returns:
            A validated instance of ``response_format``.

        Raises:
            LlmServiceError: If all retry attempts fail.
        """
        settings = get_settings()
        client = self._get_client()
        schema = response_format.model_json_schema()

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                logger.debug(
                    "llm_complete_attempt",
                    model=settings.azure_openai_deployment,
                    attempt=attempt,
                )
                response = await client.chat.completions.create(
                    model=settings.azure_openai_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_json = response.choices[0].message.content or "{}"
                result = response_format.model_validate_json(raw_json)
                logger.debug(
                    "llm_complete_success",
                    model=settings.azure_openai_deployment,
                    attempt=attempt,
                )
                return result
            except (APIError, APITimeoutError) as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "llm_complete_retry",
                        attempt=attempt,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)
            except Exception as exc:
                raise LlmServiceError(
                    f"Unexpected error during LLM completion: {exc}"
                ) from exc

        raise LlmServiceError(
            f"LLM completion failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc
