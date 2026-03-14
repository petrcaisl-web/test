"""Azure AI Document Intelligence wrapper — async OCR engine.

Wraps the Azure SDK `DocumentAnalysisClient` with retry logic and
maps SDK output to our internal `ExtractDataTO` model.

Azure credentials are NEVER stored in code — they are loaded from
environment variables via `Settings`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

import structlog
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError
from azure.identity import DefaultAzureCredential

from app.core.config import get_settings
from app.models.extract import ExtractDataTO, KeyValuePairTO, LineTO, PageTO
from app.models.request import OcrModel

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Supported Azure Document Intelligence prebuilt model IDs
SUPPORTED_MODELS: set[str] = {m.value for m in OcrModel}

# Retry configuration
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.5  # seconds


class OcrEngineError(Exception):
    """Base error raised when OCR analysis fails after retries."""


class UnsupportedModelError(OcrEngineError):
    """Raised when an unsupported OCR model ID is requested."""


class OcrEngine:
    """Async wrapper around Azure AI Document Intelligence.

    Instantiate once on app startup (singleton) and inject via FastAPI
    dependency injection.  The client is created lazily on first use.
    """

    def __init__(self) -> None:
        self._client: DocumentAnalysisClient | None = None

    def _get_client(self) -> DocumentAnalysisClient:
        """Return (or lazily create) the Azure Document Intelligence client."""
        if self._client is not None:
            return self._client

        settings = get_settings()
        endpoint = settings.azure_document_intelligence_endpoint

        if settings.azure_document_intelligence_key:
            credential = AzureKeyCredential(settings.azure_document_intelligence_key)
        else:
            credential = DefaultAzureCredential()

        self._client = DocumentAnalysisClient(endpoint=endpoint, credential=credential)
        logger.info("ocr_client_initialized", endpoint=endpoint)
        return self._client

    async def analyze(self, blob: bytes, model_id: str) -> ExtractDataTO:
        """Analyze a document blob and return structured OCR output.

        Args:
            blob: Raw document bytes (PDF or image).
            model_id: Azure Document Intelligence model ID (must be one of
                the supported prebuilt models defined in `OcrModel`).

        Returns:
            ExtractDataTO with full text, per-page breakdown, and detected
            languages.

        Raises:
            UnsupportedModelError: If *model_id* is not in SUPPORTED_MODELS.
            OcrEngineError: If the Azure API fails after all retry attempts.
        """
        if model_id not in SUPPORTED_MODELS:
            raise UnsupportedModelError(
                f"Model '{model_id}' is not supported. "
                f"Supported models: {sorted(SUPPORTED_MODELS)}"
            )

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                logger.debug(
                    "ocr_analyze_attempt",
                    model_id=model_id,
                    blob_size=len(blob),
                    attempt=attempt,
                )
                result = await asyncio.to_thread(self._run_analyze, blob, model_id)
                logger.debug("ocr_analyze_success", model_id=model_id, attempt=attempt)
                return result
            except (HttpResponseError, ServiceRequestError) as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "ocr_analyze_retry",
                        model_id=model_id,
                        attempt=attempt,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)
            except Exception as exc:
                # Non-retriable error — wrap and re-raise immediately
                raise OcrEngineError(
                    f"Unexpected error during OCR analysis: {exc}"
                ) from exc

        raise OcrEngineError(
            f"OCR analysis failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    def _run_analyze(self, blob: bytes, model_id: str) -> ExtractDataTO:
        """Synchronous inner call executed in a thread via asyncio.to_thread."""
        client = self._get_client()
        poller = client.begin_analyze_document(model_id, blob)
        result = poller.result()

        # Build per-page output
        pages: list[PageTO] = []
        for page in result.pages or []:
            lines = [
                LineTO(
                    content=line.content,
                    confidence=line.spans[0].offset if line.spans else 0.0,
                )
                for line in (page.lines or [])
            ]
            # Key-value pairs live on the top-level result, not per-page in SDK v3
            pages.append(
                PageTO(
                    page_number=page.page_number,
                    lines=lines,
                    key_value_pairs=[],
                )
            )

        # Top-level key-value pairs (assign to page 1 if pages exist)
        kv_pairs: list[KeyValuePairTO] = []
        for kv in result.key_value_pairs or []:
            key_content = kv.key.content if kv.key else ""
            value_content = kv.value.content if kv.value else ""
            confidence = kv.confidence or 0.0
            kv_pairs.append(
                KeyValuePairTO(key=key_content, value=value_content, confidence=confidence)
            )

        if pages and kv_pairs:
            pages[0] = PageTO(
                page_number=pages[0].page_number,
                lines=pages[0].lines,
                key_value_pairs=kv_pairs,
            )

        detected_languages = [
            lang.locale for lang in (result.languages or []) if lang.locale
        ]

        return ExtractDataTO(
            content=result.content or "",
            pages=pages,
            detected_languages=detected_languages,
        )
