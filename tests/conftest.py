"""Shared pytest fixtures for the OCR service test suite."""

from __future__ import annotations

import base64
import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.extract import ExtractDataTO, KeyValuePairTO, LineTO, PageTO

# Ensure required env vars are present before importing app modules that call get_settings()
os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://mock.cognitiveservices.azure.com")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com")


# ---------------------------------------------------------------------------
# Sample OCR data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_extract_data() -> ExtractDataTO:
    """Return a minimal ExtractDataTO for use in tests."""
    return ExtractDataTO(
        content="Sample document content",
        pages=[
            PageTO(
                page_number=1,
                lines=[LineTO(content="Sample document content", confidence=0.99)],
                key_value_pairs=[
                    KeyValuePairTO(key="Invoice", value="INV-001", confidence=0.95)
                ],
            )
        ],
        detected_languages=["en"],
    )


@pytest.fixture()
def sample_blob_base64() -> str:
    """Return a base64-encoded minimal PDF-like byte string."""
    return base64.b64encode(b"%PDF-1.4 fake pdf content").decode()


# ---------------------------------------------------------------------------
# App fixture with mocked OcrEngine (bypasses real lifespan startup)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ocr_engine(sample_extract_data: ExtractDataTO) -> MagicMock:
    """Return a mock OcrEngine that returns sample_extract_data."""
    engine = MagicMock()

    async def _analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        return sample_extract_data

    engine.analyze = _analyze
    return engine


@pytest.fixture()
def app_with_mock_engine(mock_ocr_engine: MagicMock):
    """Return the FastAPI app configured with a test lifespan using mock engine."""
    from app.core.config import get_settings

    get_settings.cache_clear()

    # Import the app module but rebuild the app with a test-friendly lifespan
    # that uses the mock engine instead of a real OcrEngine.
    import fastapi
    from app import main as main_module

    original_lifespan = main_module.app.router.lifespan_context

    @asynccontextmanager
    async def _test_lifespan(app: fastapi.FastAPI):
        app.state.ocr_engine = mock_ocr_engine
        yield

    main_module.app.router.lifespan_context = _test_lifespan
    yield main_module.app
    # Restore original lifespan after test
    main_module.app.router.lifespan_context = original_lifespan


@pytest.fixture()
def client(app_with_mock_engine) -> Generator[TestClient, None, None]:
    """Return a synchronous TestClient for the FastAPI app."""
    with TestClient(app_with_mock_engine, raise_server_exceptions=True) as c:
        yield c
