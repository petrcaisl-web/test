"""Tests for the POST /ocr/extract endpoint."""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.ocr_engine import OcrEngineError, UnsupportedModelError
from app.models.extract import ExtractDataTO, PageTO, LineTO, KeyValuePairTO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request_payload(model: str = "prebuilt-invoice") -> dict:
    blob_b64 = base64.b64encode(b"%PDF-1.4 fake pdf content").decode()
    return {"blob_base64": blob_b64, "ocr_model": model}


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_extract_returns_envelope(client: TestClient) -> None:
    """POST /ocr/extract returns an ApiResponse envelope with ExtractDataTO."""
    payload = _make_request_payload()
    response = client.post("/ocr/extract", json=payload)

    assert response.status_code == 200
    body = response.json()

    # Envelope fields
    assert "request_id" in body
    assert "timestamp" in body
    assert "processing_time_ms" in body
    assert body["ocr_model"] == "prebuilt-invoice"
    assert body["postprocessor"] is None

    # data field maps to ExtractDataTO
    data = body["data"]
    assert data["content"] == "Sample document content"
    assert isinstance(data["pages"], list)
    assert len(data["pages"]) == 1
    assert data["detected_languages"] == ["en"]


def test_extract_processing_time_non_negative(client: TestClient) -> None:
    """Processing time in response envelope must be >= 0."""
    response = client.post("/ocr/extract", json=_make_request_payload())
    assert response.json()["processing_time_ms"] >= 0


def test_extract_receipt_model(app_with_mock_engine, sample_extract_data) -> None:
    """Receipt model is also accepted by the endpoint."""
    # Override mock to return sample data for receipt model
    async def _analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        return sample_extract_data

    app_with_mock_engine.state.ocr_engine.analyze = _analyze

    with TestClient(app_with_mock_engine) as client:
        payload = _make_request_payload(model="prebuilt-receipt")
        response = client.post("/ocr/extract", json=payload)
        assert response.status_code == 200
        assert response.json()["ocr_model"] == "prebuilt-receipt"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_extract_invalid_base64(client: TestClient) -> None:
    """Invalid base64 input returns 422."""
    payload = {"blob_base64": "NOT_VALID_BASE64!!!", "ocr_model": "prebuilt-invoice"}
    response = client.post("/ocr/extract", json=payload)
    assert response.status_code == 422


def test_extract_invalid_model_rejected_by_pydantic(client: TestClient) -> None:
    """An unsupported ocr_model value is rejected by Pydantic validation (422)."""
    payload = _make_request_payload()
    payload["ocr_model"] = "unsupported-model"
    response = client.post("/ocr/extract", json=payload)
    # Pydantic rejects unknown enum values before the route runs
    assert response.status_code == 422


def test_extract_azure_error_returns_502() -> None:
    """Azure SDK error surfaces as 502 Bad Gateway."""
    import os
    from contextlib import asynccontextmanager
    import fastapi
    from app import main as main_module
    from app.core.config import get_settings

    os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://mock.cognitiveservices.azure.com")
    os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com")
    get_settings.cache_clear()

    failing_engine = MagicMock()

    async def _failing_analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        raise OcrEngineError("Azure timeout")

    failing_engine.analyze = _failing_analyze
    original_lifespan = main_module.app.router.lifespan_context

    @asynccontextmanager
    async def _error_lifespan(app: fastapi.FastAPI):
        app.state.ocr_engine = failing_engine
        yield

    main_module.app.router.lifespan_context = _error_lifespan
    try:
        with TestClient(main_module.app, raise_server_exceptions=True) as c:
            response = c.post("/ocr/extract", json=_make_request_payload())
            assert response.status_code == 502
    finally:
        main_module.app.router.lifespan_context = original_lifespan


def test_extract_missing_blob_returns_422(client: TestClient) -> None:
    """Missing blob_base64 field returns 422."""
    response = client.post("/ocr/extract", json={"ocr_model": "prebuilt-invoice"})
    assert response.status_code == 422


def test_extract_missing_model_returns_422(client: TestClient) -> None:
    """Missing ocr_model field returns 422."""
    blob_b64 = base64.b64encode(b"data").decode()
    response = client.post("/ocr/extract", json={"blob_base64": blob_b64})
    assert response.status_code == 422
