"""Tests for Step 6 hardening: middleware, cost estimator, and envelope updates."""

from __future__ import annotations

import base64
import os

import pytest

os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://mock.cognitiveservices.azure.com")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com")


# ---------------------------------------------------------------------------
# Cost estimator tests
# ---------------------------------------------------------------------------


def test_cost_estimate_zero_pages() -> None:
    """Zero pages → zero cost."""
    from app.services.cost_estimator import estimate_from_pages

    cost = estimate_from_pages(0)
    assert cost.doc_intelligence_pages == 0
    assert cost.openai_tokens == 0
    assert cost.estimated_cost_usd == 0.0


def test_cost_estimate_one_page() -> None:
    """One page costs $0.001 for Document Intelligence."""
    from app.services.cost_estimator import estimate_from_pages

    cost = estimate_from_pages(1)
    assert cost.doc_intelligence_pages == 1
    assert abs(cost.estimated_cost_usd - 0.001) < 1e-6


def test_cost_estimate_with_openai_tokens() -> None:
    """Cost includes both Document Intelligence and OpenAI token costs."""
    from app.services.cost_estimator import estimate

    cost = estimate(num_pages=2, openai_prompt_tokens=1000, openai_completion_tokens=200)
    assert cost.doc_intelligence_pages == 2
    assert cost.openai_tokens == 1200
    assert cost.estimated_cost_usd > 0.002  # More than just page cost


def test_cost_estimate_from_text_length() -> None:
    """estimate_from_text_length() produces a non-zero estimate for non-empty content."""
    from app.services.cost_estimator import estimate_from_text_length

    cost = estimate_from_text_length(num_pages=1, content_length=500)
    assert cost.openai_tokens > 0
    assert cost.estimated_cost_usd > 0.001


# ---------------------------------------------------------------------------
# Middleware tests — via the extract endpoint
# ---------------------------------------------------------------------------


def test_request_id_header_present(client) -> None:
    """Responses include X-Request-ID header."""
    payload = {"blob_base64": base64.b64encode(b"fake").decode(), "ocr_model": "prebuilt-invoice"}
    response = client.post("/ocr/extract", json=payload)
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_process_time_header_present(client) -> None:
    """Responses include X-Process-Time-Ms header."""
    payload = {"blob_base64": base64.b64encode(b"fake").decode(), "ocr_model": "prebuilt-invoice"}
    response = client.post("/ocr/extract", json=payload)
    assert "X-Process-Time-Ms" in response.headers
    assert int(response.headers["X-Process-Time-Ms"]) >= 0


def test_request_id_forwarded_when_provided(client) -> None:
    """If client sends X-Request-ID, the same value is echoed back."""
    custom_id = "my-custom-request-id-123"
    payload = {"blob_base64": base64.b64encode(b"fake").decode(), "ocr_model": "prebuilt-invoice"}
    response = client.post(
        "/ocr/extract",
        json=payload,
        headers={"X-Request-ID": custom_id},
    )
    assert response.headers["X-Request-ID"] == custom_id


def test_cost_estimate_in_extract_response(client) -> None:
    """Extract endpoint includes cost_estimate in the envelope."""
    payload = {"blob_base64": base64.b64encode(b"fake").decode(), "ocr_model": "prebuilt-invoice"}
    response = client.post("/ocr/extract", json=payload)
    body = response.json()
    assert "cost_estimate" in body
    cost = body["cost_estimate"]
    assert "doc_intelligence_pages" in cost
    assert "openai_tokens" in cost
    assert "estimated_cost_usd" in cost


def test_health_endpoint_includes_cost_estimate(client) -> None:
    """Health endpoint includes cost_estimate (zeroed for non-OCR calls)."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "cost_estimate" in body
    assert body["cost_estimate"]["estimated_cost_usd"] == 0.0
