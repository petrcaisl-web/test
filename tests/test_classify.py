"""Tests for the document classification postprocessor."""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://mock.cognitiveservices.azure.com")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com")

from app.models.classify import ClassificationDataTO, DocumentType
from app.models.common import FieldWithConfidence
from app.models.extract import ExtractDataTO, LineTO, PageTO
from app.postprocessors.document_classify import DocumentClassifyPostProcessor

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

INVOICE_TEXT = (
    "Faktura č. 2024-0892\n"
    "Dodavatel: ACME s.r.o.\n"
    "IČO: 12345678\n"
    "Částka k úhradě: 12 500,00 Kč\n"
)

LETTER_TEXT = (
    "Vážený zákazníku,\n"
    "dovolujeme si Vás informovat o změnách v podmínkách služby.\n"
    "S pozdravem, ABC Bank\n"
)


def _make_ocr(text: str) -> ExtractDataTO:
    return ExtractDataTO(
        content=text,
        pages=[PageTO(page_number=1, lines=[LineTO(content=text, confidence=0.99)], key_value_pairs=[])],
        detected_languages=["cs"],
    )


def _make_mock_llm(return_data: ClassificationDataTO) -> MagicMock:
    mock_llm = MagicMock()

    async def _complete(system_prompt, user_content, response_format):
        return return_data

    mock_llm.complete = _complete
    return mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_invoice_text() -> None:
    """process() classifies invoice text as 'faktura'."""
    expected = ClassificationDataTO(
        document_type=FieldWithConfidence(value=DocumentType.FAKTURA, confidence=0.97),
        summary="Toto je faktura od ACME s.r.o. na částku 12 500 Kč.",
        detected_language="cs",
    )
    mock_llm = _make_mock_llm(expected)
    pp = DocumentClassifyPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(INVOICE_TEXT))

    assert isinstance(result, ClassificationDataTO)
    assert result.document_type.value == DocumentType.FAKTURA
    assert result.document_type.confidence > 0.0
    assert result.detected_language == "cs"


@pytest.mark.asyncio
async def test_classify_letter_text() -> None:
    """process() classifies letter text as 'korespondence'."""
    expected = ClassificationDataTO(
        document_type=FieldWithConfidence(value=DocumentType.KORESPONDENCE, confidence=0.92),
        summary="Dopis od ABC Bank informující zákazníka o změnách.",
        detected_language="cs",
    )
    mock_llm = _make_mock_llm(expected)
    pp = DocumentClassifyPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(LETTER_TEXT))

    assert result.document_type.value == DocumentType.KORESPONDENCE


@pytest.mark.asyncio
async def test_classify_returns_classification_data_to() -> None:
    """process() returns ClassificationDataTO."""
    expected = ClassificationDataTO(
        document_type=FieldWithConfidence(value=DocumentType.OSTATNI, confidence=0.5),
        summary="Unknown document.",
        detected_language="en",
    )
    mock_llm = _make_mock_llm(expected)
    pp = DocumentClassifyPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr("Some text"))
    assert isinstance(result, ClassificationDataTO)


def test_classify_route_returns_200() -> None:
    """POST /ocr/document-classify returns 200 with typed ClassificationDataTO."""
    from app.api.route_factory import register_postprocessor_routes
    from app.postprocessors.registry import PostProcessorRegistry

    sample_ocr = _make_ocr(INVOICE_TEXT)
    engine = MagicMock()

    async def _analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        return sample_ocr

    engine.analyze = _analyze

    expected = ClassificationDataTO(
        document_type=FieldWithConfidence(value=DocumentType.FAKTURA, confidence=0.97),
        summary="Faktura od ACME.",
        detected_language="cs",
    )
    mock_llm = _make_mock_llm(expected)

    test_app = FastAPI()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield

    test_app.router.lifespan_context = _lifespan
    pp = DocumentClassifyPostProcessor(llm_service=mock_llm)
    registry = PostProcessorRegistry([pp])
    register_postprocessor_routes(test_app, registry, engine, mock_llm)

    blob_b64 = base64.b64encode(b"%PDF fake").decode()
    payload = {"blob_base64": blob_b64, "ocr_model": "prebuilt-layout"}

    with TestClient(test_app) as c:
        response = c.post("/ocr/document-classify", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["postprocessor"]["name"] == "document-classify"
    assert body["data"]["document_type"]["value"] == "faktura"
