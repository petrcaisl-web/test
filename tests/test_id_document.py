"""Tests for the ID document extraction postprocessor."""

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

from app.models.common import FieldWithConfidence
from app.models.extract import ExtractDataTO, LineTO, PageTO
from app.models.id_document import IdDocumentDataTO
from app.postprocessors.id_document import (
    IdDocumentPostProcessor,
    _is_valid_iso_date,
    _is_valid_nationality,
)

# ---------------------------------------------------------------------------
# Sample passport OCR text
# ---------------------------------------------------------------------------

PASSPORT_OCR = (
    "ČESKÁ REPUBLIKA\n"
    "PASSPORT / CESTOVNÍ PAS\n"
    "Příjmení: NOVÁK\n"
    "Jméno: JAN\n"
    "Státní příslušnost: CZE\n"
    "Datum narození: 15 MAR 1985\n"
    "Číslo pasu: AB1234567\n"
    "Platnost do: 20 JAN 2033\n"
    "P<CZENOTAK<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    "AB12345670CZE8503151M3301209<<<<<<<<<<<<6\n"
)

NATIONAL_ID_OCR = (
    "OBČANSKÝ PRŮKAZ\n"
    "Příjmení: SVOBODOVÁ\n"
    "Jméno: MARIE\n"
    "Datum narození: 10.06.1990\n"
    "Číslo dokladu: 123456789\n"
    "Platnost: 10.06.2030\n"
)

SAMPLE_PASSPORT_DATA = IdDocumentDataTO(
    first_name=FieldWithConfidence(value="JAN", confidence=0.98),
    last_name=FieldWithConfidence(value="NOVÁK", confidence=0.98),
    date_of_birth=FieldWithConfidence(value="1985-03-15", confidence=0.96),
    document_number=FieldWithConfidence(value="AB1234567", confidence=0.99),
    document_type=FieldWithConfidence(value="passport", confidence=0.99),
    expiry_date=FieldWithConfidence(value="2033-01-20", confidence=0.95),
    nationality=FieldWithConfidence(value="CZE", confidence=0.99),
)

SAMPLE_NATIONAL_ID_DATA = IdDocumentDataTO(
    first_name=FieldWithConfidence(value="MARIE", confidence=0.97),
    last_name=FieldWithConfidence(value="SVOBODOVÁ", confidence=0.97),
    date_of_birth=FieldWithConfidence(value="1990-06-10", confidence=0.95),
    document_number=FieldWithConfidence(value="123456789", confidence=0.99),
    document_type=FieldWithConfidence(value="national_id", confidence=0.99),
    expiry_date=FieldWithConfidence(value="2030-06-10", confidence=0.94),
    nationality=FieldWithConfidence(value="CZE", confidence=0.98),
)


def _make_ocr(text: str) -> ExtractDataTO:
    return ExtractDataTO(
        content=text,
        pages=[PageTO(page_number=1, lines=[LineTO(content=text, confidence=0.99)], key_value_pairs=[])],
        detected_languages=["cs"],
    )


def _make_mock_llm(return_data: IdDocumentDataTO) -> MagicMock:
    mock_llm = MagicMock()

    async def _complete(system_prompt, user_content, response_format):
        return return_data

    mock_llm.complete = _complete
    return mock_llm


# ---------------------------------------------------------------------------
# Unit tests for postprocessor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passport_extraction_returns_id_document_data_to() -> None:
    """process() returns IdDocumentDataTO from passport OCR."""
    mock_llm = _make_mock_llm(SAMPLE_PASSPORT_DATA)
    pp = IdDocumentPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(PASSPORT_OCR))

    assert isinstance(result, IdDocumentDataTO)
    assert result.first_name.value == "JAN"
    assert result.last_name.value == "NOVÁK"
    assert result.document_type.value == "passport"
    assert result.nationality.value == "CZE"


@pytest.mark.asyncio
async def test_national_id_extraction() -> None:
    """process() handles national ID OCR."""
    mock_llm = _make_mock_llm(SAMPLE_NATIONAL_ID_DATA)
    pp = IdDocumentPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(NATIONAL_ID_OCR))

    assert result.document_type.value == "national_id"
    assert result.first_name.value == "MARIE"


@pytest.mark.asyncio
async def test_invalid_date_zeroes_confidence() -> None:
    """Invalid date format in date_of_birth → confidence zeroed."""
    bad_data = SAMPLE_PASSPORT_DATA.model_copy(
        update={"date_of_birth": FieldWithConfidence(value="15-03-1985", confidence=0.9)}
    )
    mock_llm = _make_mock_llm(bad_data)
    pp = IdDocumentPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(PASSPORT_OCR))
    assert result.date_of_birth.confidence == 0.0


@pytest.mark.asyncio
async def test_invalid_nationality_zeroes_confidence() -> None:
    """Invalid nationality code → confidence zeroed."""
    bad_data = SAMPLE_PASSPORT_DATA.model_copy(
        update={"nationality": FieldWithConfidence(value="Czech", confidence=0.8)}
    )
    mock_llm = _make_mock_llm(bad_data)
    pp = IdDocumentPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(PASSPORT_OCR))
    assert result.nationality.confidence == 0.0


@pytest.mark.asyncio
async def test_valid_dates_keep_confidence() -> None:
    """Valid ISO dates keep their original confidence scores."""
    mock_llm = _make_mock_llm(SAMPLE_PASSPORT_DATA)
    pp = IdDocumentPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_ocr(PASSPORT_OCR))
    assert result.date_of_birth.confidence > 0.0
    assert result.expiry_date.confidence > 0.0


# ---------------------------------------------------------------------------
# Validation helper tests
# ---------------------------------------------------------------------------


def test_is_valid_iso_date_valid() -> None:
    assert _is_valid_iso_date("1985-03-15") is True
    assert _is_valid_iso_date("2033-01-20") is True
    assert _is_valid_iso_date(None) is True


def test_is_valid_iso_date_invalid() -> None:
    assert _is_valid_iso_date("15-03-1985") is False  # DD-MM-YYYY
    assert _is_valid_iso_date("15/03/1985") is False
    assert _is_valid_iso_date("1985-13-01") is False  # invalid month
    assert _is_valid_iso_date("not-a-date") is False


def test_is_valid_nationality_valid() -> None:
    assert _is_valid_nationality("CZE") is True
    assert _is_valid_nationality("SVK") is True
    assert _is_valid_nationality("DEU") is True
    assert _is_valid_nationality(None) is True


def test_is_valid_nationality_invalid() -> None:
    assert _is_valid_nationality("Czech") is False
    assert _is_valid_nationality("CZ") is False  # too short
    assert _is_valid_nationality("czE") is False  # lowercase


# ---------------------------------------------------------------------------
# Integration test via auto-generated route
# ---------------------------------------------------------------------------


def test_id_document_route_returns_200() -> None:
    """POST /ocr/id-document-extract returns 200 with typed IdDocumentDataTO."""
    from app.api.route_factory import register_postprocessor_routes
    from app.postprocessors.registry import PostProcessorRegistry

    sample_ocr = _make_ocr(PASSPORT_OCR)
    engine = MagicMock()

    async def _analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        return sample_ocr

    engine.analyze = _analyze

    mock_llm = _make_mock_llm(SAMPLE_PASSPORT_DATA)
    test_app = FastAPI()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield

    test_app.router.lifespan_context = _lifespan
    pp = IdDocumentPostProcessor(llm_service=mock_llm)
    registry = PostProcessorRegistry([pp])
    register_postprocessor_routes(test_app, registry, engine, mock_llm)

    blob_b64 = base64.b64encode(b"%PDF fake").decode()
    payload = {"blob_base64": blob_b64, "ocr_model": "prebuilt-idDocument"}

    with TestClient(test_app) as c:
        response = c.post("/ocr/id-document-extract", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["postprocessor"]["name"] == "id-document-extract"
    assert body["data"]["first_name"]["value"] == "JAN"
    assert body["data"]["nationality"]["value"] == "CZE"
