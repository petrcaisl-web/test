"""Tests for the payment extraction postprocessor."""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://mock.cognitiveservices.azure.com")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com")

from app.models.common import FieldWithConfidence
from app.models.extract import ExtractDataTO, KeyValuePairTO, LineTO, PageTO
from app.models.payment import PaymentDataTO
from app.postprocessors.payment_extraction import (
    PaymentExtractionPostProcessor,
    _validate_iban_checksum,
    _validate_variable_symbol,
    _validate_czech_account,
)

# ---------------------------------------------------------------------------
# Sample Czech invoice text (from the spec)
# ---------------------------------------------------------------------------

CZECH_INVOICE_TEXT = (
    "Faktura č. 2024-0892\n"
    "Dodavatel: ACME s.r.o.\n"
    "IČO: 12345678\n"
    "Částka k úhradě: 12 500,00 Kč\n"
    "Variabilní symbol: 20240892\n"
    "Číslo účtu: 19-2000014500/0800\n"
    "IBAN: CZ6508000000192000145399\n"
    "Datum splatnosti: 15.04.2025"
)

SAMPLE_PAYMENT_DATA = PaymentDataTO(
    account_number=FieldWithConfidence(value="19-2000014500/0800", confidence=0.97),
    iban=FieldWithConfidence(value="CZ6508000000192000145399", confidence=0.96),
    amount=FieldWithConfidence(value=12500.0, confidence=0.98),
    currency=FieldWithConfidence(value="CZK", confidence=0.99),
    variable_symbol=FieldWithConfidence(value="20240892", confidence=0.95),
    due_date=FieldWithConfidence(value="2025-04-15", confidence=0.90),
    payee_name=FieldWithConfidence(value="ACME s.r.o.", confidence=0.95),
)


def _make_sample_ocr(text: str = CZECH_INVOICE_TEXT) -> ExtractDataTO:
    """Build a minimal ExtractDataTO from text."""
    return ExtractDataTO(
        content=text,
        pages=[
            PageTO(
                page_number=1,
                lines=[LineTO(content=text, confidence=0.99)],
                key_value_pairs=[],
            )
        ],
        detected_languages=["cs"],
    )


def _make_mock_llm(return_data: PaymentDataTO) -> MagicMock:
    """Build a mock LlmService that returns return_data."""
    mock_llm = MagicMock()

    async def _complete(system_prompt, user_content, response_format):
        return return_data

    mock_llm.complete = _complete
    return mock_llm


# ---------------------------------------------------------------------------
# Unit tests for postprocessor process()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payment_extraction_returns_payment_data_to() -> None:
    """process() returns a PaymentDataTO with expected fields."""
    mock_llm = _make_mock_llm(SAMPLE_PAYMENT_DATA)
    pp = PaymentExtractionPostProcessor(llm_service=mock_llm)
    ocr = _make_sample_ocr()
    result = await pp.process(ocr)

    assert isinstance(result, PaymentDataTO)
    assert result.account_number.value == "19-2000014500/0800"
    assert result.account_number.confidence > 0.0
    assert result.variable_symbol.value == "20240892"
    assert result.variable_symbol.confidence > 0.0


@pytest.mark.asyncio
async def test_payment_extraction_valid_iban_keeps_confidence() -> None:
    """Valid IBAN checksum → confidence unchanged."""
    data = SAMPLE_PAYMENT_DATA.model_copy()
    mock_llm = _make_mock_llm(data)
    pp = PaymentExtractionPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_sample_ocr())
    assert result.iban.confidence > 0.0


@pytest.mark.asyncio
async def test_payment_extraction_invalid_iban_zeroes_confidence() -> None:
    """Invalid IBAN checksum → confidence set to 0.0."""
    bad_data = PaymentDataTO(
        account_number=FieldWithConfidence(value="19-2000014500/0800", confidence=0.97),
        iban=FieldWithConfidence(value="CZ9999000000000000000000", confidence=0.95),
        amount=FieldWithConfidence(value=None, confidence=0.0),
        currency=FieldWithConfidence(value=None, confidence=0.0),
        variable_symbol=FieldWithConfidence(value=None, confidence=0.0),
        due_date=FieldWithConfidence(value=None, confidence=0.0),
        payee_name=FieldWithConfidence(value=None, confidence=0.0),
    )
    mock_llm = _make_mock_llm(bad_data)
    pp = PaymentExtractionPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_sample_ocr())
    assert result.iban.confidence == 0.0


@pytest.mark.asyncio
async def test_payment_extraction_invalid_variable_symbol_zeroes_confidence() -> None:
    """Non-numeric variable symbol → confidence set to 0.0."""
    bad_data = SAMPLE_PAYMENT_DATA.model_copy(
        update={"variable_symbol": FieldWithConfidence(value="ABC123", confidence=0.8)}
    )
    mock_llm = _make_mock_llm(bad_data)
    pp = PaymentExtractionPostProcessor(llm_service=mock_llm)
    result = await pp.process(_make_sample_ocr())
    assert result.variable_symbol.confidence == 0.0


# ---------------------------------------------------------------------------
# Validation helper unit tests
# ---------------------------------------------------------------------------


def test_iban_checksum_valid() -> None:
    assert _validate_iban_checksum("CZ6508000000192000145399") is True


def test_iban_checksum_invalid() -> None:
    assert _validate_iban_checksum("CZ9999000000000000000000") is False


def test_iban_checksum_wrong_format() -> None:
    assert _validate_iban_checksum("DE89370400440532013000") is False


def test_variable_symbol_valid() -> None:
    assert _validate_variable_symbol("20240892") is True
    assert _validate_variable_symbol("1234567890") is True  # max 10 digits
    assert _validate_variable_symbol(None) is True


def test_variable_symbol_invalid() -> None:
    assert _validate_variable_symbol("ABC123") is False
    assert _validate_variable_symbol("12345678901") is False  # 11 digits — too long


def test_czech_account_valid() -> None:
    assert _validate_czech_account("19-2000014500/0800") is True
    assert _validate_czech_account("2000014500/0800") is True  # no prefix


def test_czech_account_invalid() -> None:
    assert _validate_czech_account("NOTANACCOUNT") is False


# ---------------------------------------------------------------------------
# Integration test via route (auto-generated POST /ocr/payment-extraction)
# ---------------------------------------------------------------------------


def _make_app_with_payment_pp(
    mock_ocr_engine: MagicMock,
    mock_llm: MagicMock,
) -> FastAPI:
    from app.api.route_factory import register_postprocessor_routes
    from app.postprocessors.registry import PostProcessorRegistry

    test_app = FastAPI()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield

    test_app.router.lifespan_context = _lifespan
    pp = PaymentExtractionPostProcessor(llm_service=mock_llm)
    registry = PostProcessorRegistry([pp])
    register_postprocessor_routes(test_app, registry, mock_ocr_engine, mock_llm)
    return test_app


def test_payment_extraction_route_returns_200() -> None:
    """POST /ocr/payment-extraction returns 200 with typed PaymentDataTO."""
    sample_ocr = _make_sample_ocr()

    engine = MagicMock()

    async def _analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        return sample_ocr

    engine.analyze = _analyze

    mock_llm = _make_mock_llm(SAMPLE_PAYMENT_DATA)
    app = _make_app_with_payment_pp(engine, mock_llm)

    blob_b64 = base64.b64encode(b"%PDF fake").decode()
    payload = {"blob_base64": blob_b64, "ocr_model": "prebuilt-invoice"}

    with TestClient(app) as c:
        response = c.post("/ocr/payment-extraction", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["postprocessor"]["name"] == "payment-extraction"
    assert body["data"]["iban"]["value"] == "CZ6508000000192000145399"
    assert body["data"]["variable_symbol"]["value"] == "20240892"
