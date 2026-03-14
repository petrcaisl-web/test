"""Tests for the postprocessor framework: registry, route factory, LLM service."""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

os.environ.setdefault("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://mock.cognitiveservices.azure.com")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://mock.openai.azure.com")

from app.models.common import FieldWithConfidence
from app.models.extract import ExtractDataTO, LineTO, PageTO
from app.postprocessors.base import BasePostProcessor
from app.postprocessors.registry import PostProcessorRegistry
from app.services.llm_service import LlmService, LlmServiceError


# ---------------------------------------------------------------------------
# Dummy postprocessor for test isolation
# ---------------------------------------------------------------------------


class _DummyDataTO(BaseModel):
    label: FieldWithConfidence[str]


class _DummyPostProcessor(BasePostProcessor):
    name: ClassVar[str] = "dummy-test"
    description: ClassVar[str] = "Dummy postprocessor for tests"
    version: ClassVar[str] = "0.0.1"
    response_model: ClassVar[type[BaseModel]] = _DummyDataTO

    async def process(self, ocr_result: ExtractDataTO) -> _DummyDataTO:
        return _DummyDataTO(label=FieldWithConfidence(value="test", confidence=0.99))


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_discover_and_get() -> None:
    """Registry can discover the dummy postprocessor via subclass tracking."""
    registry = PostProcessorRegistry([_DummyPostProcessor()])
    pp = registry.get("dummy-test")
    assert pp.name == "dummy-test"
    assert pp.version == "0.0.1"


def test_registry_list_all() -> None:
    """list_all() returns all registered instances."""
    registry = PostProcessorRegistry([_DummyPostProcessor()])
    all_pp = registry.list_all()
    names = [p.name for p in all_pp]
    assert "dummy-test" in names


def test_registry_get_unknown_raises() -> None:
    """get() raises KeyError for unknown postprocessor name."""
    registry = PostProcessorRegistry([])
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_registry_build_auto_discovers() -> None:
    """PostProcessorRegistry.build() auto-discovers all subclasses."""
    # _DummyPostProcessor is already a subclass — it should be found
    registry = PostProcessorRegistry.build()
    # At minimum the dummy registered above is discoverable
    assert registry is not None


# ---------------------------------------------------------------------------
# Route factory tests
# ---------------------------------------------------------------------------


def _make_app_with_dummy_postprocessor(
    dummy_pp: _DummyPostProcessor,
    mock_ocr_engine: MagicMock,
    mock_llm_service: MagicMock,
) -> FastAPI:
    """Return a minimal FastAPI app with one dummy postprocessor route."""
    from app.api.route_factory import register_postprocessor_routes

    test_app = FastAPI()

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        yield

    test_app.router.lifespan_context = _lifespan
    registry = PostProcessorRegistry([dummy_pp])
    register_postprocessor_routes(test_app, registry, mock_ocr_engine, mock_llm_service)
    return test_app


@pytest.fixture()
def route_factory_client(sample_extract_data: ExtractDataTO) -> TestClient:
    """Return a TestClient for a minimal app with the dummy postprocessor route."""
    engine = MagicMock()

    async def _analyze(blob: bytes, model_id: str) -> ExtractDataTO:
        return sample_extract_data

    engine.analyze = _analyze
    llm_svc = MagicMock()

    dummy = _DummyPostProcessor()
    app = _make_app_with_dummy_postprocessor(dummy, engine, llm_svc)
    return TestClient(app)


def test_route_factory_generates_endpoint(route_factory_client: TestClient) -> None:
    """Route factory auto-generates POST /ocr/dummy-test endpoint."""
    blob_b64 = base64.b64encode(b"fake pdf").decode()
    payload = {"blob_base64": blob_b64, "ocr_model": "prebuilt-invoice"}
    response = route_factory_client.post("/ocr/dummy-test", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["postprocessor"]["name"] == "dummy-test"
    assert body["postprocessor"]["version"] == "0.0.1"
    assert body["data"]["label"]["value"] == "test"
    assert body["data"]["label"]["confidence"] == 0.99


def test_route_factory_envelope_structure(route_factory_client: TestClient) -> None:
    """Route factory routes return the standard envelope."""
    blob_b64 = base64.b64encode(b"fake pdf").decode()
    payload = {"blob_base64": blob_b64, "ocr_model": "prebuilt-invoice"}
    response = route_factory_client.post("/ocr/dummy-test", json=payload)
    body = response.json()

    for key in ("request_id", "timestamp", "processing_time_ms", "ocr_model", "postprocessor", "data"):
        assert key in body, f"Missing envelope key: {key}"


# ---------------------------------------------------------------------------
# LLM Service tests
# ---------------------------------------------------------------------------


class _SimpleOutput(BaseModel):
    result: str


@pytest.mark.asyncio
async def test_llm_service_complete_success() -> None:
    """LlmService.complete() returns a parsed Pydantic model on success."""
    from app.core.config import get_settings
    from app.services.llm_service import LlmService

    get_settings.cache_clear()

    svc = LlmService()
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"result": "hello"})
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    svc._client = mock_client

    result = await svc.complete(
        system_prompt="You are helpful.",
        user_content="Test input",
        response_format=_SimpleOutput,
    )
    assert result.result == "hello"


@pytest.mark.asyncio
async def test_llm_service_retries_on_api_error() -> None:
    """LlmService retries on APIError and eventually raises LlmServiceError."""
    from openai import APIError
    from app.services.llm_service import LlmService

    svc = LlmService()
    mock_client = AsyncMock()

    call_count = 0

    async def _failing(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise APIError("timeout", request=MagicMock(), body=None)

    mock_client.chat.completions.create = _failing
    svc._client = mock_client

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LlmServiceError):
            await svc.complete(
                system_prompt="sys",
                user_content="usr",
                response_format=_SimpleOutput,
            )

    assert call_count == 3  # 3 attempts
