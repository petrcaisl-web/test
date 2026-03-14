"""Auto-generate FastAPI routes for every registered postprocessor.

For each postprocessor in the registry, a ``POST /ocr/{name}`` route is
registered that:
1. Decodes the request blob.
2. Calls ``ocr_engine.analyze()`` to get raw ExtractDataTO.
3. Calls ``postprocessor.process()`` to get the typed DataTO.
4. Returns the result wrapped in ``ApiResponse[DataTO]``.

The FastAPI ``response_model`` on each generated route is set to the
concrete ``ApiResponse[DataTO]`` generic alias so OpenAPI shows the
full typed schema per endpoint.
"""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status

from app.core.ocr_engine import OcrEngine, OcrEngineError, UnsupportedModelError
from app.models.common import PostProcessorInfo
from app.models.envelope import ApiResponse
from app.models.request import OcrRequest
from app.postprocessors.base import BasePostProcessor
from app.postprocessors.registry import PostProcessorRegistry
from app.services.llm_service import LlmService, LlmServiceError

logger = structlog.get_logger(__name__)


def register_postprocessor_routes(
    app: FastAPI,
    registry: PostProcessorRegistry,
    ocr_engine: OcrEngine,
    llm_service: LlmService,
) -> None:
    """Register one POST route per postprocessor found in the registry.

    Args:
        app: The FastAPI application instance.
        registry: A fully-built PostProcessorRegistry.
        ocr_engine: The OcrEngine singleton.
        llm_service: The LlmService singleton.
    """
    for postprocessor in registry.list_all():
        _register_route(app, postprocessor, ocr_engine, llm_service)
        logger.info(
            "route_registered",
            path=f"/ocr/{postprocessor.name}",
            postprocessor=postprocessor.name,
        )


def _register_route(
    app: FastAPI,
    postprocessor: BasePostProcessor,
    ocr_engine: OcrEngine,
    llm_service: LlmService,
) -> None:
    """Register a single route for the given postprocessor.

    This inner function is needed to capture ``postprocessor`` in the
    closure correctly (avoids the classic loop-variable capture bug).
    """
    # Capture everything we need in the closure
    _name = postprocessor.name
    _version = postprocessor.version
    _description = postprocessor.description
    _response_model = postprocessor.response_model

    # Build the concrete response model type for FastAPI's OpenAPI schema
    concrete_response: Any = ApiResponse[_response_model]  # type: ignore[valid-type]

    # Determine tags based on postprocessor name
    _tags = _tags_for(postprocessor.name)

    @app.post(
        f"/ocr/{_name}",
        response_model=concrete_response,
        summary=_description,
        tags=_tags,
    )
    async def _route_handler(request: OcrRequest) -> Any:
        """Auto-generated route for postprocessor {name}."""
        start = time.perf_counter()

        try:
            blob = base64.b64decode(request.blob_base64)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid base64 encoding: {exc}",
            ) from exc

        try:
            ocr_result = await ocr_engine.analyze(
                blob=blob, model_id=request.ocr_model
            )
        except UnsupportedModelError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except OcrEngineError as exc:
            logger.error(
                "ocr_engine_failed",
                postprocessor=_name,
                error=str(exc),
                ocr_model=request.ocr_model,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="OCR analysis failed. Please try again later.",
            ) from exc

        try:
            data = await postprocessor.process(ocr_result)
        except LlmServiceError as exc:
            logger.error(
                "llm_service_failed",
                postprocessor=_name,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM processing failed. Please try again later.",
            ) from exc
        except Exception as exc:
            logger.error(
                "postprocessor_failed",
                postprocessor=_name,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Postprocessor encountered an unexpected error.",
            ) from exc

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "postprocessor_success",
            postprocessor=_name,
            ocr_model=request.ocr_model,
            processing_time_ms=elapsed_ms,
        )

        return ApiResponse(
            timestamp=datetime.now(UTC),
            processing_time_ms=elapsed_ms,
            ocr_model=request.ocr_model,
            postprocessor=PostProcessorInfo(name=_name, version=_version),
            data=data,
        )

    # Give each route handler a unique __name__ to avoid FastAPI route conflicts
    _route_handler.__name__ = f"postprocessor_{_name.replace('-', '_')}"


def _tags_for(name: str) -> list[str]:
    """Return OpenAPI tags based on postprocessor name."""
    tag_map = {
        "payment-extraction": ["Payment"],
        "id-document-extract": ["ID Document"],
        "document-classify": ["Classification"],
    }
    return tag_map.get(name, [name])
