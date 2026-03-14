"""Route handler for POST /ocr/extract — raw OCR extraction."""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import OcrEngineDep
from app.core.ocr_engine import OcrEngineError, UnsupportedModelError
from app.models.envelope import ApiResponse
from app.models.extract import ExtractDataTO
from app.models.request import OcrRequest

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "/extract",
    response_model=ApiResponse[ExtractDataTO],
    summary="Raw OCR extraction",
    tags=["Raw OCR"],
)
async def extract(
    request: OcrRequest,
    ocr_engine: OcrEngineDep,
) -> ApiResponse[ExtractDataTO]:
    """Analyse a document with Azure Document Intelligence and return raw OCR output.

    The `postprocessor` field in the response envelope is always **null** for
    this endpoint — use a postprocessor endpoint for structured extraction.

    Args:
        request: OcrRequest containing base64-encoded document and model ID.
        ocr_engine: Injected OcrEngine singleton.

    Returns:
        ApiResponse[ExtractDataTO] with full text and per-page breakdown.

    Raises:
        422: If the request payload is invalid.
        400: If an unsupported OCR model is requested.
        502: If the Azure Document Intelligence call fails.
    """
    start = time.perf_counter()

    try:
        blob = base64.b64decode(request.blob_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 encoding: {exc}",
        ) from exc

    try:
        ocr_result = await ocr_engine.analyze(blob=blob, model_id=request.ocr_model)
    except UnsupportedModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except OcrEngineError as exc:
        logger.error("ocr_extract_failed", error=str(exc), ocr_model=request.ocr_model)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OCR analysis failed. Please try again later.",
        ) from exc

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "ocr_extract_success",
        ocr_model=request.ocr_model,
        processing_time_ms=elapsed_ms,
        pages=len(ocr_result.pages),
    )

    return ApiResponse(
        timestamp=datetime.now(UTC),
        processing_time_ms=elapsed_ms,
        ocr_model=request.ocr_model,
        postprocessor=None,
        data=ocr_result,
    )
