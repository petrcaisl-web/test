"""FastAPI dependency injection helpers.

Singletons (OcrEngine, LlmService, PostProcessorRegistry) are stored on
``app.state`` during the lifespan startup and retrieved here via FastAPI
dependencies so routes do not need to import or construct clients directly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.ocr_engine import OcrEngine
from app.services.llm_service import LlmService


def get_ocr_engine(request: Request) -> OcrEngine:
    """Return the OcrEngine singleton stored in app state.

    Raises:
        RuntimeError: If the engine was not initialised during lifespan.
    """
    engine: OcrEngine | None = getattr(request.app.state, "ocr_engine", None)
    if engine is None:
        raise RuntimeError("OcrEngine has not been initialised in app.state")
    return engine


def get_llm_service(request: Request) -> LlmService:
    """Return the LlmService singleton stored in app state.

    Raises:
        RuntimeError: If the service was not initialised during lifespan.
    """
    service: LlmService | None = getattr(request.app.state, "llm_service", None)
    if service is None:
        raise RuntimeError("LlmService has not been initialised in app.state")
    return service


OcrEngineDep = Annotated[OcrEngine, Depends(get_ocr_engine)]
LlmServiceDep = Annotated[LlmService, Depends(get_llm_service)]
