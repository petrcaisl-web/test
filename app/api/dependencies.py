"""FastAPI dependency injection helpers.

The OcrEngine singleton is stored on `app.state` during the lifespan
startup and retrieved here via a FastAPI dependency so routes do not
need to import or construct clients directly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.core.ocr_engine import OcrEngine


def get_ocr_engine(request: Request) -> OcrEngine:
    """Return the OcrEngine singleton stored in app state.

    Raises:
        RuntimeError: If the engine was not initialised during lifespan.
    """
    engine: OcrEngine | None = getattr(request.app.state, "ocr_engine", None)
    if engine is None:
        raise RuntimeError("OcrEngine has not been initialised in app.state")
    return engine


OcrEngineDep = Annotated[OcrEngine, Depends(get_ocr_engine)]
