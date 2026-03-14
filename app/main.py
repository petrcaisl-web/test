"""FastAPI application entry point.

Initialises the app, configures structured JSON logging, and registers
top-level routes (health check). OCR routes will be mounted here once
implemented.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from pythonjsonlogger import jsonlogger

from app.core.config import get_settings
from app.models.common import ResponseEnvelope

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str) -> None:
    """Configure root logger to emit structured JSON logs."""
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic around the application lifecycle."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info(
        "Starting IAIA OCR Service",
        extra={"env": settings.app_env, "version": app.version},
    )
    yield
    logger.info("Shutting down IAIA OCR Service")


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IAIA OCR Service",
    description=(
        "Centralized OCR-as-a-Service for Air Bank. "
        "Provides raw OCR extraction and LLM-powered recipe processing."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=ResponseEnvelope[dict],
    summary="Health check",
    tags=["system"],
)
async def health_check() -> ResponseEnvelope[dict]:
    """Return service liveness status.

    Always returns HTTP 200 with status 'ok' as long as the process is running.
    Does not perform deep dependency checks — use a dedicated readiness probe
    for that once Azure clients are wired up.
    """
    return ResponseEnvelope(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC),
        processing_time_ms=0,
        data={"status": "ok"},
    )
