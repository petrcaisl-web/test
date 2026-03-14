"""FastAPI application entry point.

Initialises the app, configures structured JSON logging via structlog,
and registers top-level routes (health check). OCR routes will be
mounted here once the OCR engine and postprocessors are implemented.
"""

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI

from app.core.config import get_settings
from app.models.envelope import ApiResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str) -> None:
    """Configure structlog for structured JSON output."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            # Map string level to int understood by structlog
            getattr(__import__("logging"), log_level, 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic around the application lifecycle."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("starting", service="iaia-ocr-service", env=settings.app_env, version=app.version)
    yield
    logger.info("shutdown", service="iaia-ocr-service")


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IAIA OCR Service",
    description=(
        "Centralized OCR-as-a-Service for Air Bank. "
        "Provides raw OCR extraction and LLM-powered postprocessor endpoints."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=ApiResponse[dict],
    summary="Health check",
    tags=["system"],
)
async def health_check() -> ApiResponse[dict]:
    """Return service liveness status.

    Always returns HTTP 200 with status 'ok' as long as the process is
    running. Does not perform deep dependency checks — a dedicated
    readiness probe should be added once Azure clients are wired up.
    """
    start = time.perf_counter()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return ApiResponse(
        timestamp=datetime.now(UTC),
        processing_time_ms=elapsed_ms,
        ocr_model="none",
        postprocessor=None,
        data={"status": "ok"},
    )
