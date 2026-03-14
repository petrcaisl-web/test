"""FastAPI application entry point.

Initialises the app, configures structured JSON logging via structlog,
registers the OCR engine and postprocessor routes during lifespan startup.
"""

import logging as stdlib_logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI

from app.core.config import get_settings
from app.core.ocr_engine import OcrEngine
from app.models.envelope import ApiResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str) -> None:
    """Configure structlog for structured JSON output."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(stdlib_logging, log_level, stdlib_logging.INFO)
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

    # Initialise singletons
    from app.core.ocr_engine import OcrEngine
    from app.services.llm_service import LlmService
    from app.postprocessors.registry import PostProcessorRegistry
    from app.api.route_factory import register_postprocessor_routes

    ocr_engine = OcrEngine()
    llm_service = LlmService()
    registry = PostProcessorRegistry.build()

    # Store on app.state for dependency injection
    app.state.ocr_engine = ocr_engine
    app.state.llm_service = llm_service
    app.state.registry = registry

    # Dynamically register postprocessor routes
    register_postprocessor_routes(app, registry, ocr_engine, llm_service)

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
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Include static routers
# ---------------------------------------------------------------------------

from app.api.routes.extract import router as extract_router  # noqa: E402

app.include_router(extract_router, prefix="/ocr")

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
    running.
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
