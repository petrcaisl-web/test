"""ASGI middleware for the IAIA OCR Service.

Provides:
- RequestIdMiddleware: assigns a UUID to each request, binds it to
  the structlog context so all log entries are correlated, and
  returns it in the X-Request-ID response header.
- TimingMiddleware: measures total request wall time and adds an
  X-Process-Time-Ms response header for observability.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generate or propagate a request ID and bind it to structlog context.

    If the incoming request already carries an ``X-Request-ID`` header,
    that value is reused (useful when the API gateway forwards its own
    trace IDs).  Otherwise a new UUID4 is generated.

    The request ID is:
    - Bound to the structlog context for the duration of the request.
    - Returned in the ``X-Request-ID`` response header.
    - Stored on ``request.state.request_id`` for use in route handlers.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind to structlog context for the lifetime of this async task
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure total request processing time and expose it as a header.

    Adds ``X-Process-Time-Ms`` to every response.  This is a
    convenience header for clients and monitoring dashboards; the
    per-endpoint processing time is also recorded inside the JSON
    response envelope.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

        logger.debug(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            processing_time_ms=elapsed_ms,
        )

        return response
