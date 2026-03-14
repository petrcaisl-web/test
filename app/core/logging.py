"""Structured JSON logging configuration for the IAIA OCR Service.

Uses structlog for structured, JSON-formatted log output.  A request_id
is bound to the structlog context at the start of each request via
middleware and flows through all log entries within that request.
"""

from __future__ import annotations

import logging as stdlib_logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for structured JSON output.

    Should be called once during application startup (lifespan).

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR (case-sensitive string).
    """
    level = getattr(stdlib_logging, log_level, stdlib_logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
