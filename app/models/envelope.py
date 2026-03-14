"""Generic API response envelope returned by every endpoint."""

from datetime import datetime
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.common import PostProcessorInfo

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Standardized envelope wrapping all endpoint responses.

    The `data` field is the only part that differs between endpoints —
    its type is determined by the generic parameter T.

    Example (payment extraction):
        {
            "request_id": "a1b2c3...",
            "timestamp": "2024-01-15T10:30:00Z",
            "processing_time_ms": 2340,
            "ocr_model": "prebuilt-invoice",
            "postprocessor": {"name": "payment-extraction", "version": "1.0.0"},
            "data": { ... PaymentDataTO fields ... }
        }

    For /ocr/extract the postprocessor field is null.
    """

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    processing_time_ms: int = Field(ge=0)
    ocr_model: str
    postprocessor: PostProcessorInfo | None = None
    data: T
