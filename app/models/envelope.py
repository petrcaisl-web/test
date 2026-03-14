"""Generic API response envelope returned by every endpoint."""

from datetime import datetime
from typing import Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.common import PostProcessorInfo

T = TypeVar("T")


class CostEstimate(BaseModel):
    """Estimated Azure service costs for this request.

    Values are approximate and based on published Azure pricing.
    Only used for internal monitoring and cost tracking — not a
    contractual guarantee of actual billing.
    """

    doc_intelligence_pages: int = Field(ge=0, description="Number of pages processed by Azure Document Intelligence")
    openai_tokens: int = Field(ge=0, description="Estimated total tokens consumed by Azure OpenAI")
    estimated_cost_usd: float = Field(ge=0.0, description="Estimated total cost in USD")


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
            "cost_estimate": {"doc_intelligence_pages": 1, "openai_tokens": 450, "estimated_cost_usd": 0.012},
            "data": { ... PaymentDataTO fields ... }
        }

    For /ocr/extract the postprocessor field is null and cost_estimate.openai_tokens is 0.
    """

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    processing_time_ms: int = Field(ge=0)
    ocr_model: str
    postprocessor: PostProcessorInfo | None = None
    cost_estimate: CostEstimate | None = None
    data: T
