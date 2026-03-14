"""Shared primitive models used across all response types."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class FieldWithConfidence(BaseModel, Generic[T]):
    """Wraps any extracted value together with the model's confidence score.

    Used for every field in postprocessor DataTO models so consumers can
    decide how to handle low-confidence extractions.

    Example:
        {"value": "CZ6508000000192000145", "confidence": 0.96}
    """

    value: T
    confidence: float = Field(ge=0.0, le=1.0)


class PostProcessorInfo(BaseModel):
    """Identifies which postprocessor produced the response.

    Included in the envelope for every postprocessor endpoint.
    Null on the raw /ocr/extract endpoint.
    """

    name: str
    version: str
