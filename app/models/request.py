"""Shared request model used by all OCR endpoints."""

from enum import StrEnum

from pydantic import BaseModel, Field


class OcrModel(StrEnum):
    """Supported Azure Document Intelligence prebuilt models.

    The value is passed directly to the Azure SDK as the model ID,
    so it must match the Azure API model name exactly.
    """

    INVOICE = "prebuilt-invoice"
    RECEIPT = "prebuilt-receipt"
    ID_DOCUMENT = "prebuilt-idDocument"
    LAYOUT = "prebuilt-layout"


class OcrRequest(BaseModel):
    """Request body shared across all OCR endpoints.

    The postprocessor is determined by the URL, not this payload,
    so no postprocessor_id field is needed here.
    """

    # Base64-encoded document blob (PDF or image)
    blob_base64: str = Field(min_length=1)

    # Azure Document Intelligence model to use for raw OCR
    ocr_model: OcrModel
