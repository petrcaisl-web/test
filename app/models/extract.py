"""Response models for the raw OCR extraction endpoint (/ocr/extract)."""

from pydantic import BaseModel, Field


class LineTO(BaseModel):
    """A single text line as detected by Azure Document Intelligence."""

    content: str
    confidence: float = Field(ge=0.0, le=1.0)


class KeyValuePairTO(BaseModel):
    """A key-value pair extracted from a form or structured document."""

    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class PageTO(BaseModel):
    """All extracted content for a single page of the document."""

    page_number: int = Field(ge=1)
    lines: list[LineTO]
    key_value_pairs: list[KeyValuePairTO]


class ExtractDataTO(BaseModel):
    """Raw OCR output returned by the /ocr/extract endpoint.

    This is also the input passed to every postprocessor for further
    LLM-based processing.
    """

    # Full concatenated text content of the document
    content: str

    # Per-page breakdown of lines and key-value pairs
    pages: list[PageTO]

    # BCP-47 language codes detected in the document (e.g. ["cs", "en"])
    detected_languages: list[str]
