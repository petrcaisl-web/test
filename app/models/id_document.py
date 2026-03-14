"""Response model for the ID document extraction postprocessor."""

from pydantic import BaseModel

from app.models.common import FieldWithConfidence


class IdDocumentDataTO(BaseModel):
    """Extracted fields from an identity document (ID card, passport, etc.).

    All fields are optional — the LLM returns null when a field is not
    present or cannot be extracted with sufficient confidence.
    """

    first_name: FieldWithConfidence[str | None]
    last_name: FieldWithConfidence[str | None]
    date_of_birth: FieldWithConfidence[str | None]
    document_number: FieldWithConfidence[str | None]
    document_type: FieldWithConfidence[str | None]
    expiry_date: FieldWithConfidence[str | None]
    nationality: FieldWithConfidence[str | None]
