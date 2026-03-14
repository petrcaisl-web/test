"""ID document extraction postprocessor.

Extracts personal data from Czech/EU identity documents (passport,
national ID, driver's licence) via the Azure OpenAI LLM, with
post-processing validation on dates and nationality codes.
"""

from __future__ import annotations

import re
from datetime import date
from typing import ClassVar

import structlog
from pydantic import BaseModel

from app.models.common import FieldWithConfidence
from app.models.extract import ExtractDataTO
from app.models.id_document import IdDocumentDataTO
from app.postprocessors.base import BasePostProcessor
from app.services.llm_service import LlmService

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an identity-document data extraction engine for Czech and EU identity documents.

Extract the following fields from the provided OCR text of an identity document
(passport, national ID card, or driver's licence).

Fields:
- first_name: Given name(s) as on the document. value=null if not found.
- last_name: Family name / surname as on the document. value=null if not found.
- date_of_birth: Date of birth in ISO 8601 format YYYY-MM-DD. value=null if not found.
- document_number: Document number / passport number. value=null if not found.
- document_type: One of: "passport", "national_id", "drivers_license". value=null if not found.
- expiry_date: Document expiry date in ISO 8601 format YYYY-MM-DD. value=null if not found.
- nationality: ISO 3166-1 alpha-3 nationality code (e.g. "CZE" for Czech Republic, "SVK" for Slovakia).
  value=null if not found.

For each field provide a confidence score 0.0–1.0.
If the field cannot be read or is absent, set value=null and confidence=0.0.
"""

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NATIONALITY_RE = re.compile(r"^[A-Z]{3}$")


def _is_valid_iso_date(value: str | None) -> bool:
    """Return True if value is a valid ISO 8601 date string (YYYY-MM-DD)."""
    if value is None:
        return True
    if not _ISO_DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _is_valid_nationality(value: str | None) -> bool:
    """Return True if value is a valid ISO 3166-1 alpha-3 code (3 uppercase letters)."""
    if value is None:
        return True
    return bool(_NATIONALITY_RE.match(value))


def _post_process(data: IdDocumentDataTO) -> IdDocumentDataTO:
    """Validate date fields and nationality; zero confidence on failures."""
    updates: dict = {}

    # Validate date_of_birth
    if not _is_valid_iso_date(data.date_of_birth.value):
        logger.warning(
            "id_doc_validation_failed",
            field="date_of_birth",
            value=data.date_of_birth.value,
        )
        updates["date_of_birth"] = FieldWithConfidence(
            value=data.date_of_birth.value, confidence=0.0
        )

    # Validate expiry_date
    if not _is_valid_iso_date(data.expiry_date.value):
        logger.warning(
            "id_doc_validation_failed",
            field="expiry_date",
            value=data.expiry_date.value,
        )
        updates["expiry_date"] = FieldWithConfidence(
            value=data.expiry_date.value, confidence=0.0
        )

    # Validate nationality
    if not _is_valid_nationality(data.nationality.value):
        logger.warning(
            "id_doc_validation_failed",
            field="nationality",
            value=data.nationality.value,
        )
        updates["nationality"] = FieldWithConfidence(
            value=data.nationality.value, confidence=0.0
        )

    if updates:
        data = data.model_copy(update=updates)
    return data


# ---------------------------------------------------------------------------
# Postprocessor class
# ---------------------------------------------------------------------------


class IdDocumentPostProcessor(BasePostProcessor):
    """Extracts personal data from ID documents (passport, national ID, driver's license)."""

    name: ClassVar[str] = "id-document-extract"
    description: ClassVar[str] = (
        "Extracts personal data from ID documents (passport, national ID, driver's license)"
    )
    version: ClassVar[str] = "1.0.0"
    response_model: ClassVar[type[BaseModel]] = IdDocumentDataTO

    def __init__(self, llm_service: LlmService | None = None) -> None:
        self._llm_service = llm_service

    def _get_llm_service(self) -> LlmService:
        if self._llm_service is not None:
            return self._llm_service
        from app.services.llm_service import LlmService as _LlmService
        self._llm_service = _LlmService()
        return self._llm_service

    async def process(self, ocr_result: ExtractDataTO) -> IdDocumentDataTO:
        """Extract personal data from an identity document.

        Args:
            ocr_result: Raw OCR output from the identity document.

        Returns:
            IdDocumentDataTO with validated personal data fields.
        """
        llm = self._get_llm_service()
        raw: IdDocumentDataTO = await llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_content=ocr_result.content,
            response_format=IdDocumentDataTO,
        )
        validated = _post_process(raw)
        logger.debug(
            "id_document_extract_done",
            document_type=validated.document_type.value,
            nationality=validated.nationality.value,
        )
        return validated
