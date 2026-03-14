"""Payment extraction postprocessor.

Extracts payment-relevant fields from Czech invoices and payment orders
using the Azure OpenAI LLM and post-validates the results with regex and
checksum checks.
"""

from __future__ import annotations

import re
from typing import ClassVar

import structlog
from pydantic import BaseModel

from app.models.common import FieldWithConfidence
from app.models.extract import ExtractDataTO
from app.models.payment import PaymentDataTO
from app.postprocessors.base import BasePostProcessor
from app.services.llm_service import LlmService

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a payment-data extraction engine for Czech banking documents.
Extract the following fields from the provided document text.
Return a JSON object matching the given schema exactly.

Field descriptions:
- account_number: Czech bank account in format PREFIX-NUMBER/BANK_CODE (e.g. 19-2000014500/0800) or NUMBER/BANK_CODE. value=null if not present.
- iban: International Bank Account Number, Czech format: CZ + 2 check digits + 20 digits. value=null if not present.
- amount: Numeric amount to pay (no currency symbol, just the number). value=null if not present.
- currency: ISO 4217 currency code (e.g. CZK, EUR). value=null if not present.
- variable_symbol: Variabilní symbol — up to 10 numeric digits. value=null if not present.
- due_date: Payment due date in ISO 8601 format (YYYY-MM-DD). value=null if not present.
- payee_name: Name of the payee / company to pay. value=null if not present.

For each field provide a confidence score between 0.0 (not found / uncertain) and 1.0 (highly confident).
If a field is not found in the document, set value to null and confidence to 0.0.
"""

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VARIABLE_SYMBOL_RE = re.compile(r"^\d{1,10}$")
_CZECH_ACCOUNT_RE = re.compile(r"^(\d{1,6}-)?(\d{2,10})/(\d{4})$")
_IBAN_RE = re.compile(r"^CZ\d{22}$")


def _validate_variable_symbol(value: str | None) -> bool:
    """Return True if value is a valid Czech variable symbol (max 10 digits)."""
    if value is None:
        return True  # null is acceptable
    return bool(_VARIABLE_SYMBOL_RE.match(value.strip()))


def _validate_czech_account(value: str | None) -> bool:
    """Return True if value matches Czech account number format."""
    if value is None:
        return True
    return bool(_CZECH_ACCOUNT_RE.match(value.strip()))


def _validate_iban_checksum(iban: str) -> bool:
    """Validate IBAN using ISO 7064 Mod 97-10 algorithm.

    Args:
        iban: IBAN string (e.g. "CZ6508000000192000145399").

    Returns:
        True if the checksum is valid, False otherwise.
    """
    iban = iban.strip().replace(" ", "").upper()
    if not _IBAN_RE.match(iban):
        return False

    # Move first 4 chars to end, convert letters to digits (A=10, B=11, ...)
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(
        str(ord(ch) - ord("A") + 10) if ch.isalpha() else ch
        for ch in rearranged
    )
    return int(numeric) % 97 == 1


def _post_process(data: PaymentDataTO) -> PaymentDataTO:
    """Apply validation rules and zero out confidence on invalid fields."""
    # Validate variable_symbol
    if data.variable_symbol.value is not None:
        if not _validate_variable_symbol(data.variable_symbol.value):
            logger.warning(
                "payment_validation_failed",
                field="variable_symbol",
                value=data.variable_symbol.value,
            )
            data = data.model_copy(
                update={
                    "variable_symbol": FieldWithConfidence(
                        value=data.variable_symbol.value, confidence=0.0
                    )
                }
            )

    # Validate IBAN checksum
    if data.iban.value is not None:
        if not _validate_iban_checksum(data.iban.value):
            logger.warning(
                "payment_validation_failed",
                field="iban",
                value=data.iban.value,
            )
            data = data.model_copy(
                update={
                    "iban": FieldWithConfidence(
                        value=data.iban.value, confidence=0.0
                    )
                }
            )

    # Validate Czech account number format
    if data.account_number.value is not None:
        if not _validate_czech_account(data.account_number.value):
            logger.warning(
                "payment_validation_failed",
                field="account_number",
                value=data.account_number.value,
            )
            data = data.model_copy(
                update={
                    "account_number": FieldWithConfidence(
                        value=data.account_number.value, confidence=0.0
                    )
                }
            )

    return data


# ---------------------------------------------------------------------------
# Postprocessor class
# ---------------------------------------------------------------------------


class PaymentExtractionPostProcessor(BasePostProcessor):
    """Extracts payment data (account number, IBAN, amount, variable symbol) from invoices."""

    name: ClassVar[str] = "payment-extraction"
    description: ClassVar[str] = (
        "Extracts payment data (account number, IBAN, amount, variable symbol) "
        "from invoices and payment documents"
    )
    version: ClassVar[str] = "1.0.0"
    response_model: ClassVar[type[BaseModel]] = PaymentDataTO

    def __init__(self, llm_service: LlmService | None = None) -> None:
        self._llm_service = llm_service

    def _get_llm_service(self) -> LlmService:
        if self._llm_service is not None:
            return self._llm_service
        # Lazy import to avoid circular import at module level
        from app.services.llm_service import LlmService as _LlmService
        self._llm_service = _LlmService()
        return self._llm_service

    async def process(self, ocr_result: ExtractDataTO) -> PaymentDataTO:
        """Extract payment fields from OCR text via LLM, then validate.

        Args:
            ocr_result: Raw OCR output from the document.

        Returns:
            PaymentDataTO with extracted and validated payment fields.
        """
        llm = self._get_llm_service()
        raw: PaymentDataTO = await llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_content=ocr_result.content,
            response_format=PaymentDataTO,
        )
        validated = _post_process(raw)
        logger.debug(
            "payment_extraction_done",
            iban_confidence=validated.iban.confidence,
            account_confidence=validated.account_number.confidence,
        )
        return validated
