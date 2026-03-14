"""Response model for the payment extraction postprocessor."""

from pydantic import BaseModel

from app.models.common import FieldWithConfidence


class PaymentDataTO(BaseModel):
    """Extracted payment fields from an invoice or payment order.

    All fields are optional — the LLM returns null when a field is not
    present or cannot be extracted with sufficient confidence.
    """

    account_number: FieldWithConfidence[str | None]
    iban: FieldWithConfidence[str | None]
    amount: FieldWithConfidence[float | None]
    currency: FieldWithConfidence[str | None]
    variable_symbol: FieldWithConfidence[str | None]
    due_date: FieldWithConfidence[str | None]
    payee_name: FieldWithConfidence[str | None]
