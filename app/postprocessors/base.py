"""Abstract base class for all OCR postprocessors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from app.models.extract import ExtractDataTO


class BasePostProcessor(ABC):
    """Abstract base for every postprocessor in the OCR service.

    Subclasses must define class attributes ``name``, ``description``,
    ``version``, and ``response_model``, and implement the ``process``
    coroutine.

    The ``name`` value is used directly as the URL path segment:
        POST /ocr/{name}

    Example subclass skeleton::

        class PaymentExtractionPostProcessor(BasePostProcessor):
            name = "payment-extraction"
            description = "Extracts payment data from invoices"
            version = "1.0.0"
            response_model = PaymentDataTO

            async def process(self, ocr_result: ExtractDataTO) -> PaymentDataTO:
                ...
    """

    # Each subclass MUST override these
    name: ClassVar[str]
    description: ClassVar[str]
    version: ClassVar[str]
    response_model: ClassVar[type[BaseModel]]

    @abstractmethod
    async def process(self, ocr_result: ExtractDataTO) -> BaseModel:
        """Process raw OCR output and return a typed DataTO.

        Args:
            ocr_result: The ExtractDataTO produced by the OCR engine.

        Returns:
            An instance of ``self.response_model`` populated with
            extracted / classified data.
        """
