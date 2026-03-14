"""Document classification postprocessor.

Classifies the document type and generates a brief summary using the
Azure OpenAI LLM.
"""

from __future__ import annotations

from typing import ClassVar

import structlog
from pydantic import BaseModel

from app.models.classify import ClassificationDataTO
from app.models.extract import ExtractDataTO
from app.postprocessors.base import BasePostProcessor
from app.services.llm_service import LlmService

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a document classification engine for Czech and European business documents.

Analyse the provided document text and return a JSON object with:

1. document_type: Classify the document into ONE of these categories:
   - "faktura"        — Invoice or billing document
   - "smlouva"        — Contract or agreement
   - "vypis"          — Bank statement or account extract
   - "korespondence"  — Correspondence, letter, email printout
   - "plna_moc"       — Power of attorney
   - "zadost"         — Application, request, or petition
   - "reklamace"      — Complaint or warranty claim
   - "ostatni"        — Other / unclassified

   Return as: {"value": "<category>", "confidence": <0.0-1.0>}

2. summary: A 2-3 sentence summary of the document content in the same language as the document.
   Be concise and factual.

3. detected_language: BCP-47 language code of the primary language in the document
   (e.g. "cs" for Czech, "en" for English, "de" for German).

Set confidence to 0.0 if you are unable to determine the document type.
"""


# ---------------------------------------------------------------------------
# Postprocessor class
# ---------------------------------------------------------------------------


class DocumentClassifyPostProcessor(BasePostProcessor):
    """Classifies document type and generates a brief summary."""

    name: ClassVar[str] = "document-classify"
    description: ClassVar[str] = "Classifies document type and generates a brief summary"
    version: ClassVar[str] = "1.0.0"
    response_model: ClassVar[type[BaseModel]] = ClassificationDataTO

    def __init__(self, llm_service: LlmService | None = None) -> None:
        self._llm_service = llm_service

    def _get_llm_service(self) -> LlmService:
        if self._llm_service is not None:
            return self._llm_service
        from app.services.llm_service import LlmService as _LlmService
        self._llm_service = _LlmService()
        return self._llm_service

    async def process(self, ocr_result: ExtractDataTO) -> ClassificationDataTO:
        """Classify the document and generate a summary.

        Args:
            ocr_result: Raw OCR output from the document.

        Returns:
            ClassificationDataTO with document_type, summary, and detected_language.
        """
        llm = self._get_llm_service()
        result: ClassificationDataTO = await llm.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_content=ocr_result.content,
            response_format=ClassificationDataTO,
        )
        logger.debug(
            "document_classify_done",
            document_type=result.document_type.value,
            confidence=result.document_type.confidence,
            language=result.detected_language,
        )
        return result
