"""Response model for the document classification postprocessor."""

from enum import StrEnum

from pydantic import BaseModel

from app.models.common import FieldWithConfidence


class DocumentType(StrEnum):
    """Supported document type labels used by the classification postprocessor."""

    FAKTURA = "faktura"
    SMLOUVA = "smlouva"
    VYPIS = "vypis"
    KORESPONDENCE = "korespondence"
    PLNA_MOC = "plna_moc"
    ZADOST = "zadost"
    REKLAMACE = "reklamace"
    OSTATNI = "ostatni"


class ClassificationDataTO(BaseModel):
    """Classification result returned by the document-classify postprocessor."""

    # Document type with confidence — value is always one of DocumentType
    document_type: FieldWithConfidence[DocumentType]

    # Short human-readable summary of the document content
    summary: str

    # BCP-47 language code of the primary language detected in the document
    detected_language: str
