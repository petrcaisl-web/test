# IAIA OCR Service

## Kontext
Centralizovaná OCR-as-a-Service platforma pro Air Bank. Nahrazuje
různé OCR integrace napříč bankou jedním REST API. Spadá pod tým
IAIA (AI competence center).

## Architektura
Dvouvrstvá služba:
- **Vrstva 1 (Raw OCR)**: Azure AI Document Intelligence — blob in,
  structured text out
- **Vrstva 2 (Recepty)**: Azure OpenAI LLM post-processing — extrakce,
  klasifikace, sumarizace nad OCR výstupem

Konzumenti volají buď čistý OCR (/ocr/extract) nebo OCR + recept
(/ocr/process).

## Tech stack
- Python 3.12+
- FastAPI + uvicorn
- Azure AI Document Intelligence SDK
- Azure OpenAI SDK
- Pydantic v2 pro modely a validaci
- pytest pro testy

## Konvence
- Kód a docstringy anglicky
- Komentáře anglicky
- REST API JSON response format
- Pydantic modely pro všechny requesty i response
- Každý recept je samostatný modul s vlastním prompt template
- Structured logging (JSON)
- Type hints všude
- Async kde to dává smysl (Azure SDK calls)

## Struktura projektu
```
ocr-service/
├── app/
│   ├── main.py                  # FastAPI app, startup
│   ├── api/
│   │   ├── routes/              # endpoint definice
│   │   └── dependencies.py      # DI, auth
│   ├── core/
│   │   ├── config.py            # settings z env vars
│   │   └── ocr_engine.py        # wrapper nad Azure Doc Intelligence
│   ├── recipes/
│   │   ├── base.py              # abstract Recipe class
│   │   ├── registry.py          # recipe discovery & lookup
│   │   ├── payment_extraction.py
│   │   ├── document_classify.py
│   │   └── id_document.py
│   ├── models/                  # Pydantic request/response models
│   └── services/
│       └── llm_service.py       # wrapper nad Azure OpenAI
├── tests/
├── CLAUDE.md
├── pyproject.toml
└── Dockerfile
```

## Pravidla
- Azure credentials NIKDY v kódu — vždy přes env vars / Azure Identity
- Recepty musí být pluggable — přidání nového receptu = nový soubor
  v recipes/ + registrace
- Každý endpoint vrací standardizovaný response envelope
  s request_id, timestamp, processing_time_ms
- Confidence scores u každého extrahovaného pole
