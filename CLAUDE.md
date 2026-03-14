# IAIA OCR Service

## Context
Centralized OCR-as-a-Service platform for Air Bank. Replaces various
scattered OCR integrations across the bank with a single REST API. Owned by IAIA team (AI competence center). Fully stateless — no
persistent storage, no database, all processing in-memory per request.

Two-layer architecture:
- Layer 1 (Raw OCR): Azure AI Document Intelligence — blob in,
  structured text out
- Layer 2 (Postprocessors): Azure OpenAI LLM post-processing —
  extraction, classification, summarization on top of OCR output

## API design principles
Each postprocessor has its OWN endpoint with a TYPED response. No generic dict, no dynamic data field. Every endpoint returns a
standardized envelope where only the `data` field type differs:

```
POST /ocr/extract                → ApiResponse[ExtractDataTO]
POST /ocr/payment-extraction     → ApiResponse[PaymentDataTO]
POST /ocr/id-document-extract    → ApiResponse[IdDocumentDataTO]
POST /ocr/document-classify      → ApiResponse[ClassificationDataTO]
```

The envelope is always:
```json
{
  "request_id": "uuid",
  "timestamp": "ISO 8601",
  "processing_time_ms": 2340,
  "ocr_model": "prebuilt-invoice",
  "postprocessor": {"name": "payment-extraction", "version": "1.0.0"},
  "data": { ... typed per endpoint ... }
}
```

For /ocr/extract the `postprocessor` field is null.

Request is the same for all endpoints (OcrRequest):
```json
{
  "blob_base64": "...",
  "ocr_model": "prebuilt-invoice"
}
```

The `ocr_model` determines which Azure Document Intelligence model
to use. No postprocessor_id in request — it's in the URL.

All fields in postprocessor data TOs use FieldWithConfidence[T]:
```json
{"value": "CZ6508000000192000145", "confidence": 0.96}
```

## Versioning
Each endpoint is independently versioned via URL prefix:
```
POST /ocr/v1/payment-extraction
POST /ocr/v2/payment-extraction (adds specific_symbol field)
POST /ocr/v1/id-document-extract (unchanged)
```

Rules:
- Adding optional field = minor (backward compatible, no new version)
- Removing/renaming field = major version bump (new URL, old stays)
- Changing field type = major version bump

## Tech stack
- Python 3.12+
- FastAPI + uvicorn
- Azure AI Document Intelligence SDK (azure-ai-formrecognizer)
- Azure OpenAI SDK (openai)
- Pydantic v2 for all models and validation
- pytest + pytest-asyncio for tests
- Docker for packaging
- Deployment: Google Cloud Run (testing/PoC phase)

## Project structure
```
ocr-service/
├── app/
│   ├── main.py                     # FastAPI app, lifespan, health check
│   ├── api/
│   │   ├── routes/
│   │   │   ├── extract.py          # POST /ocr/extract
│   │   │   ├── payment.py          # POST /ocr/payment-extraction
│   │   │   ├── id_document.py      # POST /ocr/id-document-extract
│   │   │   └── classify.py         # POST /ocr/document-classify
│   │   ├── route_factory.py        # Auto-generates routes from registry
│   │   └── dependencies.py         # DI, shared clients
│   ├── core/
│   │   ├── config.py               # Settings from env vars (pydantic-settings)
│   │   └── ocr_engine.py           # Azure Document Intelligence wrapper
│   ├── postprocessors/
│   │   ├── base.py                 # Abstract BasePostProcessor class
│   │   ├── registry.py             # Auto-discovery
│   │   ├── payment_extraction.py
│   │   ├── document_classify.py
│   │   └── id_document.py
│   ├── models/
│   │   ├── request.py              # OcrRequest (shared for all endpoints)
│   │   ├── envelope.py             # ApiResponse[T] generic envelope
│   │   ├── extract.py              # ExtractDataTO
│   │   ├── payment.py              # PaymentDataTO
│   │   ├── id_document.py          # IdDocumentDataTO
│   │   ├── classify.py             # ClassificationDataTO
│   │   └── common.py               # FieldWithConfidence[T], PostProcessorInfo
│   └── services/
│       └── llm_service.py          # Azure OpenAI wrapper
├── tests/
│   ├── conftest.py
│   ├── test_extract.py
│   ├── test_payment.py
│   ├── test_id_document.py
│   └── test_classify.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Conventions
- Code, docstrings, variable names: English
- REST API: JSON request/response
- Pydantic models for ALL requests and responses — no dict[str, Any]
- Every postprocessor is a standalone module with its own prompt template
- Structured logging (JSON format via structlog)
- Type hints everywhere
- Async for all Azure SDK calls
- Azure credentials: NEVER in code — env vars or Managed Identity
- Confidence scores on every extracted field via FieldWithConfidence

## Postprocessors pattern
Each postprocessor:
- Inherits from BasePostProcessor
- Defines name, description, version as class attributes
- Defines its own response_model (Pydantic class, the *DataTO)
- Implements async process(ocr_result: ExtractDataTO) -> its DataTO
- Has its own prompt template as a class constant
- Is auto-discovered by registry on startup
- Gets its own route auto-generated via route_factory

Adding a new postprocessor = new file in postprocessors/ + new
DataTO model in models/. Route is auto-generated. No manual wiring.

## Error handling
- Use FastAPI exception handlers
- Return structured error responses with error_code, message, details
- Log all errors with request_id correlation
- Never expose internal Azure errors to consumers

## Testing
- Mock Azure SDK calls in tests (never call real APIs in CI)
- Use pytest fixtures for test client and mock services
- Each postprocessor has at least one test with sample OCR output
