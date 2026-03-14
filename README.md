# IAIA OCR Service

Centralized OCR-as-a-Service platform for Air Bank. Replaces scattered OCR integrations
across the bank with a single REST API. Fully stateless — no persistent storage,
all processing in-memory per request.

## Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerised local development)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) (for Cloud Run deployment)
- Azure AI Document Intelligence resource
- Azure OpenAI resource with a GPT-4o deployment

## Local Development

### Option A — Docker Compose (recommended)

```bash
cp .env.example .env
# Fill in your Azure credentials in .env

docker compose up
```

The service starts on http://localhost:8080 with hot reload enabled.

### Option B — Python venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Fill in your Azure credentials in .env

uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## API Documentation

After starting the service, visit:

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## Endpoints

| Endpoint | Description | Response |
|---|---|---|
| `POST /ocr/extract` | Raw OCR extraction (Azure Document Intelligence output) | `ApiResponse[ExtractDataTO]` |
| `POST /ocr/payment-extraction` | Payment data extraction from invoices | `ApiResponse[PaymentDataTO]` |
| `POST /ocr/id-document-extract` | Personal data from ID documents | `ApiResponse[IdDocumentDataTO]` |
| `POST /ocr/document-classify` | Document type classification + summary | `ApiResponse[ClassificationDataTO]` |
| `GET /health` | Service liveness probe | `ApiResponse[dict]` |

All postprocessor endpoints share the same request body:

```json
{
  "blob_base64": "<base64-encoded PDF or image>",
  "ocr_model": "prebuilt-invoice"
}
```

Supported `ocr_model` values: `prebuilt-invoice`, `prebuilt-receipt`, `prebuilt-idDocument`, `prebuilt-layout`.

## Example Requests

### Raw OCR extraction

```bash
BLOB=$(base64 -w0 invoice.pdf)
curl -X POST http://localhost:8080/ocr/extract \
  -H "Content-Type: application/json" \
  -d "{\"blob_base64\": \"$BLOB\", \"ocr_model\": \"prebuilt-invoice\"}"
```

### Payment extraction

```bash
BLOB=$(base64 -w0 invoice.pdf)
curl -X POST http://localhost:8080/ocr/payment-extraction \
  -H "Content-Type: application/json" \
  -d "{\"blob_base64\": \"$BLOB\", \"ocr_model\": \"prebuilt-invoice\"}"
```

### ID document extraction

```bash
BLOB=$(base64 -w0 passport.jpg)
curl -X POST http://localhost:8080/ocr/id-document-extract \
  -H "Content-Type: application/json" \
  -d "{\"blob_base64\": \"$BLOB\", \"ocr_model\": \"prebuilt-idDocument\"}"
```

### Document classification

```bash
BLOB=$(base64 -w0 document.pdf)
curl -X POST http://localhost:8080/ocr/document-classify \
  -H "Content-Type: application/json" \
  -d "{\"blob_base64\": \"$BLOB\", \"ocr_model\": \"prebuilt-layout\"}"
```

## Deploy to Google Cloud Run

```bash
export GCP_PROJECT_ID=your-gcp-project
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://...
export AZURE_DOCUMENT_INTELLIGENCE_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
export AZURE_OPENAI_KEY=...
export AZURE_OPENAI_DEPLOYMENT=gpt-4o

./deploy.sh
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

Two-layer processing pipeline:
1. **Layer 1 (Raw OCR)**: Azure AI Document Intelligence — document in, structured text out
2. **Layer 2 (Postprocessors)**: Azure OpenAI LLM — extraction, classification, summarization

Adding a new postprocessor requires only:
- A new file in `app/postprocessors/` inheriting from `BasePostProcessor`
- A new `DataTO` model in `app/models/`
- No manual route wiring — the route factory auto-generates the endpoint
