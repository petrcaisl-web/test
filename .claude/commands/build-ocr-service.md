description: Build the full IAIA OCR Service step by step
Execute the following implementation plan sequentially. After each step:
1. Verify the code works (run tests if they exist, check imports)
2. Commit to main with a descriptive message
3. Push to origin main
4. Do NOT wait for approval — proceed to the next step immediately
If a step fails tests, fix the issue before moving on.
Read CLAUDE.md first to understand the project context and conventions.
Step 1: OCR engine + /extract endpoint
Implement the OCR engine and the extract endpoint:
1. app/core/ocr_engine.py:
   * Async wrapper around Azure AI Document Intelligence
   * Method: async analyze(blob: bytes, model_id: str) -> ExtractDataTO
   * Supported models: prebuilt-invoice, prebuilt-receipt, prebuilt-idDocument, prebuilt-layout
   * Retry logic with exponential backoff (3 attempts)
   * Proper error handling (wrap Azure SDK errors into our errors)
2. app/api/routes/extract.py:
   * POST /ocr/extract
   * Accepts OcrRequest
   * Returns ApiResponse[ExtractDataTO] (postprocessor=None)
   * Measures processing time, generates request_id
3. app/api/dependencies.py:
   * Dependency injection for OcrEngine (singleton via app lifespan)
4. tests/test_extract.py:
   * Test with mocked Azure SDK returning sample ExtractDataTO
   * Test error handling (invalid model, Azure timeout)
   * Verify response envelope structure
Initialize Azure client from env vars for now.
Step 2: Postprocessor framework + route factory
Implement the postprocessor framework:
1. app/postprocessors/base.py — abstract BasePostProcessor:
   * Class attrs: name (str), description (str), version (str)
   * Class attr: response_model (type — the Pydantic DataTO class)
   * Abstract method: async process(ocr_result: ExtractDataTO) -> BaseModel
   * The return type is the specific DataTO defined by response_model
2. app/postprocessors/registry.py:
   * PostProcessorRegistry
   * Auto-discovers all BasePostProcessor subclasses on startup
   * list_all() -> list of postprocessor instances
   * get(name: str) -> BasePostProcessor
3. app/services/llm_service.py:
   * Async wrapper around Azure OpenAI
   * Method: async complete(system_prompt: str, user_content: str, response_format: type[T]) -> T
   * Uses structured output (JSON mode) with Pydantic model
   * Retry logic with exponential backoff, error handling
4. app/api/route_factory.py:
   * Function: register_postprocessor_routes(app, registry, ocr_engine, llm_service)
   * For each postprocessor in registry, generates a route: POST /ocr/{postprocessor.name}
      * Accepts OcrRequest
      * Calls ocr_engine.analyze() to get ExtractDataTO
      * Calls postprocessor.process() to get the specific DataTO
      * Returns ApiResponse[postprocessor.response_model]
      * response_model on the FastAPI route decorator ensures typed OpenAPI schema per endpoint
   * Called during app lifespan startup
5. Update app/main.py:
   * Initialize LlmService and PostProcessorRegistry in lifespan
   * Call register_postprocessor_routes()
6. Tests:
   * Test registry auto-discovery (create a dummy postprocessor)
   * Test route factory generates correct endpoints
   * Test LLM service with mocked OpenAI client
This is the key architectural pattern: each postprocessor auto-gets its own typed endpoint. OpenAPI will show full schema for each one.
Step 3: First postprocessor — payment extraction
Implement the first postprocessor: payment_extraction.
app/postprocessors/payment_extraction.py:
* name = "payment-extraction"
* description = "Extracts payment data (account number, IBAN, amount, variable symbol) from invoices and payment documents"
* version = "1.0.0"
* response_model = PaymentDataTO
* Prompt template that instructs the LLM to extract payment fields from the OCR text (ExtractDataTO.content), returning confidence 0.0-1.0 for each field
* The prompt should be specific about Czech banking formats:
   * Czech account number format: prefix-number/bank_code (e.g. 19-2000014500/0800)
   * Variable symbol: up to 10 digits
   * IBAN: CZ + 2 check digits + 20 digits
* Post-processing after LLM response:
   * Regex validation of variable_symbol (max 10 digits, numeric only)
   * IBAN checksum validation (ISO 7064 Mod 97)
   * Czech account number format check (optional prefix, number, bank code)
   * If validation fails for a field, set its confidence to 0.0
tests/test_payment.py:
* Mock both OCR engine and LLM service
* Test with sample ExtractDataTO containing Czech invoice text: "Faktura č. 2024-0892\nDodavatel: ACME s.r.o.\nIČO: 12345678\n Částka k úhradě: 12 500,00 Kč\nVariabilní symbol: 20240892\n Číslo účtu: 19-2000014500/0800\nIBAN: CZ6508000000192000145\n Datum splatnosti: 15.04.2025"
* Verify PaymentDataTO fields and confidence scores
* Test IBAN validation (valid IBAN → high confidence, invalid → 0.0)
* Test variable symbol validation (valid → high confidence, "ABC123" → 0.0)
* Verify the auto-generated route works: POST /ocr/payment-extraction
After implementation, verify /docs shows the payment-extraction endpoint with full typed PaymentDataTO response schema.
Step 4: Remaining postprocessors
Implement two more postprocessors following the same pattern:
4a: Document classification
app/postprocessors/document_classify.py:
* name = "document-classify"
* description = "Classifies document type and generates a brief summary"
* version = "1.0.0"
* response_model = ClassificationDataTO
* Prompt template that:
   * Classifies from DocumentType enum: faktura, smlouva, vypis, korespondence, plna_moc, zadost, reklamace, ostatni
   * Generates 2-3 sentence summary in Czech
   * Detects document language
* No special post-processing needed (LLM output is the result)
tests/test_classify.py:
* Test with sample invoice text → should classify as "faktura"
* Test with sample letter text → should classify as "korespondence"
* Verify ClassificationDataTO structure
4b: ID document extraction
app/postprocessors/id_document.py:
* name = "id-document-extract"
* description = "Extracts personal data from ID documents (passport, national ID, driver's license)"
* version = "1.0.0"
* response_model = IdDocumentDataTO
* Prompt template specific to Czech/EU ID documents:
   * first_name, last_name
   * date_of_birth (ISO format)
   * document_number
   * document_type (passport, national_id, drivers_license)
   * expiry_date (ISO format)
   * nationality (ISO 3166-1 alpha-3)
* Post-processing:
   * Date format validation (must be valid ISO date)
   * Nationality code validation (3 uppercase letters)
tests/test_id_document.py:
* Test with sample passport OCR text
* Test with sample national ID OCR text
* Verify date format validation
After all postprocessors are implemented, verify /docs shows ALL 4 endpoints (extract + 3 postprocessors), each with its own typed response schema. The OcrRequest schema should be shared.
Step 5: Docker + deploy setup
Set up deployment for Google Cloud Run:
1. Dockerfile:
   * Multi-stage build:
      * Builder stage: python:3.12-slim, install dependencies
      * Runtime stage: python:3.12-slim, copy only needed files
   * Non-root user (appuser)
   * ENV PORT=8080 (Cloud Run convention)
   * CMD: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   * HEALTHCHECK using /health endpoint
2. docker-compose.yml for local development:
   * Service: ocr-service
   * Build from Dockerfile
   * Port mapping: 8080:8080
   * env_file: .env
   * Volume mount app/ for hot reload
   * Command override for reload: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
3. .env.example with all required env vars:
   * AZURE_DOC_INTELLIGENCE_ENDPOINT=https://your-instance.cognitiveservices.azure.com
   * AZURE_DOC_INTELLIGENCE_KEY=your-key-here
   * AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com
   * AZURE_OPENAI_KEY=your-key-here
   * AZURE_OPENAI_DEPLOYMENT=gpt-4o
   * LOG_LEVEL=INFO
4. deploy.sh:
#!/bin/bash
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="ocr-service"

gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars="AZURE_DOC_INTELLIGENCE_ENDPOINT=${AZURE_DOC_INTELLIGENCE_ENDPOINT},AZURE_DOC_INTELLIGENCE_KEY=${AZURE_DOC_INTELLIGENCE_KEY},AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT},AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY},AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT}"

echo "Deployed. URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
5. README.md:
   * Project description (IAIA OCR Service)
   * Prerequisites (Python 3.12+, Docker, gcloud CLI)
   * Local development: cp .env.example .env → fill in values → docker compose up
   * API docs: http://localhost:8080/docs
   * Deploy to Cloud Run: ./deploy.sh
   * Endpoint overview table:
Endpoint Description Response POST /ocr/extract Raw OCR ExtractDataTO POST /ocr/payment-extraction Payment data PaymentDataTO POST /ocr/id-document-extract ID document IdDocumentDataTO POST /ocr/document-classify Classification ClassificationDataTO
   * Example curl for each endpoint
Verify the Dockerfile builds successfully: docker build -t ocr-service .
Step 6: Hardening
Add production-readiness features:
1. Structured JSON logging with structlog:
   * Configure structlog in app/core/logging.py
   * JSON output format
   * request_id correlation across all log entries
   * Log every API call: endpoint, processing_time_ms, ocr_model, postprocessor_name, status_code
   * Add logging to ocr_engine, llm_service, and all routes
2. Middleware (app/api/middleware.py):
   * RequestIdMiddleware: generates UUID, sets X-Request-ID header, binds to structlog context
   * TimingMiddleware: measures processing time (already done per endpoint, but add as middleware for consistency)
   * CORS middleware (configurable origins from settings)
3. OpenAPI customization in app/main.py:
   * title = "IAIA OCR Service"
   * description = "Centralized OCR-as-a-Service platform for Air Bank"
   * version = "1.0.0"
   * Add example values in all Pydantic models using model_config with json_schema_extra
   * Tags for endpoint grouping:
      * "Raw OCR" for /ocr/extract
      * "Payment" for /ocr/payment-extraction
      * "ID Document" for /ocr/id-document-extract
      * "Classification" for /ocr/document-classify
4. Cost estimation (app/services/cost_estimator.py):
   * Estimate Azure Document Intelligence cost per request (based on number of pages)
   * Estimate Azure OpenAI cost per request (based on prompt + completion tokens)
   * Log estimated cost with each request
   * Add cost_estimate field to ApiResponse envelope: {"doc_intelligence_pages": 1, "openai_tokens": 450, "estimated_cost_usd": 0.012}
5. Update all existing tests to work with new middleware.
Final check
After all steps are complete:
1. Run ALL tests and show results
2. Show the final project tree
3. Start the app and verify /docs shows all endpoints with correct schemas
4. Show a summary:
   * What was built
   * Number of files, tests, endpoints
   * Any known limitations or TODOs
5. If Docker is available, verify docker build succeeds
