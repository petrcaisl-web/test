"""Azure cost estimation for Document Intelligence and OpenAI requests.

Pricing is approximate and based on Azure published rates as of early 2025.
These estimates are used for internal monitoring and logging only — they
do not reflect actual billing and may not be up to date.

Azure Document Intelligence pricing reference:
  https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/

Azure OpenAI pricing reference:
  https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
"""

from __future__ import annotations

from app.models.envelope import CostEstimate

# ---------------------------------------------------------------------------
# Pricing constants (USD)
# ---------------------------------------------------------------------------

# Azure Document Intelligence — prebuilt models
# Price per page (S0 tier, read/prebuilt models): $0.001 per page
_DOC_INTELLIGENCE_PRICE_PER_PAGE_USD = 0.001

# Azure OpenAI GPT-4o
# Input tokens: $5.00 per 1M tokens  → $0.000005 per token
# Output tokens: $15.00 per 1M tokens → $0.000015 per token
# We use a blended estimate assuming ~80% input / 20% output
_OPENAI_PRICE_PER_INPUT_TOKEN_USD = 0.000005
_OPENAI_PRICE_PER_OUTPUT_TOKEN_USD = 0.000015
_OPENAI_INPUT_RATIO = 0.80


def estimate(
    num_pages: int,
    openai_prompt_tokens: int = 0,
    openai_completion_tokens: int = 0,
) -> CostEstimate:
    """Estimate the Azure cost for a single OCR service request.

    Args:
        num_pages: Number of pages processed by Azure Document Intelligence.
        openai_prompt_tokens: Number of prompt/input tokens sent to Azure OpenAI.
            Pass 0 for the raw /ocr/extract endpoint (no LLM call).
        openai_completion_tokens: Number of completion/output tokens returned
            by Azure OpenAI.  Pass 0 for the raw /ocr/extract endpoint.

    Returns:
        CostEstimate with per-service breakdown and total in USD.
    """
    doc_cost = num_pages * _DOC_INTELLIGENCE_PRICE_PER_PAGE_USD
    openai_cost = (
        openai_prompt_tokens * _OPENAI_PRICE_PER_INPUT_TOKEN_USD
        + openai_completion_tokens * _OPENAI_PRICE_PER_OUTPUT_TOKEN_USD
    )
    total_tokens = openai_prompt_tokens + openai_completion_tokens
    total_cost = doc_cost + openai_cost

    return CostEstimate(
        doc_intelligence_pages=num_pages,
        openai_tokens=total_tokens,
        estimated_cost_usd=round(total_cost, 6),
    )


def estimate_from_pages(num_pages: int) -> CostEstimate:
    """Estimate cost for a raw OCR extraction (no LLM call).

    Args:
        num_pages: Number of document pages processed.

    Returns:
        CostEstimate with zero OpenAI tokens.
    """
    return estimate(num_pages=num_pages)


def estimate_from_text_length(num_pages: int, content_length: int) -> CostEstimate:
    """Rough token estimate based on content character length.

    Uses an approximate ratio of 4 characters per token (common for
    English/Czech mixed text) to estimate prompt tokens.  Assumes
    ~15% of total tokens are completion tokens.

    Args:
        num_pages: Number of document pages.
        content_length: Length of the document content string (characters).

    Returns:
        CostEstimate with blended token estimate.
    """
    # Rough heuristic: 4 chars per token for the OCR text
    estimated_prompt_tokens = content_length // 4
    # Add ~200 tokens for the system prompt
    total_prompt_tokens = estimated_prompt_tokens + 200
    # Estimate completion as ~15% of prompt
    estimated_completion_tokens = max(50, total_prompt_tokens // 7)

    return estimate(
        num_pages=num_pages,
        openai_prompt_tokens=total_prompt_tokens,
        openai_completion_tokens=estimated_completion_tokens,
    )
