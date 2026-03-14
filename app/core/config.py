"""Application configuration loaded from environment variables.

All Azure credentials must be provided via environment variables or
Azure Identity (managed identity / service principal) — never hardcoded.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings class — values are read from environment variables.

    Pydantic-settings will automatically map env var names to field names
    (case-insensitive). A .env file is supported for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------------------
    # Application
    # ---------------------------------------------------------------------------

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ---------------------------------------------------------------------------
    # Azure AI Document Intelligence
    # ---------------------------------------------------------------------------

    # Endpoint URL, e.g. https://<resource>.cognitiveservices.azure.com/
    azure_document_intelligence_endpoint: str

    # API key — leave empty to use Azure Identity (DefaultAzureCredential)
    azure_document_intelligence_key: str = ""

    # ---------------------------------------------------------------------------
    # Azure OpenAI
    # ---------------------------------------------------------------------------

    # Endpoint URL, e.g. https://<resource>.openai.azure.com/
    azure_openai_endpoint: str

    # API key — leave empty to use Azure Identity (DefaultAzureCredential)
    azure_openai_api_key: str = ""

    # Deployment name for the LLM used in recipes
    azure_openai_deployment: str = "gpt-4o"

    # API version for the Azure OpenAI endpoint
    azure_openai_api_version: str = "2024-02-01"

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @property
    def use_azure_identity(self) -> bool:
        """Return True when API keys are absent and Azure Identity should be used."""
        return not self.azure_document_intelligence_key or not self.azure_openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Using lru_cache ensures the .env file is read only once per process.
    In tests, call get_settings.cache_clear() to reload settings.
    """
    return Settings()
