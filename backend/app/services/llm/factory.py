"""LLM provider factory."""

import logging

from app.config import LLMProvider as LLMProviderEnum, settings
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_provider_instance: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Get or create the configured LLM provider singleton."""
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_type = settings.llm_provider
    logger.info(f"Initializing LLM provider: {provider_type.value}")

    if provider_type == LLMProviderEnum.GEMINI:
        from app.services.llm.gemini import GeminiProvider
        _provider_instance = GeminiProvider()
    elif provider_type == LLMProviderEnum.OPENAI:
        from app.services.llm.openai_provider import OpenAIProvider
        _provider_instance = OpenAIProvider()
    elif provider_type == LLMProviderEnum.ANTHROPIC:
        from app.services.llm.anthropic_provider import AnthropicProvider
        _provider_instance = AnthropicProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}")

    return _provider_instance


def reset_provider():
    """Reset the singleton (for testing)."""
    global _provider_instance
    _provider_instance = None
