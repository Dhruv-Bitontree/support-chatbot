"""Anthropic Claude LLM provider."""

import logging
from collections.abc import AsyncIterator

import anthropic

from app.config import settings
from app.exceptions import LLMProviderError
from app.models.chat import Message, MessageRole
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self):
        if not settings.anthropic_api_key:
            raise LLMProviderError("anthropic", "ANTHROPIC_API_KEY not set")
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model_name = settings.get_default_model()

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        formatted = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "assistant"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            formatted = self._format_messages(messages)
            response = await self.client.messages.create(
                model=self.model_name,
                messages=formatted,
                system=system_prompt or "You are a helpful customer support assistant.",
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise LLMProviderError("anthropic", str(e))

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            formatted = self._format_messages(messages)
            async with self.client.messages.stream(
                model=self.model_name,
                messages=formatted,
                system=system_prompt or "You are a helpful customer support assistant.",
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMProviderError("anthropic", str(e))

    async def classify(self, text: str, categories: list[str]) -> str:
        prompt = (
            "You are an intent classifier for an e-commerce support chatbot.\n"
            f"Valid categories: {', '.join(categories)}.\n\n"
            "Rules:\n"
            "- Use order_tracking only for explicit tracking requests (track/check/find status) or when an order ID is provided.\n"
            "- If the message is angry, threatening, sarcastic, or negative about service (even with order words), use complaint.\n"
            "- Use faq for policy/how-to/company-process questions.\n"
            "- Use greeting for short social openers.\n"
            "- Use general for small talk, vague text, or when uncertain.\n"
            "- If uncertain between categories, choose general.\n\n"
            f"Text: {text}\n"
            "Respond with ONLY one category token."
        )
        messages = [Message(role=MessageRole.USER, content=prompt)]
        result = await self.generate(messages, temperature=0.0)
        result = result.strip().lower()
        for cat in categories:
            if cat.lower() in result:
                return cat
        fallback = next((cat for cat in categories if cat.lower() == "general"), categories[0])
        return fallback
