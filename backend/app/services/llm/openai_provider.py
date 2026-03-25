"""OpenAI LLM provider."""

import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.exceptions import LLMProviderError
from app.models.chat import Message, MessageRole
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self):
        if not settings.openai_api_key:
            raise LLMProviderError("openai", "OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model_name = settings.get_default_model()

    def _format_messages(self, messages: list[Message], system_prompt: str) -> list[dict]:
        formatted = []
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})
        for msg in messages:
            role = msg.role.value
            if role == "assistant":
                role = "assistant"
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
            formatted = self._format_messages(messages, system_prompt)
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted,
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise LLMProviderError("openai", str(e))

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            formatted = self._format_messages(messages, system_prompt)
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted,
                temperature=temperature or settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMProviderError("openai", str(e))

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
