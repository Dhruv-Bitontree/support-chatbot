"""Google Gemini LLM provider."""

import logging
from collections.abc import AsyncIterator

import google.generativeai as genai

from app.config import settings
from app.exceptions import LLMProviderError
from app.models.chat import Message, MessageRole
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self):
        if not settings.gemini_api_key:
            raise LLMProviderError("gemini", "GEMINI_API_KEY not set")
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.get_default_model()
        self.model = genai.GenerativeModel(self.model_name)

    def _format_messages(self, messages: list[Message], system_prompt: str) -> tuple[str | None, list[dict]]:
        history = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            history.append({"role": role, "parts": [msg.content]})
        return system_prompt or None, history

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            system, history = self._format_messages(messages, system_prompt)
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system,
            )
            config = genai.types.GenerationConfig(
                temperature=temperature or settings.llm_temperature,
                max_output_tokens=max_tokens or settings.llm_max_tokens,
            )
            response = await model.generate_content_async(
                history,
                generation_config=config,
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMProviderError("gemini", str(e))

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        try:
            system, history = self._format_messages(messages, system_prompt)
            model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system,
            )
            config = genai.types.GenerationConfig(
                temperature=temperature or settings.llm_temperature,
                max_output_tokens=max_tokens or settings.llm_max_tokens,
            )
            response = await model.generate_content_async(
                history,
                generation_config=config,
                stream=True,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise LLMProviderError("gemini", str(e))

    async def classify(self, text: str, categories: list[str]) -> str:
        prompt = (
            f"Classify the following text into exactly one of these categories: {', '.join(categories)}.\n"
            f"Text: {text}\n"
            f"Respond with ONLY the category name, nothing else."
        )
        messages = [Message(role=MessageRole.USER, content=prompt)]
        result = await self.generate(messages, temperature=0.0)
        result = result.strip().lower()
        for cat in categories:
            if cat.lower() in result:
                return cat
        return categories[-1]  # fallback
