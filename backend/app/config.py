"""Application configuration via environment variables."""

from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreProvider(str, Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM
    llm_provider: LLMProvider = LLMProvider.GEMINI
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = ""  # Override default model per provider
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Vector Store
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.FAISS
    pinecone_api_key: str = ""
    pinecone_index_name: str = "support-chatbot"
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss_index"

    # Database
    database_url: str = "sqlite+aiosqlite:///./support.db"

    # Server
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    log_level: str = "info"

    # Sentiment thresholds
    sentiment_escalation_threshold: float = -0.5
    sentiment_negative_threshold: float = -0.2

    # Rate limiting
    rate_limit_per_minute: int = 30

    def get_default_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        defaults = {
            LLMProvider.GEMINI: "gemini-2.0-flash",
            LLMProvider.OPENAI: "gpt-4o-mini",
            LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
        }
        return defaults[self.llm_provider]


settings = Settings()
