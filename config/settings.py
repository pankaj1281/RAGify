"""Application settings and configuration management.

Loads configuration from environment variables with sensible defaults.
"""

import os
from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = "RAGify"
    app_version: str = "1.0.0"
    debug: bool = False

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 512
    openai_temperature: float = 0.2

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_cache_dir: str = "./data/embedding_cache"

    # Vector store
    faiss_index_path: str = "./data/faiss_index"
    top_k: int = 5

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Data
    data_dir: str = "./data"
    upload_dir: str = "./data/uploads"

    # Logging
    log_level: str = "INFO"

    # Cache
    query_cache_size: int = 128


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
