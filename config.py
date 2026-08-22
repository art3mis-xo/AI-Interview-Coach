# config.py
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    groq_api_key: str = ""

    # Model settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000

    # Interview settings
    max_questions: int = 5
    default_difficulty: Literal["easy", "medium", "hard"] = "medium"

    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_k: int = 3

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()