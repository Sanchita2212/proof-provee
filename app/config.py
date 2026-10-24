from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DATA_DIR: str = "./data_storage"

    # LLM Settings
    LLM_PROVIDER: str = "groq"  # groq, opencode, openai
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    LLM_API_KEY: str = ""

    # Langfuse Settings
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = False

    # CEC Parameters
    MAX_EVIDENCE_CHUNKS: int = 12
    # A subquestion can be answered by a focused excerpt without repeating every
    # token in its wording.  This threshold is deliberately applied only to
    # answerable facets, not to the compiler's full-query routing facet.
    MIN_COVERAGE_THRESHOLD: float = 0.50
    MAX_DOC_CANDIDATES: int = 4
    # Broad questions need a wider, soft document pool before facet-aware
    # evidence selection; focused lookups retain the smaller pool.
    MAX_BROAD_DOC_CANDIDATES: int = 8
    CHUNK_SIZE_CHARS: int = 750
    CHUNK_OVERLAP_CHARS: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
