"""Application settings.

All defaults live here, in code, so behaviour changes are versioned and reviewed.
Only secrets (OPENAI_API_KEY) are expected to come from `.env` / the environment.

Only the composition root (app/) should call `get_settings()`. Other layers
receive the specific values they need as arguments, which keeps them decoupled
from configuration and easy to test.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/rights_assistant_agent/config.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated variables in .env
        frozen=True,  # settings are read-only after startup
    )

    # --- Secrets (required, no default) ---
    openai_api_key: SecretStr  # SecretStr hides the value in logs/repr

    # --- Models ---
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.0  # deterministic routing/extraction
    # Must be the SAME at ingest and query time. Changing it requires
    # deleting `chroma_dir` and re-running ingest.
    embedding_model: str = "text-embedding-3-small"

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    chroma_dir: Path = PROJECT_ROOT / ".chroma"
    chroma_collection: str = "kolzchut"

    # --- Retrieval ---
    chunk_size: int = Field(default=1000, gt=0)  # characters per chunk
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)  # chunks per search = citable sources

    # --- Agent behaviour ---
    # Clarifying rounds before answering with stated assumptions.
    max_clarify_rounds: int = Field(default=2, ge=0, le=5)


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse the same instance."""
    return Settings()
