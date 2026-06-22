"""Central configuration.

All settings have sensible local defaults and can be overridden via a `.env`
file or environment variables. There are deliberately NO API keys here — every
model in this project (embeddings, reranker, generator, judge) runs locally.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of the `backend/` package. Used to resolve relative paths
# (CHROMA_DIR, TRACE_DB) the same way no matter where the process is launched.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Ollama (local LLM: generation + judging) ---
    ollama_base_url: str = "http://localhost:11434"
    # Default to the model actually installed on this machine. Swap to
    # "llama3.1:8b" or "mistral:7b" via OLLAMA_MODEL once those are pulled.
    ollama_model: str = "qwen3.5:9b"
    request_timeout: int = 180  # local inference is slow; be generous

    # --- Local embedding + reranker models (HuggingFace, cached locally) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Retrieval knobs ---
    retrieve_top_k: int = 8   # how many ChromaDB pulls before rerank
    rerank_top_n: int = 4     # how many survive rerank into the generator

    # --- Local state ---
    chroma_dir: str = "data/chroma"
    collection_name: str = "corpus"
    trace_db: str = "data/traces.db"

    # --- Backend ---
    port: int = 8030

    @property
    def chroma_path(self) -> str:
        """Absolute path to the Chroma persistence dir (created if missing)."""
        p = (ROOT_DIR / self.chroma_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def trace_db_path(self) -> str:
        """Absolute path to the SQLite trace DB (parent dir created if missing)."""
        p = (ROOT_DIR / self.trace_db).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
