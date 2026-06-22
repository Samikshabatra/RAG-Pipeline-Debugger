"""Local embedding model (sentence-transformers, all-MiniLM-L6-v2).

Lazy singleton: the model (~80MB) loads on first use and is reused after that,
so importing this module stays cheap and tests can monkeypatch around it.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents/passages. Vectors are L2-normalized so that
    inner product == cosine similarity."""
    vecs = _model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query. all-MiniLM is symmetric, so no special prefix
    (unlike BGE) is needed."""
    return embed_texts([query])[0]
