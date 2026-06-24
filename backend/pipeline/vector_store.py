"""Local vector store (ChromaDB persistent client).

Wraps a single Chroma collection configured for cosine similarity. We compute
embeddings ourselves (see embeddings.py) and hand them to Chroma, rather than
letting Chroma pick a default embedding function -- this keeps the embedding
model explicit and swappable.
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings
from ..models import Document
from . import embeddings

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        cfg = get_settings()
        _client = chromadb.PersistentClient(
            path=cfg.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _get_collection():
    cfg = get_settings()
    # cosine space => distance in [0, 2]; similarity = 1 - distance.
    return _get_client().get_or_create_collection(
        name=cfg.collection_name, metadata={"hnsw:space": "cosine"}
    )


def index_documents(docs: list[Document]) -> int:
    """(Re)index the corpus. Clears the collection first so re-running ingest
    is idempotent. Returns the number of documents indexed."""
    cfg = get_settings()
    client = _get_client()
    # Drop and recreate for a clean, idempotent ingest.
    try:
        client.delete_collection(cfg.collection_name)
    except Exception:
        pass  # collection didn't exist yet
    collection = client.get_or_create_collection(
        name=cfg.collection_name, metadata={"hnsw:space": "cosine"}
    )

    vectors = embeddings.embed_texts([d.text for d in docs])
    collection.add(
        ids=[d.doc_id for d in docs],
        embeddings=vectors,
        documents=[d.text for d in docs],
        metadatas=[{"title": d.title, "category": d.category} for d in docs],
    )
    return len(docs)


def add_documents(docs: list[Document]) -> int:
    """Append documents to the existing collection (does NOT clear it). Used by
    the upload/ingest path so users can grow the corpus without a reset."""
    if not docs:
        return 0
    collection = _get_collection()
    vectors = embeddings.embed_texts([d.text for d in docs])
    collection.add(
        ids=[d.doc_id for d in docs],
        embeddings=vectors,
        documents=[d.text for d in docs],
        metadatas=[{"title": d.title, "category": d.category} for d in docs],
    )
    return len(docs)


def query(query_vector: list[float], top_k: int) -> list[dict]:
    """Return the top_k nearest documents as plain dicts with a 0-1 similarity
    score (higher = more similar)."""
    collection = _get_collection()
    res = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[dict] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
        out.append(
            {
                "doc_id": doc_id,
                "title": meta.get("title", doc_id),
                "text": text,
                # cosine distance -> similarity; clamp tiny negatives from FP error
                "score": max(0.0, 1.0 - float(dist)),
            }
        )
    return out


def count() -> int:
    """Number of documents currently indexed."""
    try:
        return _get_collection().count()
    except Exception:
        return 0
