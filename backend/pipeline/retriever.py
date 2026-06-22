"""Step 1: Retrieve.

Embed the query, pull the top_k nearest chunks from ChromaDB, and return them
as ranked RetrievedChunk objects (rank 1 = most similar).
"""
from __future__ import annotations

from ..models import RetrievedChunk, RetrieveInput, RetrieveOutput
from . import embeddings, vector_store


def retrieve(inp: RetrieveInput) -> RetrieveOutput:
    qv = embeddings.embed_query(inp.query)
    hits = vector_store.query(qv, top_k=inp.top_k)
    chunks = [
        RetrievedChunk(
            doc_id=h["doc_id"],
            title=h["title"],
            text=h["text"],
            score=h["score"],
            rank=i + 1,
        )
        for i, h in enumerate(hits)
    ]
    return RetrieveOutput(chunks=chunks)
