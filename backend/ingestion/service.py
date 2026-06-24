"""Glue: uploaded file bytes -> text -> chunks -> indexed in the vector store."""
from __future__ import annotations

import os

from . import chunker, loaders
from ..pipeline import vector_store


def ingest_file(filename: str, data: bytes) -> dict:
    """Extract, chunk, and index one uploaded file. Returns a small summary."""
    text = loaders.extract_text(filename, data)
    source = os.path.splitext(os.path.basename(filename))[0]
    chunks = chunker.chunk_text(text, source=source)
    indexed = vector_store.add_documents(chunks)
    return {"source": source, "chunks_indexed": indexed, "chars": len(text)}
