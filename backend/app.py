"""FastAPI backend.

Phase 1 surface:
  GET  /health        -> service + Ollama status
  POST /ingest        -> (re)index the built-in test corpus
  POST /query         -> run a query end-to-end through the pipeline

Tracing endpoints arrive in Phase 2.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .config import get_settings
from .corpus import DOCUMENTS
from .pipeline import ollama_client, rag_pipeline, vector_store
from .models import PipelineResult

app = FastAPI(title="RAG Pipeline Debugger", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    top_n: int | None = None


@app.get("/health")
def health() -> dict:
    cfg = get_settings()
    return {
        "status": "ok",
        "ollama": "up" if ollama_client.health() else "down",
        "model": cfg.ollama_model,
        "documents_indexed": vector_store.count(),
    }


@app.post("/ingest")
def ingest() -> dict:
    """(Re)index the built-in test corpus into ChromaDB."""
    n = vector_store.index_documents(DOCUMENTS)
    return {"indexed": n}


@app.post("/query", response_model=PipelineResult)
def query(req: QueryRequest) -> PipelineResult:
    return rag_pipeline.run(req.query, top_k=req.top_k, top_n=req.top_n)
