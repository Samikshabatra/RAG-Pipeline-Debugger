"""FastAPI backend.

Phase 1 surface:
  GET  /health        -> service + Ollama status
  POST /ingest        -> (re)index the built-in test corpus
  POST /query         -> run a query end-to-end through the pipeline

Tracing endpoints arrive in Phase 2.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .corpus import DOCUMENTS
from .pipeline import ollama_client, rag_pipeline, vector_store
from .models import PipelineResult
from .analysis import analyzer
from .tracing import store as trace_store
from .tracing import tracer
from .tracing.trace_models import Trace, TraceSummary

app = FastAPI(title="RAG Pipeline Debugger", version="0.2.0")


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    top_n: int | None = None


class TraceRequest(BaseModel):
    query: str
    top_k: int | None = None
    top_n: int | None = None
    use_reranker: bool | None = None
    expected_doc_id: str | None = None


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


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
@app.post("/trace", response_model=Trace)
def trace(req: TraceRequest) -> Trace:
    """Run a query end-to-end with full tracing and persist the trace."""
    return tracer.run_traced(
        req.query,
        top_k=req.top_k,
        top_n=req.top_n,
        use_reranker=req.use_reranker,
        expected_doc_id=req.expected_doc_id,
    )


@app.get("/traces", response_model=list[TraceSummary])
def list_traces(limit: int = 200) -> list[TraceSummary]:
    return trace_store.list_summaries(limit=limit)


@app.get("/traces/{trace_id}", response_model=Trace)
def get_trace(trace_id: str) -> Trace:
    t = trace_store.get(trace_id)
    if t is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return t


@app.delete("/traces")
def clear_traces() -> dict:
    return {"deleted": trace_store.clear()}


# --------------------------------------------------------------------------- #
# Analysis (backward trace analyzer / LLM-as-judge)
# --------------------------------------------------------------------------- #
@app.post("/traces/{trace_id}/analyze", response_model=Trace)
def analyze_trace(trace_id: str) -> Trace:
    """Grade each step with the LLM-as-judge and attribute the root cause."""
    t = trace_store.get(trace_id)
    if t is None:
        raise HTTPException(status_code=404, detail="trace not found")
    analyzer.analyze(t)
    trace_store.save(t)
    return t
