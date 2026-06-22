"""Run the pipeline with full tracing.

Same three steps as `rag_pipeline.run`, but each step is timed and recorded as a
Span, and the config levers (top_n, use_reranker) are exposed per-run so we can
deliberately produce passing *and* failing traces:

  * use_reranker=False  -> feed raw retrieval top_n (no rerank rescue)
  * small top_n         -> correct-but-low-ranked chunks get cut before the LLM

The step functions themselves are unchanged -- tracing only wraps them.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from time import perf_counter

from ..config import get_settings
from ..models import GenerateInput, RerankInput, RetrieveInput, RetrievedChunk
from ..pipeline import generator, reranker, retriever
from . import store
from .trace_models import Span, Trace


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gold_rank(doc_id: str | None, chunks: list[RetrievedChunk]) -> int | None:
    if doc_id is None:
        return None
    for c in chunks:
        if c.doc_id == doc_id:
            return c.rank
    return None


def run_traced(
    query: str,
    *,
    top_k: int | None = None,
    top_n: int | None = None,
    use_reranker: bool | None = None,
    generator_model: str | None = None,
    expected_doc_id: str | None = None,
    persist: bool = True,
) -> Trace:
    cfg = get_settings()
    top_k = top_k or cfg.retrieve_top_k
    top_n = top_n or cfg.rerank_top_n
    use_reranker = cfg.use_reranker if use_reranker is None else use_reranker

    spans: list[Span] = []

    # --- Step 1: Retrieve ---
    t0 = perf_counter()
    retrieved = retriever.retrieve(RetrieveInput(query=query, top_k=top_k))
    spans.append(
        Span(
            step="retrieve",
            latency_ms=(perf_counter() - t0) * 1000,
            meta={"top_k": top_k, "n_returned": len(retrieved.chunks)},
            chunks=retrieved.chunks,
        )
    )

    # --- Step 2: Rerank (or bypass) ---
    t1 = perf_counter()
    if use_reranker:
        reranked_chunks = reranker.rerank(
            RerankInput(query=query, chunks=retrieved.chunks, top_n=top_n)
        ).chunks
    else:
        # Bypass: keep retrieval order, take top_n, renumber ranks.
        reranked_chunks = [
            c.model_copy(update={"rank": i + 1})
            for i, c in enumerate(retrieved.chunks[:top_n])
        ]
    spans.append(
        Span(
            step="rerank",
            latency_ms=(perf_counter() - t1) * 1000,
            meta={"top_n": top_n, "enabled": use_reranker},
            chunks=reranked_chunks,
        )
    )

    # --- Step 3: Generate ---
    t2 = perf_counter()
    generated = generator.generate(
        GenerateInput(query=query, chunks=reranked_chunks), model=generator_model
    )
    spans.append(
        Span(
            step="generate",
            latency_ms=(perf_counter() - t2) * 1000,
            meta={"model": generated.model, "n_context_chunks": len(reranked_chunks)},
            prompt=generated.prompt,
            answer=generated.answer,
            confidence=generated.confidence,
        )
    )

    # --- Provisional status: did the gold chunk reach the generator? ---
    # (A retrieval-level signal. Phase 3's judge produces the real verdict.)
    gold_reached = None
    status = "unknown"
    if expected_doc_id is not None:
        gold_reached = _gold_rank(expected_doc_id, reranked_chunks) is not None
        status = "pass" if gold_reached else "fail"

    trace = Trace(
        trace_id=uuid.uuid4().hex[:12],
        query=query,
        created_at=_now_iso(),
        config={
            "top_k": top_k,
            "top_n": top_n,
            "use_reranker": use_reranker,
            "model": generated.model,
        },
        spans=spans,
        answer=generated.answer,
        confidence=generated.confidence,
        total_latency_ms=sum(s.latency_ms for s in spans),
        expected_doc_id=expected_doc_id,
        gold_reached_generator=gold_reached,
        status=status,
    )

    if persist:
        store.save(trace)
    return trace
