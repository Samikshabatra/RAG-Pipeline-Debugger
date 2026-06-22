"""The full 3-step pipeline: Retrieve -> Rerank -> Generate.

Phase 1 returns a plain PipelineResult. Phase 2 will wrap each step call in a
Span (capturing latency, prompt, confidence) -- but the step functions
themselves stay unchanged, which is the whole point of the typed boundaries.
"""
from __future__ import annotations

from ..config import get_settings
from ..models import (
    GenerateInput,
    PipelineResult,
    RerankInput,
    RetrieveInput,
)
from . import generator, reranker, retriever


def run(query: str, top_k: int | None = None, top_n: int | None = None) -> PipelineResult:
    cfg = get_settings()
    top_k = top_k or cfg.retrieve_top_k
    top_n = top_n or cfg.rerank_top_n

    retrieved = retriever.retrieve(RetrieveInput(query=query, top_k=top_k))
    reranked = reranker.rerank(
        RerankInput(query=query, chunks=retrieved.chunks, top_n=top_n)
    )
    generated = generator.generate(
        GenerateInput(query=query, chunks=reranked.chunks)
    )

    return PipelineResult(
        query=query,
        retrieved=retrieved.chunks,
        reranked=reranked.chunks,
        answer=generated.answer,
        confidence=generated.confidence,
        model=generated.model,
        prompt=generated.prompt,
    )
