"""Step 2: Rerank.

A cross-encoder (ms-marco-MiniLM-L-6-v2) scores each (query, chunk) pair jointly
-- which is more accurate than the bi-encoder similarity used for retrieval --
and we keep the top_n. This is exactly where a good reranker is supposed to
*rescue* a correct-but-low-ranked chunk; when it fails to, Phase 3 flags it.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from ..config import get_settings
from ..models import RerankInput, RerankOutput, RetrievedChunk


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(get_settings().reranker_model)


def rerank(inp: RerankInput) -> RerankOutput:
    if not inp.chunks:
        return RerankOutput(chunks=[])

    pairs = [(inp.query, c.text) for c in inp.chunks]
    scores = _model().predict(pairs)  # higher = more relevant (raw logit)

    ranked = sorted(
        zip(inp.chunks, scores), key=lambda t: float(t[1]), reverse=True
    )
    out = [
        RetrievedChunk(
            doc_id=c.doc_id,
            title=c.title,
            text=c.text,
            score=float(score),
            rank=i + 1,
        )
        for i, (c, score) in enumerate(ranked[: inp.top_n])
    ]
    return RerankOutput(chunks=out)
