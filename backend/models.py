"""Typed Pydantic models for the whole pipeline.

Every step (Retrieve -> Rerank -> Generate) has an explicit typed input and
output. These clean boundaries are what make the tracing layer (Phase 2) and
the backward analyzer (Phase 3) straightforward: a Span is just the typed
input/output of one step plus some metadata.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
class Document(BaseModel):
    """A single document in the test corpus."""

    doc_id: str
    title: str
    text: str
    category: str = "general"


class RetrievedChunk(BaseModel):
    """A document as it flows through the pipeline, carrying its current score.

    `score` is stage-relative: after retrieval it's the embedding similarity
    (0-1, higher = more similar); after rerank it's the cross-encoder relevance
    score (unbounded logit, higher = more relevant). `rank` is the 1-based
    position within the stage that produced it.
    """

    doc_id: str
    title: str
    text: str
    score: float
    rank: int


# --------------------------------------------------------------------------- #
# Step 1: Retrieve
# --------------------------------------------------------------------------- #
class RetrieveInput(BaseModel):
    query: str
    top_k: int = 8


class RetrieveOutput(BaseModel):
    chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------- #
# Step 2: Rerank
# --------------------------------------------------------------------------- #
class RerankInput(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    top_n: int = 4


class RerankOutput(BaseModel):
    chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------- #
# Step 3: Generate
# --------------------------------------------------------------------------- #
class GenerateInput(BaseModel):
    query: str
    chunks: list[RetrievedChunk]


class GenerateOutput(BaseModel):
    answer: str
    confidence: int = Field(ge=1, le=5)  # model's self-reported confidence
    prompt: str                          # exact prompt sent to the local LLM
    model: str


# --------------------------------------------------------------------------- #
# End-to-end result (Phase 1). Tracing wraps this with spans/latency in Phase 2.
# --------------------------------------------------------------------------- #
class PipelineResult(BaseModel):
    query: str
    retrieved: list[RetrievedChunk]
    reranked: list[RetrievedChunk]
    answer: str
    confidence: int
    model: str
    prompt: str
