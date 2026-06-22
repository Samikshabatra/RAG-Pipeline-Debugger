"""Typed trace/span models.

A Trace is one end-to-end pipeline run. It holds one Span per step. A Span
captures everything needed to debug that step *without rerunning the pipeline*:
its input params, its output (chunks / prompt / answer), latency, and the
model's self-reported confidence.

Phase 3 (the analyzer) adds judge scores per span plus a final verdict; those
fields live here too but default to empty so a Phase-2 trace is still valid.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import RetrievedChunk


class Span(BaseModel):
    """One pipeline step within a trace."""

    step: str                                   # "retrieve" | "rerank" | "generate"
    latency_ms: float
    meta: dict = Field(default_factory=dict)    # step input params (top_k, top_n, model, ...)

    # Outputs (which fields are set depends on the step)
    chunks: list[RetrievedChunk] = Field(default_factory=list)  # retrieve / rerank
    prompt: str | None = None                   # generate
    answer: str | None = None                   # generate
    confidence: int | None = None               # generate (self-reported 1-5)

    # --- Filled by the Phase 3 analyzer (LLM-as-judge) ---
    judge_score: int | None = None              # 1-5 quality of this step's output
    judge_reason: str | None = None


class Trace(BaseModel):
    """A full pipeline run, persisted and inspectable."""

    trace_id: str
    query: str
    created_at: str                             # ISO-8601 UTC
    config: dict = Field(default_factory=dict)  # top_k, top_n, use_reranker, model
    spans: list[Span] = Field(default_factory=list)

    answer: str = ""
    confidence: int = 0
    total_latency_ms: float = 0.0

    # Labeling / status
    expected_doc_id: str | None = None          # gold chunk, if a labeled query
    gold_reached_generator: bool | None = None  # did gold survive into the generator?
    status: str = "unknown"                      # "pass" | "fail" | "unknown"

    # --- Filled by the Phase 3 analyzer ---
    verdict: str | None = None                   # human-readable root-cause string
    root_cause_step: str | None = None           # "retrieve" | "rerank" | "generate"


class TraceSummary(BaseModel):
    """Compact row for the Phase 4 list view."""

    trace_id: str
    query: str
    created_at: str
    status: str
    confidence: int
    total_latency_ms: float
    root_cause_step: str | None = None
