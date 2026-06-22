"""Backward trace analyzer -- the core differentiator.

Grades each step of a trace independently with the LLM-as-judge, then walks the
pipeline to attribute the root cause:

    retrieve  -> "is the answer anywhere in the retrieved candidate pool?"
    rerank    -> "is the answer in the final context fed to the generator?"
    generate  -> "is the answer correct given that final context?"

A step is *weak* if its judge score is below the configured threshold. Walking
from the earliest step forward, the FIRST weak step is the root cause -- because
a failure at an early step dooms everything downstream, so that's where quality
was first lost:

    retrieve weak                 -> Retrieval failure   (answer never retrieved)
    retrieve ok, rerank weak      -> Ranking failure     (retrieved but not surfaced)
    retrieve+rerank ok, gen weak  -> Generation failure  (good context, bad answer)
    all ok                        -> no failure

This is independent of the gold label, so it works on real, unlabeled traces.
"""
from __future__ import annotations

from ..config import get_settings
from ..tracing.trace_models import Trace
from . import judge


def _span(trace: Trace, step: str):
    for s in trace.spans:
        if s.step == step:
            return s
    return None


def analyze(trace: Trace) -> Trace:
    """Grade every step, attribute root cause, and write the verdict back into
    the trace (spans get judge_score/judge_reason; trace gets verdict +
    root_cause_step + status). Returns the same trace, mutated."""
    cfg = get_settings()
    threshold = cfg.judge_pass_threshold

    retrieve_span = _span(trace, "retrieve")
    rerank_span = _span(trace, "rerank")
    generate_span = _span(trace, "generate")

    # --- Judge each step independently ---
    r_score, r_reason = judge.score_context_relevance(trace.query, retrieve_span.chunks)
    retrieve_span.judge_score, retrieve_span.judge_reason = r_score, r_reason

    k_score, k_reason = judge.score_context_relevance(trace.query, rerank_span.chunks)
    rerank_span.judge_score, rerank_span.judge_reason = k_score, k_reason

    g_score, g_reason = judge.score_answer(
        trace.query, rerank_span.chunks, generate_span.answer or ""
    )
    generate_span.judge_score, generate_span.judge_reason = g_score, g_reason

    # --- Walk forward; first weak step is the root cause ---
    root_step: str | None = None
    if r_score < threshold:
        root_step = "retrieve"
        verdict = (
            f"Retrieval failure: the information needed to answer was not found "
            f"in the retrieved candidate pool (relevance {r_score}/5). {r_reason}"
        )
    elif k_score < threshold:
        root_step = "rerank"
        verdict = (
            f"Ranking failure: relevant content was retrieved (pool {r_score}/5) "
            f"but was not surfaced into the context sent to the LLM "
            f"(final context {k_score}/5). {k_reason}"
        )
    elif g_score < threshold:
        root_step = "generate"
        verdict = (
            f"Generation failure: the correct context was provided "
            f"(final context {k_score}/5) but the answer was wrong or unsupported "
            f"(answer {g_score}/5). {g_reason}"
        )
    else:
        verdict = (
            f"No failure detected: retrieval {r_score}/5, ranking {k_score}/5, "
            f"answer {g_score}/5."
        )

    trace.root_cause_step = root_step
    trace.verdict = verdict
    trace.status = "fail" if root_step else "pass"
    return trace
