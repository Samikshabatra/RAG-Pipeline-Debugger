"""Phase 3 tests: root-cause attribution for each failure type (judge mocked)."""
from __future__ import annotations

import pytest

from backend.analysis import analyzer, judge
from backend.models import RetrievedChunk
from backend.tracing.trace_models import Span, Trace


def _chunk(doc_id="d", rank=1):
    return RetrievedChunk(doc_id=doc_id, title="t", text="x", score=1.0, rank=rank)


def _trace():
    return Trace(
        trace_id="t1",
        query="q",
        created_at="2026-01-01T00:00:00+00:00",
        spans=[
            Span(step="retrieve", latency_ms=1.0, chunks=[_chunk()]),
            Span(step="rerank", latency_ms=1.0, chunks=[_chunk()]),
            Span(step="generate", latency_ms=1.0, prompt="p", answer="a", confidence=5),
        ],
        answer="a",
        confidence=5,
    )


def _mock_scores(monkeypatch, retrieve, rerank, answer):
    # The analyzer calls score_context_relevance twice (retrieve then rerank),
    # then score_answer once. Sequence the return values accordingly.
    calls = {"relevance": [retrieve, rerank], "answer": [answer]}
    monkeypatch.setattr(judge, "score_context_relevance",
                        lambda q, chunks: (calls["relevance"].pop(0), "ctx reason"))
    monkeypatch.setattr(judge, "score_answer",
                        lambda q, chunks, ans: (calls["answer"].pop(0), "ans reason"))


def test_pass_all_strong(monkeypatch):
    _mock_scores(monkeypatch, retrieve=5, rerank=5, answer=5)
    t = analyzer.analyze(_trace())
    assert t.status == "pass"
    assert t.root_cause_step is None
    assert "No failure" in t.verdict


def test_retrieval_failure(monkeypatch):
    _mock_scores(monkeypatch, retrieve=1, rerank=1, answer=1)
    t = analyzer.analyze(_trace())
    assert t.status == "fail"
    assert t.root_cause_step == "retrieve"
    assert "Retrieval failure" in t.verdict


def test_ranking_failure(monkeypatch):
    # answer present in pool, but dropped from final context
    _mock_scores(monkeypatch, retrieve=5, rerank=1, answer=2)
    t = analyzer.analyze(_trace())
    assert t.status == "fail"
    assert t.root_cause_step == "rerank"
    assert "Ranking failure" in t.verdict


def test_generation_failure(monkeypatch):
    # good context all the way, but the answer is wrong
    _mock_scores(monkeypatch, retrieve=5, rerank=5, answer=1)
    t = analyzer.analyze(_trace())
    assert t.status == "fail"
    assert t.root_cause_step == "generate"
    assert "Generation failure" in t.verdict


def test_judge_scores_written_to_spans(monkeypatch):
    _mock_scores(monkeypatch, retrieve=5, rerank=4, answer=4)
    t = analyzer.analyze(_trace())
    by_step = {s.step: s for s in t.spans}
    assert by_step["retrieve"].judge_score == 5
    assert by_step["rerank"].judge_score == 4
    assert by_step["generate"].judge_score == 4
    assert all(s.judge_reason for s in t.spans)


def test_judge_parse_fallback():
    score, reason = judge._parse("not valid json")
    assert score == 3
    assert "could not be parsed" in reason
