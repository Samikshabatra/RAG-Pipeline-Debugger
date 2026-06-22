"""Phase 2 tests: span capture, config levers, SQLite persistence."""
from __future__ import annotations

import pytest

from backend.tracing import store, tracer


@pytest.fixture(autouse=True)
def _mock_ollama(monkeypatch):
    """Every traced run goes through the generator -> mock the LLM call."""
    monkeypatch.setattr(
        tracer.generator.ollama_client,
        "generate",
        lambda *a, **k: '{"answer": "mocked answer", "confidence": 4}',
    )


def test_trace_has_three_spans_with_latency(indexed_corpus):
    t = tracer.run_traced("How many vacation days?", persist=False)
    assert [s.step for s in t.spans] == ["retrieve", "rerank", "generate"]
    assert all(s.latency_ms >= 0 for s in t.spans)
    assert t.total_latency_ms == pytest.approx(sum(s.latency_ms for s in t.spans))
    # generate span carries prompt + answer + confidence
    gen = t.spans[-1]
    assert gen.prompt and gen.answer == "mocked answer" and gen.confidence == 4


def test_reranker_bypass_changes_context(indexed_corpus):
    """use_reranker=False feeds raw retrieval top_n (the degraded lever)."""
    on = tracer.run_traced("vacation days per year", use_reranker=True, top_n=4, persist=False)
    off = tracer.run_traced("vacation days per year", use_reranker=False, top_n=4, persist=False)
    assert on.config["use_reranker"] is True
    assert off.config["use_reranker"] is False
    assert off.spans[1].meta["enabled"] is False
    # both still feed at most top_n chunks to the generator
    assert len(on.spans[1].chunks) <= 4
    assert len(off.spans[1].chunks) <= 4


def test_status_from_gold_label(indexed_corpus):
    # Clean query: gold should reach the generator -> pass
    good = tracer.run_traced(
        "How many vacation days do full-time employees get per year?",
        expected_doc_id="leave-vacation",
        top_n=4,
        use_reranker=True,
        persist=False,
    )
    assert good.status == "pass"
    assert good.gold_reached_generator is True

    # Degraded config on the cancellation trap: gold dropped -> fail
    bad = tracer.run_traced(
        "How much notice must I give to cancel my monthly subscription?",
        expected_doc_id="cancel-monthly",
        top_n=1,
        use_reranker=True,
        persist=False,
    )
    assert bad.status == "fail"
    assert bad.gold_reached_generator is False


def test_persist_and_fetch_roundtrip(indexed_corpus, settings_tmp_state):
    store.clear()
    t = tracer.run_traced("API rate limits", expected_doc_id="api-v2-limits")
    fetched = store.get(t.trace_id)
    assert fetched is not None
    assert fetched.trace_id == t.trace_id
    assert fetched.query == "API rate limits"
    assert [s.step for s in fetched.spans] == ["retrieve", "rerank", "generate"]

    summaries = store.list_summaries()
    assert any(s.trace_id == t.trace_id for s in summaries)


def test_clear_traces(indexed_corpus, settings_tmp_state):
    tracer.run_traced("anything", persist=True)
    assert store.clear() >= 1
    assert store.list_summaries() == []
