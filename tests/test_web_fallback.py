"""Tests for the web-search fallback (web call mocked, no network)."""
from __future__ import annotations

from backend.config import get_settings
from backend.models import RetrievedChunk
from backend.tracing import tracer


def _fake_web(monkeypatch, chunks):
    monkeypatch.setattr(tracer.web_search, "search_chunks", lambda q, num_results=4: chunks)


def _mock_generator(monkeypatch):
    monkeypatch.setattr(tracer.generator.ollama_client, "generate",
                        lambda *a, **k: '{"answer": "from web", "confidence": 4}')


def test_web_fallback_triggers_when_local_is_weak(indexed_corpus, settings_tmp_state, monkeypatch):
    _mock_generator(monkeypatch)
    cfg = get_settings()
    cfg.web_fallback_threshold = 999.0  # force "local is always too weak"
    web_chunks = [RetrievedChunk(doc_id="web-00", title="Cloud Computing",
                                 text="Cloud computing delivers computing over the internet.",
                                 score=0.9, rank=1)]
    _fake_web(monkeypatch, web_chunks)

    t = tracer.run_traced("What is cloud computing?", use_web_fallback=True, persist=False)
    assert t.config["source"] == "web"
    assert t.config["web_fallback"] is True
    assert any(c.doc_id == "web-00" for c in t.spans[0].chunks)  # retrieve span = web


def test_web_fallback_stays_local_when_web_empty(indexed_corpus, settings_tmp_state, monkeypatch):
    _mock_generator(monkeypatch)
    cfg = get_settings()
    cfg.web_fallback_threshold = 999.0
    _fake_web(monkeypatch, [])  # web returns nothing -> keep local

    t = tracer.run_traced("vacation days", use_web_fallback=True, persist=False)
    assert t.config["source"] == "local"
    assert t.config["web_fallback"] is False


def test_no_fallback_when_disabled(indexed_corpus, settings_tmp_state, monkeypatch):
    _mock_generator(monkeypatch)
    # even if web would return something, disabled means it is never called
    def _boom(*a, **k):
        raise AssertionError("web search should not be called when disabled")
    monkeypatch.setattr(tracer.web_search, "search_chunks", _boom)

    t = tracer.run_traced("vacation days", use_web_fallback=False, persist=False)
    assert t.config["source"] == "local"
