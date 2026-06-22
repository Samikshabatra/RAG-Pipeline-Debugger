"""Phase 1 tests: typed models, retrieval, rerank, generation parsing, e2e."""
from __future__ import annotations

import pytest

from backend.corpus import DOCUMENTS, TEST_QUERIES
from backend.models import GenerateInput, RerankInput, RetrieveInput
from backend.pipeline import generator, rag_pipeline, reranker, retriever


def test_corpus_well_formed():
    ids = [d.doc_id for d in DOCUMENTS]
    assert len(ids) == len(set(ids)), "doc_ids must be unique"
    assert 10 <= len(DOCUMENTS) <= 30, "spec asks for 10-30 docs"
    # every gold answer points at a real doc
    valid = set(ids)
    for q in TEST_QUERIES:
        assert q.expected_doc_id in valid


def test_retrieve_returns_ranked_chunks(indexed_corpus):
    out = retriever.retrieve(RetrieveInput(query="how many vacation days", top_k=5))
    assert len(out.chunks) == 5
    # ranks are 1..N and scores are descending
    assert [c.rank for c in out.chunks] == [1, 2, 3, 4, 5]
    scores = [c.score for c in out.chunks]
    assert scores == sorted(scores, reverse=True)
    # the obviously-correct doc should be retrieved somewhere in top-5
    assert "leave-vacation" in {c.doc_id for c in out.chunks}


def test_rerank_orders_and_truncates(indexed_corpus):
    retrieved = retriever.retrieve(RetrieveInput(query="vacation days per year", top_k=8))
    reranked = reranker.rerank(
        RerankInput(query="vacation days per year", chunks=retrieved.chunks, top_n=3)
    )
    assert len(reranked.chunks) == 3
    assert [c.rank for c in reranked.chunks] == [1, 2, 3]
    scores = [c.score for c in reranked.chunks]
    assert scores == sorted(scores, reverse=True)


def test_rerank_handles_empty():
    out = reranker.rerank(RerankInput(query="x", chunks=[], top_n=3))
    assert out.chunks == []


def test_generator_parse_clean_json():
    ans, conf = generator._parse('{"answer": "20 days", "confidence": 5}')
    assert ans == "20 days"
    assert conf == 5


def test_generator_parse_clamps_and_falls_back():
    # out-of-range confidence is clamped
    _, conf = generator._parse('{"answer": "x", "confidence": 9}')
    assert conf == 5
    # non-JSON falls back to raw text + neutral confidence
    ans, conf = generator._parse("not json at all")
    assert ans == "not json at all"
    assert conf == 3


def test_generator_step_with_mocked_ollama(indexed_corpus, monkeypatch):
    monkeypatch.setattr(
        generator.ollama_client,
        "generate",
        lambda *a, **k: '{"answer": "20 days of paid vacation", "confidence": 5}',
    )
    retrieved = retriever.retrieve(RetrieveInput(query="vacation days", top_k=4))
    out = generator.generate(GenerateInput(query="vacation days", chunks=retrieved.chunks))
    assert out.answer == "20 days of paid vacation"
    assert out.confidence == 5
    assert "Context:" in out.prompt  # grounded prompt was built


def test_full_pipeline_with_mocked_ollama(indexed_corpus, monkeypatch):
    monkeypatch.setattr(
        rag_pipeline.generator.ollama_client,
        "generate",
        lambda *a, **k: '{"answer": "mocked", "confidence": 4}',
    )
    result = rag_pipeline.run("How many vacation days do employees get?")
    assert result.answer == "mocked"
    assert result.confidence == 4
    assert len(result.retrieved) > 0
    assert len(result.reranked) > 0
    # reranked is a subset/reordering, never larger than retrieved
    assert len(result.reranked) <= len(result.retrieved)
