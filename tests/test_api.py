"""API smoke tests via FastAPI TestClient (Ollama mocked)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.tracing import tracer


@pytest.fixture
def client(indexed_corpus, settings_tmp_state, monkeypatch):
    monkeypatch.setattr(
        tracer.generator.ollama_client,
        "generate",
        lambda *a, **k: '{"answer": "mocked", "confidence": 4}',
    )
    return TestClient(app_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_trace_then_fetch_by_id(client):
    r = client.post("/trace", json={"query": "vacation days", "expected_doc_id": "leave-vacation"})
    assert r.status_code == 200
    trace_id = r.json()["trace_id"]

    r2 = client.get(f"/traces/{trace_id}")
    assert r2.status_code == 200
    assert r2.json()["trace_id"] == trace_id
    assert len(r2.json()["spans"]) == 3


def test_fetch_unknown_trace_404(client):
    assert client.get("/traces/doesnotexist").status_code == 404


def test_list_traces(client):
    client.post("/trace", json={"query": "API rate limits"})
    r = client.get("/traces")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) >= 1


def test_traces_full_includes_spans(client):
    client.post("/trace", json={"query": "vacation days"})
    r = client.get("/traces/full")
    assert r.status_code == 200
    assert len(r.json()) >= 1
    assert len(r.json()[0]["spans"]) == 3  # full traces carry spans


def test_upload_documents(client):
    files = [("files", ("note.txt", b"The mascot of our team is a purple otter named Glitch.", "text/plain"))]
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 200
    assert r.json()["total_chunks_indexed"] >= 1


def test_upload_rejects_unsupported(client):
    files = [("files", ("archive.zip", b"PK\x03\x04", "application/zip"))]
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 415


def test_analyze_endpoint(client, monkeypatch):
    # mock the judge so analysis is offline + deterministic
    from backend.analysis import analyzer
    monkeypatch.setattr(analyzer.judge, "score_context_relevance", lambda q, c: (5, "ok"))
    monkeypatch.setattr(analyzer.judge, "score_answer", lambda q, c, a: (5, "ok"))

    tid = client.post("/trace", json={"query": "vacation days"}).json()["trace_id"]
    r = client.post(f"/traces/{tid}/analyze")
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] is not None
    assert all(s["judge_score"] == 5 for s in body["spans"])
