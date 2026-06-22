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
