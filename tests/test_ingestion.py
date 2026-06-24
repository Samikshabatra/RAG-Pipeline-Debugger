"""Tests for uploading and indexing your own documents."""
from __future__ import annotations

from backend.ingestion import chunker, loaders, service
from backend.models import RetrieveInput
from backend.pipeline import retriever, vector_store


def test_chunk_text_splits_and_ids():
    text = ("Para one is short.\n\n" + ("word " * 200) + "\n\nPara three here.")
    chunks = chunker.chunk_text(text, source="mydoc")
    assert len(chunks) >= 2
    assert all(c.doc_id.startswith("mydoc-") for c in chunks)
    assert all(c.category == "uploaded" for c in chunks)


def test_loaders_txt_and_unsupported():
    assert "hello" in loaders.extract_text("a.txt", b"hello world")
    import pytest
    with pytest.raises(loaders.UnsupportedFile):
        loaders.extract_text("a.zip", b"x")


def test_ingest_file_is_searchable(indexed_corpus, settings_tmp_state):
    before = vector_store.count()
    text = ("Quantum widget calibration requires a torque of 42 newton-metres "
            "applied to the flux coupler before each launch.")
    res = service.ingest_file("manual.txt", text.encode())
    assert res["chunks_indexed"] >= 1
    assert vector_store.count() > before
    # the uploaded fact is now retrievable
    out = retriever.retrieve(RetrieveInput(query="torque for the flux coupler", top_k=3))
    assert any("torque" in c.text for c in out.chunks)
