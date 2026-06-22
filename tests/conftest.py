"""Shared test fixtures.

Tests run fully offline: the embedding + reranker models are local (downloaded
once from HuggingFace, no API key), and every Ollama call is monkeypatched so
the suite never needs a running LLM server.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.config import get_settings


@pytest.fixture(scope="session")
def settings_tmp_state():
    """Point Chroma/SQLite at a throwaway temp dir for the whole test session."""
    tmp = Path(tempfile.mkdtemp(prefix="ragdbg_test_"))
    cfg = get_settings()
    cfg.chroma_dir = str(tmp / "chroma")
    cfg.trace_db = str(tmp / "traces.db")
    yield cfg


@pytest.fixture(scope="session")
def indexed_corpus(settings_tmp_state):
    """Index the built-in corpus once for retrieval/rerank tests."""
    from backend.corpus import DOCUMENTS
    from backend.pipeline import vector_store

    vector_store.index_documents(DOCUMENTS)
    return DOCUMENTS
