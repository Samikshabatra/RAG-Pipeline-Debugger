"""Web-search fallback (agentic RAG) -- keyless.

When the local corpus has nothing relevant, the pipeline can fall back to the
live web: search with DuckDuckGo (no API key), fetch the top pages, extract their
main text, and return them as RetrievedChunk objects so the rest of the pipeline
(rerank -> generate) and the tracer treat them exactly like local chunks.

Everything degrades gracefully: if the network is down or a page fails to fetch,
we fall back to the search snippet, and if search itself fails we return [].
"""
from __future__ import annotations

import logging

import requests

from ..models import RetrievedChunk
from . import embeddings

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-Pipeline-Debugger; local research tool)"}
_MAX_CHARS = 1200
_FETCH_TIMEOUT = 8


def _fetch_page_text(url: str) -> str:
    """Best-effort main-text extraction from a web page."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(t for t in paras if t)
        return text[:_MAX_CHARS]
    except Exception:  # parsing is best-effort
        return ""


def _search(query: str, num_results: int) -> list[dict]:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=num_results))
    except Exception as e:  # network / library issues -> no web results
        log.warning("Web search failed: %s", e)
        return []


def search_chunks(query: str, num_results: int = 4) -> list[RetrievedChunk]:
    """Return web results as ranked RetrievedChunks (score = cosine similarity to
    the query). Empty list if search/network is unavailable."""
    results = _search(query, num_results)
    if not results:
        return []

    texts, metas = [], []
    for r in results:
        title = r.get("title") or r.get("href", "web result")
        page = _fetch_page_text(r.get("href", "")) or (r.get("body") or "")
        body = f"{title}. {page}".strip()
        if body:
            texts.append(body)
            metas.append({"title": title, "url": r.get("href", "")})

    if not texts:
        return []

    # Score by similarity to the query so the retrieve span has meaningful scores.
    qv = embeddings.embed_query(query)
    dvs = embeddings.embed_texts(texts)
    scored = []
    for meta, text, dv in zip(metas, texts, dvs):
        sim = sum(a * b for a, b in zip(qv, dv))  # vectors are normalized
        scored.append((sim, meta, text))
    scored.sort(key=lambda t: t[0], reverse=True)

    return [
        RetrievedChunk(
            doc_id=f"web-{i:02d}",
            title=meta["title"][:80],
            text=text,
            score=max(0.0, float(sim)),
            rank=i + 1,
        )
        for i, (sim, meta, text) in enumerate(scored)
    ]
