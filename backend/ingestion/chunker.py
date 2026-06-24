"""Split a document's text into overlapping chunks.

Chunks are the unit that gets embedded and retrieved. We split on paragraph and
sentence boundaries where possible, targeting ~600 characters with a small
overlap so a fact that straddles a boundary is not lost.
"""
from __future__ import annotations

import re

from ..models import Document

_CHUNK_SIZE = 600
_OVERLAP = 100


def _split_units(text: str) -> list[str]:
    # Prefer paragraph breaks, then fall back to sentences for long paragraphs.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paras:
        if len(para) <= _CHUNK_SIZE:
            units.append(para)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
    return units


def chunk_text(text: str, source: str) -> list[Document]:
    """Turn raw text into a list of Document chunks with stable ids."""
    units = _split_units(text)
    chunks: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + 1 + len(unit) <= _CHUNK_SIZE:
            buf = f"{buf} {unit}"
        else:
            chunks.append(buf)
            # carry a tail of the previous chunk for overlap/context
            tail = buf[-_OVERLAP:]
            buf = f"{tail} {unit}" if tail else unit
    if buf:
        chunks.append(buf)

    title = source
    return [
        Document(
            doc_id=f"{source}-{i:03d}",
            title=f"{title} (part {i + 1})" if len(chunks) > 1 else title,
            text=chunk,
            category="uploaded",
        )
        for i, chunk in enumerate(chunks)
    ]
