"""Extract plain text from uploaded files.

Supports PDF, Word (.docx), and plain text / markdown. The point is that a user
never has to hand-edit the corpus: they upload a file and the system pulls the
text out, ready to be chunked and indexed.
"""
from __future__ import annotations

import io

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    """Return the plain text of an uploaded file, chosen by extension."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _from_pdf(data)
    if name.endswith(".docx"):
        return _from_docx(data)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise UnsupportedFile(
        f"Unsupported file type: {filename}. Supported: {sorted(SUPPORTED)}"
    )


def _from_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages)


def _from_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)
