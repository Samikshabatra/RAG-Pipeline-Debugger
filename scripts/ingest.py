"""Index the built-in test corpus into ChromaDB.

    python -m scripts.ingest
"""
from __future__ import annotations

from backend.corpus import DOCUMENTS
from backend.pipeline import vector_store


def main() -> None:
    n = vector_store.index_documents(DOCUMENTS)
    print(f"Indexed {n} documents into ChromaDB.")


if __name__ == "__main__":
    main()
