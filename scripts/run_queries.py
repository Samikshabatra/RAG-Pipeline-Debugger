"""Run the labeled test queries through the full pipeline and print a report.

This is the Phase 1 end-to-end demo. It shows, per query:
  * the final answer + self-reported confidence
  * whether the gold chunk was retrieved, and at what rank
  * whether the gold chunk survived rerank into the top_n
  * a PASS/FAIL marker (did the gold chunk reach the generator?)

The engineered "trap" queries are expected to fail here -- that's the point;
Phase 3 will diagnose *why* each failure happened.

    python -m scripts.run_queries          # all queries
    python -m scripts.run_queries --traps  # only the engineered failure cases
"""
from __future__ import annotations

import argparse

from backend.corpus import DOCUMENTS, TEST_QUERIES
from backend.pipeline import rag_pipeline, vector_store


def _rank_of(doc_id: str, chunks) -> int | None:
    for c in chunks:
        if c.doc_id == doc_id:
            return c.rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traps", action="store_true", help="only engineered failure cases")
    parser.add_argument("--reindex", action="store_true", help="reindex corpus first")
    args = parser.parse_args()

    if args.reindex or vector_store.count() == 0:
        print(f"Indexing {len(DOCUMENTS)} documents...")
        vector_store.index_documents(DOCUMENTS)

    queries = [q for q in TEST_QUERIES if q.expect_fail] if args.traps else TEST_QUERIES

    passed = 0
    for q in queries:
        result = rag_pipeline.run(q.query)
        retr_rank = _rank_of(q.expected_doc_id, result.retrieved)
        rerank_rank = _rank_of(q.expected_doc_id, result.reranked)
        reached_generator = rerank_rank is not None
        passed += int(reached_generator)

        marker = "PASS" if reached_generator else "FAIL"
        print("=" * 78)
        print(f"[{marker}] {q.query}")
        print(f"  gold chunk      : {q.expected_doc_id}")
        print(f"  retrieved rank  : {retr_rank}   reranked rank: {rerank_rank}")
        print(f"  confidence      : {result.confidence}/5")
        print(f"  answer          : {result.answer}")
        if q.expect_fail:
            print(f"  trap            : {q.trap}")

    print("=" * 78)
    print(f"{passed}/{len(queries)} queries delivered the gold chunk to the generator.")


if __name__ == "__main__":
    main()
