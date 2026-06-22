"""Generate a dataset of traces for the debugger to analyze and the UI to show.

Runs every labeled test query under two configs:

  * "good"     -> reranker on, top_n=4   (the strong baseline; mostly passes)
  * "degraded" -> reranker off, top_n=1  (cheap config; near-duplicate traps
                  now drop the gold chunk -> genuine retrieval failures)

This is the payoff of the config-levers approach: the same questions pass under
a good config and fail under a cheap one, so the trace store contains both
PASS and FAIL traces for Phase 3 to diagnose and Phase 4 to chart.

    python -m scripts.generate_traces             # both configs, all queries
    python -m scripts.generate_traces --clear     # wipe existing traces first
"""
from __future__ import annotations

import argparse

from backend.corpus import DOCUMENTS, TEST_QUERIES
from backend.pipeline import vector_store
from backend.tracing import store, tracer

CONFIGS = [
    {"label": "good", "use_reranker": True, "top_n": 4},
    {"label": "degraded", "use_reranker": False, "top_n": 1},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="wipe traces first")
    args = parser.parse_args()

    if args.clear:
        print(f"Cleared {store.clear()} existing traces.")

    if vector_store.count() == 0:
        print(f"Indexing {len(DOCUMENTS)} documents...")
        vector_store.index_documents(DOCUMENTS)

    n_pass = n_fail = 0
    for cfg in CONFIGS:
        print(f"\n=== config: {cfg['label']} "
              f"(reranker={cfg['use_reranker']}, top_n={cfg['top_n']}) ===")
        for q in TEST_QUERIES:
            t = tracer.run_traced(
                q.query,
                top_n=cfg["top_n"],
                use_reranker=cfg["use_reranker"],
                expected_doc_id=q.expected_doc_id,
            )
            n_pass += int(t.status == "pass")
            n_fail += int(t.status == "fail")
            mark = "PASS" if t.status == "pass" else "FAIL"
            print(f"  [{mark}] {t.trace_id}  conf={t.confidence}  "
                  f"{int(t.total_latency_ms)}ms  {q.query[:55]}")

    total = n_pass + n_fail
    print(f"\nGenerated {total} traces -> {n_pass} pass, {n_fail} fail "
          f"(stored in SQLite).")


if __name__ == "__main__":
    main()
