"""Run the backward analyzer over every stored trace and print verdicts.

    python -m scripts.analyze_traces

For each trace this prints the per-step judge scores, the attributed root-cause
step, and the human-readable verdict. When a trace came from a labeled query,
it also shows whether the judge's pass/fail agrees with the gold label.
"""
from __future__ import annotations

from backend.analysis import analyzer
from backend.tracing import store

STEP_ORDER = ["retrieve", "rerank", "generate"]


def main() -> None:
    summaries = store.list_summaries()
    if not summaries:
        print("No traces found. Run: python -m scripts.generate_traces --clear")
        return

    agree = total_labeled = 0
    for summ in summaries:
        trace = store.get(summ.trace_id)
        analyzer.analyze(trace)
        store.save(trace)

        scores = {s.step: s.judge_score for s in trace.spans}
        score_str = "  ".join(f"{s}={scores.get(s)}" for s in STEP_ORDER)
        mark = "PASS" if trace.status == "pass" else f"FAIL@{trace.root_cause_step}"

        print("=" * 80)
        print(f"[{mark}] {trace.query}")
        print(f"  config        : {trace.config}")
        print(f"  judge scores  : {score_str}")
        print(f"  verdict       : {trace.verdict}")

        # Cross-check the RETRIEVAL-STAGE verdict against the gold label.
        # (The gold proxy only knows whether the gold chunk reached the LLM, so
        # it can validate retrieve/rerank verdicts -- but it is structurally
        # blind to generation failures, which only the judge can catch.)
        if trace.expected_doc_id is not None:
            total_labeled += 1
            gold_reached = trace.gold_reached_generator
            judge_retrieval_ok = trace.root_cause_step not in ("retrieve", "rerank")
            ok = (gold_reached == judge_retrieval_ok)
            agree += int(ok)
            tag = ""
            if trace.root_cause_step == "generate":
                tag = " (generation failure: invisible to the gold proxy)"
            print(f"  gold check    : gold_reached={gold_reached} "
                  f"judge_retrieval_ok={judge_retrieval_ok} "
                  f"-> {'agree' if ok else 'DISAGREE'}{tag}")

    print("=" * 80)
    if total_labeled:
        print(f"Judge vs. gold (retrieval-stage) agreement: {agree}/{total_labeled}")


if __name__ == "__main__":
    main()
