"""Trace Explorer — Streamlit dashboard for the RAG Pipeline Debugger.

Talks to the FastAPI backend (default http://localhost:8030). Three things:

  1. Aggregate view: pass/fail counts, % of failures by stage, avg latency/step.
  2. Trace list: every run with a pass/fail badge, root-cause step, confidence.
  3. Trace detail: the step-by-step waterfall (query -> retrieved -> reranked ->
     answer) with the root-cause step highlighted and the judge's reasoning shown
     inline.

Run the backend first:  uvicorn backend.app:app --reload --port 8030
Then:                   streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8030")
STEP_ORDER = ["retrieve", "rerank", "generate"]
STEP_LABEL = {"retrieve": "1 · Retrieve", "rerank": "2 · Rerank", "generate": "3 · Generate"}

st.set_page_config(page_title="RAG Pipeline Debugger", page_icon="🔍", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def _get(path: str):
    return requests.get(f"{API_BASE}{path}", timeout=300)


def _post(path: str, json=None):
    return requests.post(f"{API_BASE}{path}", json=json, timeout=600)


def health() -> dict | None:
    try:
        r = _get("/health")
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=10, show_spinner="Loading traces...")
def load_traces() -> list[dict]:
    """Fetch every trace in full (spans included) in a single request."""
    try:
        r = _get("/traces/full?limit=500")
        return r.json() if r.status_code == 200 else []
    except requests.RequestException:
        return []


# --------------------------------------------------------------------------- #
# Sidebar — status + actions
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Backend")
    h = health()
    if h is None:
        st.error(f"Backend unreachable at {API_BASE}")
        st.caption("Start it: `uvicorn backend.app:app --port 8030`")
    else:
        ollama_ok = h.get("ollama") == "up"
        if ollama_ok:
            st.success("Backend up")
        else:
            st.warning("Backend up, Ollama down")
        st.caption(f"Model: `{h.get('model')}` · Docs indexed: {h.get('documents_indexed')}")

    st.divider()
    st.subheader("Run a new trace")
    with st.form("new_trace"):
        q = st.text_input("Question", placeholder="e.g. How much notice to cancel monthly?")
        use_reranker = st.toggle("Use reranker", value=True)
        top_n = st.slider("top_n (chunks to the LLM)", 1, 8, 4)
        weak = st.toggle("Weak generator (llama3.2:3b)", value=False,
                         help="Induce generation failures with a small model")
        web = st.toggle("Web-search fallback", value=False,
                        help="If the local corpus has nothing relevant, search the web")
        run_analyze = st.checkbox("Analyze after running", value=True)
        submitted = st.form_submit_button("Run", use_container_width=True)

    if submitted and q.strip():
        payload = {"query": q, "top_n": top_n, "use_reranker": use_reranker,
                   "use_web_fallback": web}
        if weak:
            payload["generator_model"] = "llama3.2:3b"
        try:
            with st.spinner("Running pipeline..."):
                r = _post("/trace", json=payload)
            if r.status_code == 200:
                tid = r.json()["trace_id"]
                if run_analyze:
                    with st.spinner("Judging each step..."):
                        _post(f"/traces/{tid}/analyze")
                load_traces.clear()
                st.session_state["selected"] = tid
                st.success(f"Trace {tid} ready.")
                st.rerun()
            else:
                st.error(f"Run failed: {r.status_code} {r.text[:200]}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Add your own documents")
    uploads = st.file_uploader("Upload files (PDF / Word / txt / md)",
                               type=["pdf", "docx", "txt", "md"],
                               accept_multiple_files=True)
    if st.button("Index uploaded files", use_container_width=True, disabled=not uploads):
        files = [("files", (u.name, u.getvalue())) for u in uploads]
        try:
            with st.spinner("Extracting, chunking and indexing..."):
                resp = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=600)
            if resp.status_code == 200:
                st.success(f"Indexed {resp.json()['total_chunks_indexed']} chunks "
                           f"from {len(uploads)} file(s).")
            else:
                st.error(f"Upload failed: {resp.status_code} {resp.text[:200]}")
        except requests.RequestException as e:
            st.error(f"Upload failed: {e}")

    st.divider()
    if st.button("↻ Refresh", use_container_width=True):
        load_traces.clear()
        st.rerun()
    if st.button("📥 Re-index built-in corpus", use_container_width=True):
        _post("/ingest")
        st.toast("Corpus re-indexed")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🔍 RAG Pipeline Debugger")
st.caption("Failure forensics for a local RAG pipeline — trace every step, judge "
           "each independently, and pinpoint the root cause. 100% local, no API keys.")

traces = load_traces()
if not traces:
    st.info("No traces yet. Generate some with "
            "`python -m scripts.generate_traces --clear`, or run one from the sidebar.")
    st.stop()

df = pd.DataFrame(traces)


# --------------------------------------------------------------------------- #
# Aggregate view
# --------------------------------------------------------------------------- #
n_total = len(df)
n_fail = int((df["status"] == "fail").sum())
n_pass = n_total - n_fail

c1, c2, c3, c4 = st.columns(4)
c1.metric("Traces", n_total)
c2.metric("Passed", n_pass)
c3.metric("Failed", n_fail)
c4.metric("Failure rate", f"{(n_fail / n_total * 100):.0f}%")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Failures by stage")
    failed = df[df["status"] == "fail"]
    if failed.empty:
        st.caption("No failures recorded.")
    else:
        counts = (failed["root_cause_step"].fillna("unknown")
                  .value_counts().reindex(STEP_ORDER).fillna(0).astype(int))
        counts.index = [STEP_LABEL.get(s, s) for s in counts.index]
        st.bar_chart(counts, color="#e45756", height=260)

with col_b:
    st.subheader("Avg latency per step (ms)")
    rows = []
    for t in traces:
        for s in t["spans"]:
            rows.append({"step": s["step"], "latency_ms": s["latency_ms"]})
    lat = pd.DataFrame(rows).groupby("step")["latency_ms"].mean()
    lat = lat.reindex(STEP_ORDER).fillna(0)
    lat.index = [STEP_LABEL.get(s, s) for s in lat.index]
    st.bar_chart(lat, color="#4c78a8", height=260)

st.divider()


# --------------------------------------------------------------------------- #
# Trace list
# --------------------------------------------------------------------------- #
st.subheader("Traces")

def _badge(row) -> str:
    if row["status"] == "pass":
        return "✅ pass"
    return f"❌ {row['root_cause_step']}"

table = pd.DataFrame({
    "": df.apply(_badge, axis=1),
    "query": df["query"],
    "config": df["config"].apply(
        lambda c: f"rerank={c.get('use_reranker')}, top_n={c.get('top_n')}, "
                  f"gen={c.get('model')}"),
    "conf": df["confidence"],
    "latency (ms)": df["total_latency_ms"].round(0).astype(int),
    "trace_id": df["trace_id"],
})

st.dataframe(table, use_container_width=True, hide_index=True,
             column_config={"trace_id": None})

ids = df["trace_id"].tolist()
default_idx = ids.index(st.session_state["selected"]) if st.session_state.get("selected") in ids else 0
selected = st.selectbox(
    "Inspect a trace", ids, index=default_idx,
    format_func=lambda tid: f"{_badge(df[df.trace_id == tid].iloc[0])} — "
                            f"{df[df.trace_id == tid].iloc[0]['query'][:70]}",
)


# --------------------------------------------------------------------------- #
# Trace detail — the waterfall
# --------------------------------------------------------------------------- #
trace = next(t for t in traces if t["trace_id"] == selected)
st.divider()
st.markdown(f"### Query\n> {trace['query']}")
if trace.get("config", {}).get("source") == "web":
    st.caption("🌐 Answered using the **web-search fallback** (local corpus had nothing relevant).")

# Verdict banner
if trace.get("verdict"):
    banner = st.error if trace["status"] == "fail" else st.success
    banner(f"**Verdict:** {trace['verdict']}")
else:
    st.info("Not analyzed yet — run analysis to get a root-cause verdict.")
    if st.button("Analyze this trace"):
        with st.spinner("Judging each step..."):
            _post(f"/traces/{selected}/analyze")
        load_traces.clear()
        st.rerun()

spans = {s["step"]: s for s in trace["spans"]}


def _score_tag(score) -> str:
    if score is None:
        return "not judged"
    dot = "🟢" if score >= 4 else ("🟡" if score == 3 else "🔴")
    return f"{dot} judge {score}/5"


def render_chunks(chunks: list[dict]):
    if not chunks:
        st.caption("(no chunks)")
        return
    cdf = pd.DataFrame([{
        "rank": c["rank"], "doc_id": c["doc_id"], "title": c["title"],
        "score": round(c["score"], 3),
        "text": c["text"][:160] + ("…" if len(c["text"]) > 160 else ""),
    } for c in chunks])
    st.dataframe(cdf, use_container_width=True, hide_index=True)


for step in STEP_ORDER:
    s = spans.get(step)
    if not s:
        continue
    is_root = trace.get("root_cause_step") == step
    title = f"{STEP_LABEL[step]} · {s['latency_ms']:.0f} ms · {_score_tag(s.get('judge_score'))}"
    if is_root:
        title = "⛔ " + title + "  ← ROOT CAUSE"

    container = st.container(border=True)
    with container:
        if is_root:
            st.error(f"**{title}**")
        else:
            st.markdown(f"**{title}**")
        if s.get("judge_reason"):
            st.caption(f"Judge: {s['judge_reason']}")
        if step in ("retrieve", "rerank"):
            render_chunks(s.get("chunks", []))
        else:  # generate
            st.markdown(f"**Answer** (self-reported confidence {s.get('confidence')}/5):")
            st.info(s.get("answer") or "(no answer)")
            with st.expander("Prompt sent to the LLM"):
                st.code(s.get("prompt") or "", language="text")
