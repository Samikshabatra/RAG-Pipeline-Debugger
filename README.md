# RAG Pipeline Debugger

**Failure forensics for RAG systems — 100% local, no API keys.**

RAG systems fail silently: the retriever pulls the wrong chunk, but the generator
answers confidently anyway. This tool traces every step of a Retrieve → Rerank →
Generate pipeline, scores each step *independently* with a local LLM-as-judge, and
**pinpoints exactly which step introduced a failure** — retrieval, ranking, or
generation.

Every model runs on your own machine (Ollama + sentence-transformers + a local
cross-encoder). No OpenAI/Anthropic/Cohere key anywhere — which also makes it
usable for privacy-sensitive data that can't leave the machine.

---

## What it does

- **Traces** every run as a `Trace` of timed `Span`s — inputs, outputs, retrieved
  chunks + scores, the exact prompt sent to the LLM, latency, and the model's
  self-reported confidence — persisted to SQLite and inspectable without rerunning.
- **Judges** each step independently (1–5 + a one-line reason): *is the answer in
  the retrieved pool? in the final context? is the answer correct given that
  context?*
- **Attributes the root cause** by walking the pipeline — the first step whose
  quality drops is flagged:
  - **Retrieval failure** — the answer was never retrieved.
  - **Ranking failure** — it was retrieved but not surfaced into the LLM's context.
  - **Generation failure** — the context was correct but the LLM got it wrong.
- **Explores** it all in a Streamlit dashboard: pass/fail badges, a step-by-step
  waterfall with the root-cause step highlighted and the judge's reasoning inline,
  a *failures-by-stage* chart, and an *average-latency-per-step* chart.

## Architecture

```
                 ┌─────────── FastAPI backend (:8030) ───────────┐
  query  ─────►  │  Retrieve ──► Rerank ──► Generate              │
                 │  (Chroma +    (cross-    (Ollama LLM)           │
                 │   MiniLM)      encoder)                         │
                 │      │            │            │                │
                 │      ▼            ▼            ▼                │
                 │   ┌──────────  Tracing  ──────────┐            │
                 │   │  Span(latency, chunks, prompt, │            │
                 │   │  confidence)  ──►  SQLite      │            │
                 │   └────────────────────────────────┘           │
                 │                  │                              │
                 │   ┌──────  Backward Analyzer  ──────┐          │
                 │   │  LLM-as-judge scores each step,  │          │
                 │   │  attributes the root cause       │          │
                 │   └──────────────────────────────────┘         │
                 └───────────────────┬────────────────────────────┘
                                     ▼
                       Streamlit Trace Explorer (:8502)
```

| Component       | Local tool                                            |
|-----------------|-------------------------------------------------------|
| Embeddings      | `sentence-transformers` · `all-MiniLM-L6-v2`          |
| Vector store    | ChromaDB (persistent, cosine)                         |
| Reranker        | cross-encoder · `ms-marco-MiniLM-L-6-v2`              |
| Generation+Judge| Ollama (`qwen3.5:9b` default) over local HTTP         |
| Backend         | FastAPI                                               |
| Storage         | SQLite (trace metadata + spans as JSON)              |
| Frontend        | Streamlit                                            |

## Quickstart

**Prereqs:** [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull qwen3.5:9b        # default generator + judge
ollama pull llama3.2:3b       # optional: weak model to demo generation failures
```

**Install & run:**

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt          # (use venv/bin on macOS/Linux)

# 1) backend
venv\Scripts\python -m uvicorn backend.app:app --reload --port 8030

# 2) build a trace dataset (good vs. degraded vs. weak-generator configs)
venv\Scripts\python -m scripts.generate_traces --clear
venv\Scripts\python -m scripts.analyze_traces

# 3) dashboard
venv\Scripts\python -m streamlit run frontend/app.py     # talks to :8030
```

Open the dashboard, pick a failing trace, and read its root-cause verdict.

## Configuration

All local, no keys — override via a `.env` (see `.env.example`):

| Setting              | Default                              | Notes                                  |
|----------------------|--------------------------------------|----------------------------------------|
| `OLLAMA_MODEL`       | `qwen3.5:9b`                         | generator + judge                      |
| `RETRIEVE_TOP_K`     | `8`                                  | candidates pulled from Chroma          |
| `RERANK_TOP_N`       | `4`                                  | chunks sent to the LLM                 |
| `USE_RERANKER`       | `true`                               | off ⇒ raw retrieval (a "cheap" config) |
| `JUDGE_PASS_THRESHOLD`| `4`                                 | step below this is flagged weak        |
| `PORT`               | `8030`                               | backend port                           |

`top_n` and `use_reranker` are the **config levers**: the same question passes
under a strong config and fails under a cheap one — handy for generating real
failures to diagnose.

## What I learned building it

- **A good local stack is hard to break.** MiniLM + a cross-encoder + a 9B model
  answered even my deliberately ambiguous, near-duplicate corpus correctly. Real
  retrieval failures only appeared under cheap configs (tight `top_n`, no reranker)
  or with adversarial near-duplicates where the *wrong* doc carries the query's
  literal phrasing.
- **Self-reported confidence is unreliable.** Failing traces routinely report
  confidence 5/5 — a small model insisted "280 days" of parental leave with high
  confidence when the context said 4 weeks. That's the whole case for an
  *independent* judge.
- **Generation is where the time goes.** The latency chart makes it obvious:
  generation dominates total latency; retrieval and reranking are nearly free.

## Testing

```bash
venv\Scripts\python -m pytest -q
```

25 tests, fully offline — embedding/rerank models run locally and every LLM call
is mocked, so the suite needs no running Ollama server.
