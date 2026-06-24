<div align="center">

# 🔍 RAG Pipeline Debugger

### Failure forensics for RAG systems — find *which* step broke, not just *that* it broke.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama&logoColor=white)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)]()
[![No API Keys](https://img.shields.io/badge/API%20keys-zero-success)]()
[![100% Local](https://img.shields.io/badge/runs-100%25%20local-blueviolet)]()

</div>

---

> **RAG systems fail silently.** The retriever pulls the wrong chunk, but the language model answers
> confidently anyway — so a wrong answer looks exactly like a right one. There's no error, no stack trace.
>
> This tool **traces every step** of a Retrieve → Rerank → Generate pipeline, **scores each step
> independently** with a local LLM-as-judge, and **pinpoints the exact step that introduced the failure** —
> retrieval, ranking, or generation. Every model runs on your own machine. **No OpenAI / Anthropic / Cohere
> key anywhere.**

---

## ✨ Why this project is different

Most RAG demos build the pipeline and stop. This one treats the pipeline as something to be **debugged**:

- 🧭 **Root-cause attribution** — not "the answer is wrong" but *"Ranking failure: the correct chunk was
  retrieved but never reached the model."*
- 🔬 **Independent per-step judging** — each step is graded on its own merit, so a generator that honestly
  refuses bad context is *exonerated* and the blame falls upstream where it belongs.
- 🔒 **100% local, zero cost** — Ollama + sentence-transformers + a local cross-encoder. Works on
  privacy-sensitive data that can't leave the machine — a genuine differentiator, not just a budget hack.
- 📊 **Full observability** — every run is a persisted trace of timed spans you can replay and re-analyze
  without re-running the (slow) local models.
- 📥 **Bring your own documents** — upload PDFs / Word / text files in the dashboard; they're auto-extracted,
  chunked, embedded, and indexed. No editing code.
- 🌐 **Web-search fallback (agentic RAG)** — when the local corpus has nothing relevant, the pipeline
  automatically searches the web (keyless DuckDuckGo), fetches the top pages, and answers from those —
  so out-of-domain questions like *"what is cloud computing?"* still get a real, sourced answer.

---

## 🖼️ Screenshots

> _Drop two screenshots into a `docs/` folder named `dashboard.png` and `waterfall.png`, then
> uncomment the block below — they'll render automatically._

<!--
| Dashboard & charts | Root-cause waterfall |
|---|---|
| ![Dashboard](docs/dashboard.png) | ![Waterfall](docs/waterfall.png) |

*Aggregate failure-by-stage and latency charts (left); a single trace with the guilty **Generate** step
highlighted in red and the judge's reasoning inline (right).*
-->

- **Dashboard** — pass/fail metrics, a *failures-by-stage* chart, and an *average-latency-per-step* chart.
- **Trace detail** — a step-by-step waterfall where the root-cause step is highlighted in red with the
  judge's reasoning shown inline.

---

## 🧠 How it works

```
  User question
      │
      ▼
  ┌──────────────── FastAPI backend (:8030) ─────────────────┐
  │                                                          │
  │   RETRIEVE   ──►   RERANK    ──►   GENERATE              │
  │   Chroma +         cross-          Ollama LLM            │
  │   MiniLM           encoder         (qwen3.5:9b)          │
  │      │               │                │                  │
  │      ▼               ▼                ▼                  │
  │   ┌──────────  TRACING: one Span per step  ──────────┐   │
  │   │  latency · chunks+scores · prompt · confidence   │   │
  │   │                  └──►  SQLite                     │   │
  │   └──────────────────────────────────────────────────┘   │
  │                          │                                │
  │   ┌────────────  BACKWARD ANALYZER  ───────────────┐      │
  │   │  LLM-as-judge scores each step independently,   │      │
  │   │  walks the pipeline, names the root-cause step  │      │
  │   └─────────────────────────────────────────────────┘     │
  └───────────────────────────┬──────────────────────────────┘
                              ▼
                Streamlit Trace Explorer (dashboard)
```

**One question → one trace → three spans.** Each span records everything that happened in its step. The
analyzer then grades each step with an independent judge and walks the pipeline: the **first step whose
quality drops is the root cause**, because an early failure dooms everything downstream.

---

## 🎯 The three failure types it detects

| Verdict | What it means | Example caught in the demo |
|---|---|---|
| 🔴 **Retrieval failure** | The answer was never retrieved from the corpus. | — |
| 🟠 **Ranking failure** | It *was* retrieved, but cut before reaching the model. | *"cancel monthly subscription"* → the **Annual** doc (which literally says "cancellation notice period") out-ranked the correct monthly doc. |
| 🟣 **Generation failure** | The right context reached the model, but it answered wrong anyway. | *"parental leave for fathers"* → `llama3.2:3b` answered **"280 days"** (at confidence 4/5!) when the context said 4 weeks. |

---

## 🧾 Sample output

```text
Query:    How many days of paid parental leave do new fathers get?
Verdict:  Generation failure — the correct context was provided (final context 4/5)
          but the answer was wrong or unsupported (answer 1/5). The answer "280 days"
          is incorrect because the context states fathers get 4 weeks (~28 days).

  1 · Retrieve   13 ms   judge 4/5   ✅  (answer is in the candidate pool)
  2 · Rerank      0 ms   judge 4/5   ✅  (answer kept in the final context)
  3 · Generate 1668 ms   judge 1/5   ⛔  ← ROOT CAUSE
       self-reported confidence: 4/5   |   actual: wrong
```

> 💡 **Key insight:** the failing run self-reported **4/5 confidence**. That's the whole case for an
> *independent* judge — you can't trust a model to grade itself.

---

## 🛠️ Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers` · all-MiniLM-L6-v2 | Small (~80MB), CPU-friendly, free |
| Vector store | **ChromaDB** (persistent, cosine) | Local, simple, no server |
| Reranker | cross-encoder · ms-marco-MiniLM-L-6-v2 | Purpose-built, lightweight, accurate |
| Generation + Judge | **Ollama** · `qwen3.5:9b` | Strong, fits an 8GB GPU, local, no key |
| Weak generator (demo) | `llama3.2:3b` | Produces *real* generation failures |
| Backend | **FastAPI** | Typed, fast, auto-generated docs |
| Storage | **SQLite** | Zero-setup; spans stored as JSON blobs |
| Frontend | **Streamlit** | Fast, clean data dashboards |

---

## 🚀 Quickstart

**1. Prerequisites** — [Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull qwen3.5:9b      # generator + judge
ollama pull llama3.2:3b     # optional: weak model to demo generation failures
```

**2. Install:**

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # macOS/Linux: venv/bin/pip
```

**3. Run:**

```bash
# backend
venv\Scripts\python -m uvicorn backend.app:app --reload --port 8030

# build & analyze a demo dataset (good vs. degraded vs. weak-generator configs)
venv\Scripts\python -m scripts.generate_traces --clear
venv\Scripts\python -m scripts.analyze_traces

# dashboard
venv\Scripts\python -m streamlit run frontend/app.py
```

Open the dashboard, pick a failing trace, and read its root-cause verdict.

---

## 📥 Using your own data

You don't have to touch any code to point this at your own content:

- **Upload files** — in the dashboard sidebar, drop in PDFs / Word / `.txt` / `.md` files and click
  *Index uploaded files*. They're extracted, chunked, embedded, and added to the index immediately
  (also available as `POST /documents/upload`).
- **Web fallback** — flip on *Web-search fallback* (or `ENABLE_WEB_FALLBACK=true`). For any question the
  local corpus can't answer, the system searches the web and answers from the fetched pages, clearly
  marked as a web-sourced answer in the trace.

> RAG always retrieves from *some* source. This project supports three: the built-in corpus, your uploaded
> files, and the live web — so "I have to hand-write documents" is never a constraint.

## ⚙️ Configuration & the "config levers"

All local, no keys. Override via a `.env` (see `.env.example`):

| Setting | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3.5:9b` | Generator + judge |
| `RETRIEVE_TOP_K` | `8` | Candidates pulled from Chroma |
| `RERANK_TOP_N` | `4` | Chunks sent to the LLM |
| `USE_RERANKER` | `true` | Off ⇒ raw retrieval (a "cheap" config) |
| `JUDGE_PASS_THRESHOLD` | `4` | A step scoring below this is flagged weak |
| `ENABLE_WEB_FALLBACK` | `false` | Search the web when local retrieval is weak |
| `WEB_FALLBACK_THRESHOLD` | `0.0` | Below this rerank score ⇒ trigger web search |
| `PORT` | `8030` | Backend port |

`top_n` and `use_reranker` are the **config levers**: the *same* question passes under a strong config and
fails under a cheap one. This is how the tool manufactures genuine, reproducible failures to diagnose —
*"cheap configs fail; better configs fix it."*

---

## 📁 Project structure

```
backend/
  config.py            settings + config levers (no API keys)
  models.py            typed Pydantic models for each step
  corpus.py            24-doc test corpus + labeled trap queries
  app.py               FastAPI endpoints
  pipeline/            retrieve → rerank → generate (+ Ollama client, web search)
  ingestion/           upload → extract text → chunk → index your own files
  tracing/             Span/Trace models, SQLite store, traced runner
  analysis/            LLM-as-judge + backward root-cause analyzer
frontend/app.py        Streamlit trace explorer
scripts/               ingest · run_queries · generate_traces · analyze_traces
tests/                 25 offline tests (LLM calls mocked)
```

---

## 🧪 Testing

```bash
venv\Scripts\python -m pytest -q      # 25 tests, fully offline
```

The embedding/rerank models run locally and **every Ollama call is mocked**, so the suite needs no running
model server. Tests cover the corpus, each pipeline step, JSON parsing edge cases, trace persistence, the
config levers, root-cause attribution for every failure type, and the API endpoints.

---

## 🔑 What I learned building it

- **A good local stack is hard to break.** MiniLM + a cross-encoder + a 9B model answered even a
  deliberately ambiguous, near-duplicate corpus correctly. Real failures only appeared under cheap configs
  or with adversarial near-duplicates where the *wrong* doc carries the query's literal phrasing.
- **Self-reported confidence is unreliable.** Failing runs routinely reported 5/5 confidence — which is
  exactly why an *independent* judge is necessary.
- **Generation dominates latency** (~7–10× the retrieval/rerank steps) — clear from the latency chart.

---

## 🗺️ Roadmap

- [ ] Side-by-side comparison of two configs on the same question
- [ ] Trend view across many runs over time
- [ ] Package the tracer as a decorator that wraps *any* RAG pipeline, not just this demo
- [ ] More adversarial corpus traps

---

## 📄 License

MIT — free to use and adapt.

<div align="center">

**Built by [Samiksha Batra](https://github.com/Samikshabatra)** · 100% local · zero API keys

</div>
