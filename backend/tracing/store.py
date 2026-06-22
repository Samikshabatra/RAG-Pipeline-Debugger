"""SQLite persistence for traces.

One row per trace: scalar columns for cheap list/filter queries, plus the full
Trace serialized as a JSON blob in `data` so a trace is fully reconstructable
without rerunning the pipeline. Uses only the stdlib `sqlite3`.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from ..config import get_settings
from .trace_models import Trace, TraceSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id          TEXT PRIMARY KEY,
    query             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL,
    confidence        INTEGER NOT NULL,
    total_latency_ms  REAL NOT NULL,
    root_cause_step   TEXT,
    data              TEXT NOT NULL          -- full Trace JSON
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(get_settings().trace_db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save(trace: Trace) -> None:
    """Insert or replace a trace (idempotent on trace_id)."""
    with _conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO traces
               (trace_id, query, created_at, status, confidence,
                total_latency_ms, root_cause_step, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace.trace_id,
                trace.query,
                trace.created_at,
                trace.status,
                trace.confidence,
                trace.total_latency_ms,
                trace.root_cause_step,
                trace.model_dump_json(),
            ),
        )


def get(trace_id: str) -> Trace | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
        ).fetchone()
    return Trace.model_validate_json(row["data"]) if row else None


def list_summaries(limit: int = 200) -> list[TraceSummary]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT trace_id, query, created_at, status, confidence,
                      total_latency_ms, root_cause_step
               FROM traces ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [TraceSummary(**dict(r)) for r in rows]


def clear() -> int:
    """Delete all traces. Returns the number removed."""
    with _conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        conn.execute("DELETE FROM traces")
    return int(n)
