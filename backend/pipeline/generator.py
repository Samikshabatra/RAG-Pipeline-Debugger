"""Step 3: Generate.

Build a grounded prompt from the reranked chunks, call the local Ollama model,
and parse a structured answer + self-reported confidence (1-5). The exact prompt
is returned so the tracing layer (Phase 2) can persist it.
"""
from __future__ import annotations

import json

from ..models import GenerateInput, GenerateOutput, RetrievedChunk
from . import ollama_client
from ..config import get_settings

PROMPT_TEMPLATE = """You are a precise assistant answering questions strictly from the provided context.

Rules:
- Answer ONLY using the context below. Do not use outside knowledge.
- If the context does not contain the answer, say you don't know.
- Be concise (1-3 sentences).
- Report your confidence as an integer from 1 (guessing) to 5 (certain),
  based on how well the context supports your answer.

Context:
{context}

Question: {question}

Respond with a JSON object of exactly this shape:
{{"answer": "<your answer>", "confidence": <1-5>}}"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context retrieved)"
    return "\n\n".join(
        f"[{i + 1}] {c.title}: {c.text}" for i, c in enumerate(chunks)
    )


def generate(inp: GenerateInput, model: str | None = None) -> GenerateOutput:
    prompt = PROMPT_TEMPLATE.format(
        context=_format_context(inp.chunks), question=inp.query
    )
    raw = ollama_client.generate(
        prompt, model=model, temperature=0.2, num_predict=1024, fmt="json"
    )

    answer, confidence = _parse(raw)
    return GenerateOutput(
        answer=answer,
        confidence=confidence,
        prompt=prompt,
        model=model or get_settings().ollama_model,
    )


def _parse(raw: str) -> tuple[str, int]:
    """Parse the model's JSON response, with graceful fallbacks."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        answer = str(data.get("answer", "")).strip()
        confidence = int(data.get("confidence", 3))
        confidence = max(1, min(5, confidence))
        if answer:
            return answer, confidence
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: model didn't return clean JSON -> treat raw text as the answer
    # with neutral confidence so the pipeline never hard-fails here.
    return (raw or "(no answer produced)"), 3
