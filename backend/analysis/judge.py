"""LLM-as-judge.

Two narrowly-scoped judgments, each returning a 1-5 score + one-line reason:

  * `score_context_relevance` -- does a set of passages CONTAIN what's needed to
    answer the question? (used to grade retrieval and the final context)
  * `score_answer` -- given the context the system actually had, is the answer
    correct and supported? (used to grade generation, independent of whether the
    context was any good -- a sensible refusal over bad context still scores well)

Prompts are deliberately short and JSON-structured to keep local inference fast.
The judge uses a STRONG model (qwen3.5 by default) even when the generator under
test is weak: a reliable grader of a fallible system.
"""
from __future__ import annotations

import json

from ..config import get_settings
from ..models import RetrievedChunk
from ..pipeline import ollama_client

_RELEVANCE_PROMPT = """You are grading a retrieval system, not answering the question.

Question: {question}

Retrieved passages:
{passages}

On a scale of 1 to 5, how well do these passages CONTAIN the information needed
to correctly answer the SPECIFIC question?
5 = the answer is clearly and directly present.
3 = related, but the specific answer is missing or ambiguous.
1 = none of the passages are relevant.

CRITICAL: the question may name a specific subject (a particular plan, version,
caregiver type, etc.). A passage about a DIFFERENT subject (e.g. the Annual plan
when the question asks about the Monthly plan) does NOT answer the question, even
if it is on the same general topic -- score those 1-2, not 3+.

Respond with JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}"""

_ANSWER_PROMPT = """You are grading an answer against the context the system was given.

Question: {question}

Context the system had:
{passages}

Answer the system produced:
{answer}

On a scale of 1 to 5, how correct and well-supported is the answer GIVEN THAT
CONTEXT?
5 = correct and fully supported by the context.
3 = partially correct, or only weakly supported.
1 = wrong, or states facts the context does not support.
Note: if the context genuinely lacks the answer and the system honestly says so,
that is an APPROPRIATE answer for that context -- score it high (4-5).

Respond with JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no passages)"
    return "\n".join(f"[{i + 1}] {c.title}: {c.text}" for i, c in enumerate(chunks))


def _judge_model() -> str:
    cfg = get_settings()
    return cfg.judge_model or cfg.ollama_model


def _ask(prompt: str) -> tuple[int, str]:
    raw = ollama_client.generate(
        prompt, model=_judge_model(), temperature=0.0, num_predict=256, fmt="json"
    )
    return _parse(raw)


def _parse(raw: str) -> tuple[int, str]:
    try:
        data = json.loads(raw.strip())
        score = max(1, min(5, int(data.get("score", 3))))
        reason = str(data.get("reason", "")).strip() or "(no reason given)"
        return score, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        return 3, "(judge response could not be parsed)"


def score_context_relevance(query: str, chunks: list[RetrievedChunk]) -> tuple[int, str]:
    return _ask(_RELEVANCE_PROMPT.format(
        question=query, passages=_format_passages(chunks)
    ))


def score_answer(query: str, chunks: list[RetrievedChunk], answer: str) -> tuple[int, str]:
    return _ask(_ANSWER_PROMPT.format(
        question=query, passages=_format_passages(chunks), answer=answer
    ))
