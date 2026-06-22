"""Low-level client for the local Ollama server.

Plain HTTP to http://localhost:11434 -- no API key, no auth header. Shared by
the generator (Phase 1) and the LLM-as-judge (Phase 3).

IMPORTANT (thinking-model gotcha): the default model here, qwen3.5, is a
*thinking* model. With a small token budget it spends the whole budget on hidden
reasoning and returns an empty `response`. We therefore (a) ask Ollama to
disable thinking via "think": false, and (b) defensively strip any leftover
<think>...</think> block, and (c) use a generous num_predict.
"""
from __future__ import annotations

import re

import requests

from ..config import get_settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    """Raised when the local Ollama server is unreachable or errors out."""


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def generate(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    num_predict: int = 1024,
    fmt: str | None = None,
) -> str:
    """Call Ollama /api/generate and return the cleaned text response.

    `model` overrides the configured default (e.g. a weak generator vs. a strong
    judge). `fmt="json"` asks the model to emit strict JSON (used for structured
    answer+confidence and judge scores).
    """
    cfg = get_settings()
    payload: dict = {
        "model": model or cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # disable qwen3 thinking; ignored by non-thinking models
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if fmt:
        payload["format"] = fmt

    try:
        resp = requests.post(
            f"{cfg.ollama_base_url}/api/generate",
            json=payload,
            timeout=cfg.request_timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaError(
            f"Could not reach Ollama at {cfg.ollama_base_url}. "
            f"Is it running and is '{model or cfg.ollama_model}' pulled? ({e})"
        ) from e

    data = resp.json()
    return _strip_thinking(data.get("response", ""))


def health() -> bool:
    """True if the Ollama server responds to /api/tags."""
    cfg = get_settings()
    try:
        r = requests.get(f"{cfg.ollama_base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False
