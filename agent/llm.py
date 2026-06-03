"""VseGPT (OpenAI-compatible) client for Stage 4 agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore
    _openai_import_error = e

from utils.config import AGENT_LLM_MODEL, VSEGPT_BASE_URL

_client: Optional["OpenAI"] = None


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int


def get_client(*, base_url: str = VSEGPT_BASE_URL) -> "OpenAI":
    if OpenAI is None:  # pragma: no cover
        raise RuntimeError(
            "Python package 'openai' is required for Stage 4. "
            "Install dependencies: pip install -r requirements.txt"
        ) from _openai_import_error

    global _client
    if _client is None:
        api_key = os.getenv("VSEGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VSEGPT_API_KEY is not set (and OPENAI_API_KEY fallback is missing). "
                "Add VSEGPT_API_KEY=... to .env."
            )
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def llm_call(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 60,
    temperature: float = 0.0,
) -> Tuple[str, LLMUsage]:
    response = get_client().chat.completions.create(
        model=model or AGENT_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = (response.choices[0].message.content or "").strip()
    usage = response.usage
    return content, LLMUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )
