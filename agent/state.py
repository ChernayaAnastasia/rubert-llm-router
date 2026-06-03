"""Typed state for the Stage 4 relevance agent graph."""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    row: dict[str, Any]
    threshold: float
    bert_proba1: float
    bert_pred: int
    llm_model: str
    use_cache: bool
    next_action: Optional[str]
    search_query: Optional[str]
    search_result: str
    final_pred: int
    routed_to: str
    search_used: bool
    log: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
