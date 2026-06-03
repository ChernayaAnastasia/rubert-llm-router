"""Graph nodes for Stage 4 hybrid relevance agent."""

from __future__ import annotations

import logging
from typing import Any

from agent.llm import llm_call
from agent.prompts import (
    CLASSIFICATION_PROMPT,
    SEARCH_DECISION_PROMPT,
    format_org_context,
    format_search_decision_vars,
)
from agent.search import search as tavily_search
from agent.state import AgentState
from utils.config import AGENT_LLM_MODEL

logger = logging.getLogger(__name__)


def max_confidence(proba1: float) -> float:
    p = float(proba1)
    return max(p, 1.0 - p)


def parse_label(raw: str) -> int:
    text = (raw or "").strip()
    if text in ("0", "1"):
        return int(text)
    return -1


def _add_tokens(state: AgentState, prompt_tokens: int, completion_tokens: int) -> None:
    state["prompt_tokens"] = int(state.get("prompt_tokens", 0)) + prompt_tokens
    state["completion_tokens"] = int(state.get("completion_tokens", 0)) + completion_tokens


def bert_route_node(state: AgentState) -> AgentState:
    """High-confidence BERT predictions skip LLM; low-confidence go to search decision."""
    threshold = float(state["threshold"])
    proba1 = float(state["bert_proba1"])

    if max_confidence(proba1) >= threshold:
        state["final_pred"] = int(state["bert_pred"])
        state["routed_to"] = "bert"
        state["search_used"] = False
        state["search_query"] = None
        state["search_result"] = ""
        state["next_action"] = "end"
        return state

    state["next_action"] = "decide"
    state["search_result"] = state.get("search_result", "")
    state["search_query"] = None
    state["search_used"] = False
    return state


def decide_search_node(state: AgentState) -> AgentState:
    """Ask LLM whether Tavily search is needed."""
    row = state["row"]
    model = state.get("llm_model") or AGENT_LLM_MODEL
    log = state.setdefault("log", {})

    prompt = SEARCH_DECISION_PROMPT.format(**format_search_decision_vars(row))
    try:
        decision, usage = llm_call(prompt, model=model, max_tokens=60)
        _add_tokens(state, usage.prompt_tokens, usage.completion_tokens)
        log["search_decision_prompt"] = prompt
        log["search_decision_raw"] = decision

        if decision.startswith("SEARCH:"):
            state["search_query"] = decision.replace("SEARCH:", "", 1).strip() or None
            state["next_action"] = "search"
        else:
            state["search_query"] = None
            state["next_action"] = "classify"
    except Exception as exc:
        logger.error("decide_search_node failed: %s", exc)
        log["search_decision_error"] = str(exc)
        state["search_query"] = None
        state["next_action"] = "classify"

    return state


def search_node(state: AgentState) -> AgentState:
    """Run Tavily with the query suggested by the LLM."""
    query = state.get("search_query") or ""
    use_cache = bool(state.get("use_cache", True))
    log = state.setdefault("log", {})

    try:
        result = tavily_search(query, use_cache=use_cache)
        state["search_result"] = result
        state["search_used"] = bool(result)
        log["search_query"] = query
        log["search_result_len"] = len(result)
    except Exception as exc:
        logger.error("search_node failed: %s", exc)
        log["search_error"] = str(exc)
        state["search_result"] = ""
        state["search_used"] = False

    state["next_action"] = "classify"
    return state


def classify_node(state: AgentState) -> AgentState:
    """Final LLM classification: 0 or 1."""
    row = state["row"]
    model = state.get("llm_model") or AGENT_LLM_MODEL
    log = state.setdefault("log", {})

    org_context = format_org_context(row)
    search_result = state.get("search_result") or ""
    search_context = ""
    if search_result:
        search_context = f"\nДополнительно найдено в интернете:\n{search_result}"

    prompt = CLASSIFICATION_PROMPT.format(
        org_context=org_context,
        search_context=search_context,
    )

    try:
        raw, usage = llm_call(prompt, model=model, max_tokens=5)
        _add_tokens(state, usage.prompt_tokens, usage.completion_tokens)
        log["classification_prompt"] = prompt
        log["classification_raw"] = raw

        state["final_pred"] = parse_label(raw)
        state["routed_to"] = "llm"
    except Exception as exc:
        logger.error("classify_node failed: %s", exc)
        log["classification_error"] = str(exc)
        state["final_pred"] = -1
        state["routed_to"] = "llm"

    state["next_action"] = "end"
    return state


def state_to_result(state: AgentState) -> dict[str, Any]:
    """Extract prediction dict from final graph state."""
    return {
        "final_pred": int(state.get("final_pred", -1)),
        "routed_to": state.get("routed_to", "llm"),
        "search_used": bool(state.get("search_used", False)),
        "search_query": state.get("search_query"),
        "prompt_tokens": int(state.get("prompt_tokens", 0)),
        "completion_tokens": int(state.get("completion_tokens", 0)),
    }


def make_initial_state(
    row: dict[str, Any],
    *,
    bert_proba1: float,
    bert_pred: int,
    threshold: float,
    llm_model: str | None = None,
    use_cache: bool = True,
) -> AgentState:
    return AgentState(
        row=row,
        threshold=float(threshold),
        bert_proba1=float(bert_proba1),
        bert_pred=int(bert_pred),
        llm_model=llm_model or AGENT_LLM_MODEL,
        use_cache=use_cache,
        next_action=None,
        search_query=None,
        search_result="",
        final_pred=-1,
        routed_to="llm",
        search_used=False,
        log={},
        prompt_tokens=0,
        completion_tokens=0,
    )
