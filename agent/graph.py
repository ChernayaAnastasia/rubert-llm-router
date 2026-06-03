"""LangGraph assembly for Stage 4 hybrid relevance agent."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes import (
    bert_route_node,
    classify_node,
    decide_search_node,
    make_initial_state,
    search_node,
    state_to_result,
)
from agent.state import AgentState

_graph = None


def _route_bert(state: AgentState) -> str:
    return "end" if state.get("next_action") == "end" else "decide"


def _route_decide(state: AgentState) -> str:
    return "search" if state.get("next_action") == "search" else "classify"


def build_agent_graph():
    """Compile and cache the relevance agent graph."""
    global _graph
    if _graph is not None:
        return _graph

    builder = StateGraph(AgentState)
    builder.add_node("bert_route", bert_route_node)
    builder.add_node("decide_search", decide_search_node)
    builder.add_node("search", search_node)
    builder.add_node("classify", classify_node)

    builder.set_entry_point("bert_route")
    builder.add_conditional_edges(
        "bert_route",
        _route_bert,
        {"end": END, "decide": "decide_search"},
    )
    builder.add_conditional_edges(
        "decide_search",
        _route_decide,
        {"search": "search", "classify": "classify"},
    )
    builder.add_edge("search", "classify")
    builder.add_edge("classify", END)

    _graph = builder.compile()
    return _graph


def run_agent(
    row: dict,
    *,
    bert_proba1: float,
    bert_pred: int,
    threshold: float,
    llm_model: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Run one example through the agent graph and return prediction fields."""
    graph = build_agent_graph()
    initial = make_initial_state(
        row,
        bert_proba1=bert_proba1,
        bert_pred=bert_pred,
        threshold=threshold,
        llm_model=llm_model,
        use_cache=use_cache,
    )
    final = graph.invoke(initial)
    return state_to_result(final)
