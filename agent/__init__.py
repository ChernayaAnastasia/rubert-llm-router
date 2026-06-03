"""Stage 4 hybrid agent: BERT routing + LLM + optional Tavily search."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["build_agent_graph", "run_agent"]

if TYPE_CHECKING:
    from agent.graph import build_agent_graph, run_agent


def __getattr__(name: str):
    if name in __all__:
        from agent.graph import build_agent_graph, run_agent

        return build_agent_graph if name == "build_agent_graph" else run_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
