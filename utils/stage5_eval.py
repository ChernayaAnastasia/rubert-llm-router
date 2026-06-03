"""
Stage 5 — thin facade over split modules.

Prefer direct imports:
  - Stage 5A (BERT, no langchain): ``utils.stage5_bert``
  - Stage 5B (agent):              ``utils.stage5_agent``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from utils.config import AGENT_LLM_MODEL
from utils.stage5_agent import Stage5Paths, run_stage5_agent
from utils.stage5_bert import load_eval_df, run_stage5_bert_inference

__all__ = [
    "Stage5Paths",
    "load_eval_df",
    "run_stage5",
    "run_stage5_agent",
    "run_stage5_bert_inference",
]


def run_stage5(
    *,
    sample_n: Optional[int] = None,
    sleep_sec: float = 0.0,
    llm_model: str = AGENT_LLM_MODEL,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Full Stage 5: BERT inference then agent (single machine, debug only)."""
    run_stage5_bert_inference(sample_n=sample_n, overwrite=overwrite)
    return run_stage5_agent(
        sample_n=sample_n,
        sleep_sec=sleep_sec,
        llm_model=llm_model,
        overwrite=overwrite,
    )
