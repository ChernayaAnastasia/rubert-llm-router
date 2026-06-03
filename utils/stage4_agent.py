"""
Stage 4 — Agent evaluation on low-confidence validation examples.

Implements the pipeline described in `project_spec_post_eda.md`:
  - read Stage 3 threshold from reports/stage3_error_analysis/comparison.json
  - route low-confidence examples to LLM agent (DeepSeek via VseGPT) with optional Tavily search
  - save predictions to predictions/agent_low_conf_preds.parquet
  - write a Stage 4 report to reports/stage4_agent/agent_metrics.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, total=None):  # type: ignore
        return it

from utils.bert_routing import load_confidence_threshold, max_confidence_series
from utils.metrics import eval_core, eval_subset
from utils.config import (
    AGENT_LLM_MODEL,
    AGENT_LOW_CONF_PREDS_PATH,
    AGENT_USE_CACHE,
    BERT_VAL_PREDS_PATH,
    COL_BERT_MAX_PROBA,
    COL_BERT_PRED,
    COL_BERT_PROBA1,
    COL_FINAL_PRED,
    COL_ID,
    COL_NAME,
    COL_QUERY,
    COL_RUBRIC,
    COL_ROUTED_TO,
    COL_SEARCH_USED,
    HYBRID_VAL_PREDS_PATH,
    PREDICTIONS_DIR,
    STAGE4_REPORT_PATH,
    STAGE4_REPORTS_DIR,
    TARGET,
    VAL_MERGED_PREDS_PATH,
)
from utils.stage3_error_analysis import enrich_bert_predictions

logger = logging.getLogger(__name__)

BERT_VAL_PREDS_PATH = Path(BERT_VAL_PREDS_PATH)
VAL_MERGED_PREDS_PATH = Path(VAL_MERGED_PREDS_PATH)
AGENT_PREDS_PATH = Path(AGENT_LOW_CONF_PREDS_PATH)
HYBRID_VAL_PREDS_PATH = Path(HYBRID_VAL_PREDS_PATH)
STAGE4_REPORT_PATH = Path(STAGE4_REPORT_PATH)


@dataclass(frozen=True)
class Stage4Paths:
    agent_preds_path: str
    report_path: str


def run_stage4(
    *,
    sample_n: Optional[int] = None,
    sleep_sec: float = 0.3,
    llm_model: str = AGENT_LLM_MODEL,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Run Stage 4 on low-confidence subset of val_merged_preds.

    Parameters
    ----------
    sample_n:
        If set, runs only on a random sample of low-confidence rows (debug mode).
    sleep_sec:
        Pause between requests (avoids spamming the provider).
    llm_model:
        VseGPT model id, defaults to spec.
    overwrite:
        If false and output parquet exists, raises.
    """
    from utils.agent_import import import_run_agent

    run_agent = import_run_agent()

    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(STAGE4_REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    threshold = load_confidence_threshold()
    if VAL_MERGED_PREDS_PATH.exists():
        val_merged = pd.read_parquet(VAL_MERGED_PREDS_PATH)
    elif BERT_VAL_PREDS_PATH.exists():
        val_merged = enrich_bert_predictions(pd.read_parquet(BERT_VAL_PREDS_PATH))
    else:
        raise FileNotFoundError(
            f"Missing {BERT_VAL_PREDS_PATH} (Stage 2) or {VAL_MERGED_PREDS_PATH} (Stage 3)."
        )
    if COL_BERT_PROBA1 not in val_merged.columns or COL_BERT_PRED not in val_merged.columns:
        raise ValueError(
            f"val_merged_preds.parquet must contain {COL_BERT_PROBA1} and {COL_BERT_PRED} columns."
        )
    if TARGET not in val_merged.columns:
        raise ValueError(f"val_merged_preds.parquet must contain target column: {TARGET}")

    bert_max = max_confidence_series(val_merged[COL_BERT_PROBA1])
    low_conf = val_merged[bert_max < threshold].copy()

    if sample_n is not None:
        sample_n = int(sample_n)
        if sample_n <= 0:
            raise ValueError("sample_n must be positive")
        low_conf = low_conf.sample(min(sample_n, len(low_conf)), random_state=42)

    if not overwrite and AGENT_PREDS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {AGENT_PREDS_PATH}")

    logger.info(
        "Stage 4: threshold=%.2f, low-confidence=%d / %d (%.1f%%)",
        threshold,
        len(low_conf),
        len(val_merged),
        100.0 * len(low_conf) / max(len(val_merged), 1),
    )

    results: list[dict[str, Any]] = []
    t_start = time.time()

    for _, row in tqdm(low_conf.iterrows(), total=len(low_conf)):
        t0 = time.time()
        res = run_agent(
            row.to_dict(),
            bert_proba1=float(row[COL_BERT_PROBA1]),
            bert_pred=int(row[COL_BERT_PRED]),
            threshold=float(threshold),
            llm_model=llm_model,
            use_cache=AGENT_USE_CACHE,
        )
        res[COL_ID] = row.get(COL_ID)
        res[TARGET] = int(row[TARGET])
        res["latency_sec"] = round(time.time() - t0, 3)
        results.append(res)
        if sleep_sec and sleep_sec > 0:
            time.sleep(float(sleep_sec))

    total_time = time.time() - t_start
    agent_df = pd.DataFrame(results)
    agent_df.to_parquet(AGENT_PREDS_PATH, index=False)
    logger.info("Saved %s (%d rows)", AGENT_PREDS_PATH, len(agent_df))

    valid_mask = agent_df[COL_FINAL_PRED] != -1
    agent_metrics = (
        eval_core(agent_df.loc[valid_mask, TARGET], agent_df.loc[valid_mask, COL_FINAL_PRED])
        if valid_mask.any()
        else {"accuracy": float("nan"), "macro_f1": float("nan")}
    )
    agent_acc = agent_metrics["accuracy"]

    bert_metrics = eval_core(val_merged[TARGET], val_merged[COL_BERT_PRED])
    bert_acc_val = bert_metrics["accuracy"]
    bert_f1_val = bert_metrics["macro_f1"]

    # breakdown
    search_used = agent_df[agent_df[COL_SEARCH_USED] == True]
    search_unused = agent_df[(agent_df[COL_SEARCH_USED] == False) & (agent_df[COL_ROUTED_TO] == "llm")]

    # hybrid metric on val (high-conf BERT + agent on low-conf)
    high_conf = val_merged[bert_max >= threshold].copy()
    hybrid_parts = pd.concat(
        [
            high_conf[[TARGET]].assign(**{COL_FINAL_PRED: high_conf[COL_BERT_PRED]}),
            agent_df.loc[valid_mask, [TARGET, COL_FINAL_PRED]],
        ],
        ignore_index=True,
    )
    hybrid_metrics = eval_core(hybrid_parts[TARGET], hybrid_parts[COL_FINAL_PRED])
    hybrid_acc = hybrid_metrics["accuracy"]
    hybrid_f1 = hybrid_metrics["macro_f1"]

    low_conf_bert = eval_subset(low_conf[TARGET], low_conf[COL_BERT_PRED])

    _val_meta_cols = [
        COL_ID,
        COL_QUERY,
        COL_NAME,
        COL_RUBRIC,
        TARGET,
        COL_BERT_PRED,
        COL_BERT_PROBA1,
        COL_BERT_MAX_PROBA,
    ]
    val_lookup = val_merged[[c for c in _val_meta_cols if c in val_merged.columns]].copy()
    if COL_BERT_MAX_PROBA not in val_lookup.columns:
        val_lookup[COL_BERT_MAX_PROBA] = max_confidence_series(val_lookup[COL_BERT_PROBA1])

    high_hybrid = high_conf[[COL_ID]].assign(
        **{COL_FINAL_PRED: high_conf[COL_BERT_PRED].values, COL_ROUTED_TO: "bert", COL_SEARCH_USED: False},
    )
    agent_hybrid = agent_df.loc[valid_mask, [COL_ID, COL_FINAL_PRED, COL_ROUTED_TO, COL_SEARCH_USED]]
    hybrid_val_preds = pd.concat([high_hybrid, agent_hybrid], ignore_index=True).merge(
        val_lookup, on=COL_ID, how="left"
    )
    _hybrid_out_cols = [
        c
        for c in [
            COL_ID,
            COL_QUERY,
            COL_NAME,
            COL_RUBRIC,
            TARGET,
            COL_BERT_PRED,
            COL_BERT_PROBA1,
            COL_BERT_MAX_PROBA,
            COL_FINAL_PRED,
            COL_ROUTED_TO,
            COL_SEARCH_USED,
        ]
        if c in hybrid_val_preds.columns
    ]
    hybrid_val_preds = hybrid_val_preds[_hybrid_out_cols]
    hybrid_val_preds.to_parquet(HYBRID_VAL_PREDS_PATH, index=False)
    logger.info("Saved %s (%d rows)", HYBRID_VAL_PREDS_PATH, len(hybrid_val_preds))

    # cost estimate from spec (rub per 1000 tokens)
    total_prompt_tokens = int(agent_df.get("prompt_tokens", pd.Series([0])).sum())
    total_completion_tokens = int(agent_df.get("completion_tokens", pd.Series([0])).sum())
    actual_cost_rub = round(total_prompt_tokens / 1000 * 0.04 + total_completion_tokens / 1000 * 0.08, 2)

    report: dict[str, Any] = {
        "threshold": float(threshold),
        "llm_model": f"{llm_model} (VseGPT)",
        "search_provider": "Tavily",
        "bert_only_acc": bert_acc_val,
        "bert_only_macro_f1": bert_f1_val,
        "hybrid_acc": hybrid_acc,
        "hybrid_macro_f1": hybrid_f1,
        "routed_to_agent_share": float(len(low_conf) / max(len(val_merged), 1)),
        "search_used_share": float(len(search_used) / max(len(agent_df), 1)),
        "agent_acc_on_low_conf": agent_acc,
        "bert_low_conf_acc_baseline": low_conf_bert["accuracy"],
        "bert_low_conf_f1_baseline": low_conf_bert["macro_f1"],
        "acc_with_search": float((search_used[COL_FINAL_PRED] == search_used[TARGET]).mean()) if len(search_used) else None,
        "acc_without_search": float((search_unused[COL_FINAL_PRED] == search_unused[TARGET]).mean()) if len(search_unused) else None,
        "total_time_min": round(total_time / 60.0, 2),
        "latency_mean_sec": round(float(agent_df["latency_sec"].mean()), 3) if len(agent_df) else None,
        "latency_median_sec": round(float(agent_df["latency_sec"].median()), 3) if len(agent_df) else None,
        "latency_with_search_sec": round(float(search_used["latency_sec"].median()), 3) if len(search_used) else None,
        "latency_without_search_sec": round(float(agent_df[agent_df[COL_SEARCH_USED] == False]["latency_sec"].median()), 3) if len(agent_df) else None,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "actual_cost_rub": actual_cost_rub,
        "known_limitations": [
            "LLM agent receives query, name, address, rubric, reviews, pricelist (spec v9)",
            "Routing threshold from Stage 3; BERT probs are temperature-scaled",
        ],
    }

    with open(STAGE4_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", STAGE4_REPORT_PATH)

    return {
        "threshold": float(threshold),
        "low_conf_n": int(len(low_conf)),
        "agent_acc_on_low_conf": agent_acc,
        "hybrid_acc": hybrid_acc,
        "hybrid_macro_f1": hybrid_f1,
        "paths": Stage4Paths(
            agent_preds_path=str(AGENT_PREDS_PATH),
            report_path=str(STAGE4_REPORT_PATH),
        ),
    }

