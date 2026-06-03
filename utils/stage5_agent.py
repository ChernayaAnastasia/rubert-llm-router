"""
Stage 5B — Agent loop on precomputed BERT eval predictions + hybrid metrics.

Imports the LLM agent lazily inside ``run_stage5_agent()`` only.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, total=None):  # type: ignore
        return it

from utils.bert_routing import load_confidence_threshold
from utils.config import (
    AGENT_EVAL_PREDS_PATH,
    AGENT_LLM_MODEL,
    AGENT_USE_CACHE,
    BERT_EVAL_PREDS_PATH,
    COL_BERT_MAX_PROBA,
    COL_BERT_PRED,
    COL_BERT_PROBA1,
    COL_FINAL_PRED,
    COL_ID,
    COL_ROUTED_TO,
    COL_SEARCH_USED,
    ERROR_MATRIX_FIG_PATH,
    ERROR_MATRIX_PATH,
    FINAL_EVAL_DIR,
    FINAL_METRICS_PATH,
    PREDICTIONS_DIR,
    TARGET,
)
from utils.metrics import error_matrix, eval_core, eval_subset

logger = logging.getLogger(__name__)

BERT_EVAL_PREDS_PATH = Path(BERT_EVAL_PREDS_PATH)
AGENT_EVAL_PREDS_PATH = Path(AGENT_EVAL_PREDS_PATH)
FINAL_EVAL_DIR = Path(FINAL_EVAL_DIR)
FINAL_METRICS_PATH = Path(FINAL_METRICS_PATH)
ERROR_MATRIX_PATH = Path(ERROR_MATRIX_PATH)
ERROR_MATRIX_FIG_PATH = Path(ERROR_MATRIX_FIG_PATH)

_BERT_PRED_COLS = [COL_BERT_PRED, COL_BERT_PROBA1, COL_BERT_MAX_PROBA]


@dataclass(frozen=True)
class Stage5Paths:
    agent_preds_path: str
    metrics_path: str
    error_matrix_path: str
    fig_path: str


def _plot_hybrid_vs_bert(
    metrics: dict[str, Any],
    matrix: dict[str, Any],
    output_path: Path = ERROR_MATRIX_FIG_PATH,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    models = ["BERT-only", "Hybrid"]
    accs = [metrics["bert_only"]["accuracy"], metrics["hybrid"]["accuracy"]]
    f1s = [metrics["bert_only"]["macro_f1"], metrics["hybrid"]["macro_f1"]]
    x = range(len(models))
    width = 0.35

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], accs, width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], f1s, width, label="Macro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.set_title("Eval metrics: BERT-only vs Hybrid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels = list(matrix["counts"].keys())
    counts = [matrix["counts"][k] for k in labels]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    ax.bar(labels, counts, color=colors)
    ax.set_title("Error matrix: Hybrid vs BERT (eval)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    for i, (label, count) in enumerate(zip(labels, counts)):
        share = matrix["shares"].get(label, 0.0)
        ax.text(i, count, f"{share:.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def run_stage5_agent(
    *,
    bert_preds_path: Optional[Path] = None,
    sample_n: Optional[int] = None,
    sleep_sec: float = 0.0,
    llm_model: str = AGENT_LLM_MODEL,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Stage 5B — Agent loop on precomputed BERT predictions (local).

    Loads ``bert_eval_preds.parquet`` (from Stage 5A / Colab), routes low-confidence
    rows to the LLM agent, computes hybrid metrics and error matrix.
    """
    from utils.agent_import import import_run_agent

    run_agent = import_run_agent()

    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    FINAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    preds_path = Path(bert_preds_path or BERT_EVAL_PREDS_PATH)
    if not preds_path.exists():
        raise FileNotFoundError(
            f"Missing {preds_path}. Run Stage 5A (notebooks/stage5a_bert_inference.ipynb "
            "on Colab) or utils.stage5_bert.run_stage5_bert_inference() first."
        )
    if not overwrite and AGENT_EVAL_PREDS_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {AGENT_EVAL_PREDS_PATH}")

    eval_df = pd.read_parquet(preds_path).reset_index(drop=True)
    missing_bert = [c for c in _BERT_PRED_COLS if c not in eval_df.columns]
    if missing_bert:
        raise ValueError(f"{preds_path} missing BERT columns: {missing_bert}")
    if TARGET not in eval_df.columns:
        raise ValueError(f"{preds_path} missing target column: {TARGET}")

    if sample_n is not None:
        sample_n = int(sample_n)
        if sample_n <= 0:
            raise ValueError("sample_n must be positive")
        eval_df = eval_df.sample(min(sample_n, len(eval_df)), random_state=42).reset_index(drop=True)

    threshold = load_confidence_threshold()
    low_conf_mask = eval_df[COL_BERT_MAX_PROBA] < threshold
    low_conf_share = float(low_conf_mask.mean())
    low_conf_n = int(low_conf_mask.sum())
    high_conf_n = len(eval_df) - low_conf_n

    logger.info(
        "Stage 5B: threshold=%.2f, low-confidence=%d / %d (%.1f%%)",
        threshold,
        low_conf_n,
        len(eval_df),
        100.0 * low_conf_share,
    )
    logger.info(
        "Stage 5B: run_agent on %d low-conf rows (%d high-conf → BERT, skipped)",
        low_conf_n,
        high_conf_n,
    )

    low_conf_df = eval_df.loc[low_conf_mask]
    agent_low_results: dict[int, dict[str, Any]] = {}
    t_start = time.time()

    low_conf_records = low_conf_df.to_dict("records")
    for row_idx, row in zip(
        low_conf_df.index,
        tqdm(low_conf_records, total=len(low_conf_records), desc="Agent eval (low-conf)"),
    ):
        t0 = time.time()
        res = run_agent(
            row,
            bert_proba1=float(row[COL_BERT_PROBA1]),
            bert_pred=int(row[COL_BERT_PRED]),
            threshold=float(threshold),
            llm_model=llm_model,
            use_cache=AGENT_USE_CACHE,
        )
        res[COL_ID] = row.get(COL_ID)
        res[TARGET] = int(row[TARGET])
        res["latency_sec"] = round(time.time() - t0, 3)
        agent_low_results[int(row_idx)] = res
        if sleep_sec and sleep_sec > 0:
            time.sleep(float(sleep_sec))

    total_time = time.time() - t_start

    high_conf_mask = ~low_conf_mask
    agent_parts: list[pd.DataFrame] = []
    if agent_low_results:
        agent_parts.append(pd.DataFrame.from_dict(agent_low_results, orient="index"))
    if high_conf_mask.any():
        agent_parts.append(
            pd.DataFrame(
                {
                    COL_FINAL_PRED: eval_df.loc[high_conf_mask, COL_BERT_PRED].astype(int),
                    COL_ROUTED_TO: "bert",
                    COL_SEARCH_USED: False,
                    "search_query": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    COL_ID: eval_df.loc[high_conf_mask, COL_ID],
                    TARGET: eval_df.loc[high_conf_mask, TARGET].astype(int),
                    "latency_sec": 0.0,
                },
                index=eval_df.index[high_conf_mask],
            )
        )
    agent_df = pd.concat(agent_parts).sort_index().reset_index(drop=True)
    agent_df.to_parquet(AGENT_EVAL_PREDS_PATH, index=False)
    logger.info(
        "Saved %s (%d rows, low-conf agent loop %.1f min)",
        AGENT_EVAL_PREDS_PATH,
        len(agent_df),
        total_time / 60.0,
    )

    y_true = eval_df[TARGET].to_numpy()
    bert_pred = eval_df[COL_BERT_PRED].to_numpy()

    valid_mask = agent_df[COL_FINAL_PRED].to_numpy() != -1
    if not valid_mask.all():
        logger.warning(
            "Hybrid parse errors (%s=-1): %d / %d",
            COL_FINAL_PRED,
            int((~valid_mask).sum()),
            len(agent_df),
        )

    hybrid_pred = agent_df.loc[valid_mask, COL_FINAL_PRED].to_numpy()
    y_hybrid = agent_df.loc[valid_mask, TARGET].to_numpy()

    bert_metrics = eval_core(y_true, bert_pred)
    hybrid_metrics = eval_core(y_hybrid, hybrid_pred)

    low_conf_df = eval_df.loc[low_conf_mask]
    agent_low = agent_df.loc[low_conf_mask.values]
    agent_valid_low = agent_low[COL_FINAL_PRED] != -1
    agent_on_low_conf = (
        eval_subset(
            agent_low.loc[agent_valid_low, TARGET],
            agent_low.loc[agent_valid_low, COL_FINAL_PRED],
        )
        if agent_valid_low.any()
        else {"accuracy": float("nan"), "macro_f1": float("nan")}
    )
    bert_on_low_conf = eval_subset(
        low_conf_df[TARGET],
        low_conf_df[COL_BERT_PRED],
    )

    search_used = agent_df[agent_df[COL_SEARCH_USED] == True]
    search_unused = agent_df[
        (agent_df[COL_SEARCH_USED] == False) & (agent_df[COL_ROUTED_TO] == "llm")
    ]

    matrix = error_matrix(
        y_true[valid_mask],
        bert_pred[valid_mask],
        hybrid_pred,
    )

    final_metrics: dict[str, Any] = {
        "bert_only": {
            "accuracy": bert_metrics["accuracy"],
            "macro_f1": bert_metrics["macro_f1"],
        },
        "hybrid": {
            "accuracy": hybrid_metrics["accuracy"],
            "macro_f1": hybrid_metrics["macro_f1"],
        },
        "threshold": float(threshold),
        "eval_size": int(len(eval_df)),
        "low_conf_share": round(low_conf_share, 4),
        "routed_to_agent_share": round(low_conf_share, 4),
        "agent_acc_on_low_conf": agent_on_low_conf["accuracy"],
        "bert_low_conf_acc_baseline": bert_on_low_conf["accuracy"],
        "bert_low_conf_f1_baseline": bert_on_low_conf["macro_f1"],
        "search_used_share": round(float(len(search_used) / max(len(agent_df), 1)), 4),
        "acc_with_search": round(
            float((search_used[COL_FINAL_PRED] == search_used[TARGET]).mean()), 4
        )
        if len(search_used)
        else None,
        "acc_without_search": round(
            float((search_unused[COL_FINAL_PRED] == search_unused[TARGET]).mean()), 4
        )
        if len(search_unused)
        else None,
        "parse_error_share": round(float((~valid_mask).mean()), 4),
        "total_time_min": round(total_time / 60.0, 2),
        "known_limitations": [
            "Temperature scaling fitted on val — slight optimistic bias on eval",
            "CONFIDENCE_THRESHOLD chosen on val (Stage 3)",
        ],
    }

    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", FINAL_METRICS_PATH)

    with open(ERROR_MATRIX_PATH, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", ERROR_MATRIX_PATH)

    _plot_hybrid_vs_bert(final_metrics, matrix)

    return {
        "threshold": float(threshold),
        "eval_size": int(len(eval_df)),
        "low_conf_share": low_conf_share,
        "bert_acc": bert_metrics["accuracy"],
        "bert_macro_f1": bert_metrics["macro_f1"],
        "hybrid_acc": hybrid_metrics["accuracy"],
        "hybrid_macro_f1": hybrid_metrics["macro_f1"],
        "error_matrix": matrix,
        "paths": Stage5Paths(
            agent_preds_path=str(AGENT_EVAL_PREDS_PATH),
            metrics_path=str(FINAL_METRICS_PATH),
            error_matrix_path=str(ERROR_MATRIX_PATH),
            fig_path=str(ERROR_MATRIX_FIG_PATH),
        ),
    }
