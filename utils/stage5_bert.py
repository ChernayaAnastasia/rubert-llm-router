"""
Stage 5A — BERT cross-encoder inference on locked eval_baseline.parquet.

No agent, langchain, or langgraph imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.bert_routing import max_confidence_series
from utils.config import (
    BERT_CALIBRATION_PATH,
    BERT_EVAL_PREDS_PATH,
    COL_ADDRESS,
    COL_BERT_MAX_PROBA,
    COL_BERT_PRED,
    COL_BERT_PROBA1,
    COL_ID,
    COL_NAME,
    COL_ORG_TEXT,
    COL_PRICELIST,
    COL_QUERY,
    COL_REVIEWS,
    COL_RUBRIC,
    EVAL_BASELINE_PATH,
    PREDICTIONS_DIR,
    TARGET,
)
from utils.data_loader import make_org_text
from utils.predict import predict_bert

logger = logging.getLogger(__name__)

EVAL_BASELINE_PATH = Path(EVAL_BASELINE_PATH)
BERT_EVAL_PREDS_PATH = Path(BERT_EVAL_PREDS_PATH)

_REQUIRED_EVAL_COLS = [
    COL_ID,
    COL_QUERY,
    COL_NAME,
    COL_ADDRESS,
    COL_RUBRIC,
    COL_REVIEWS,
    COL_PRICELIST,
    TARGET,
]


def load_eval_df(*, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Load eval_baseline.parquet and attach org_text."""
    if not EVAL_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EVAL_BASELINE_PATH}. Run EDA to create eval_baseline.parquet."
        )
    eval_df = pd.read_parquet(EVAL_BASELINE_PATH).reset_index(drop=True)
    missing = [c for c in _REQUIRED_EVAL_COLS if c not in eval_df.columns]
    if missing:
        raise ValueError(f"eval_baseline.parquet missing columns: {missing}")
    eval_df[COL_ORG_TEXT] = make_org_text(eval_df)
    if sample_n is not None:
        sample_n = int(sample_n)
        if sample_n <= 0:
            raise ValueError("sample_n must be positive")
        eval_df = eval_df.sample(min(sample_n, len(eval_df)), random_state=42).reset_index(drop=True)
    return eval_df


def run_stage5_bert_inference(
    *,
    sample_n: Optional[int] = None,
    overwrite: bool = True,
    output_path: Optional[Path] = None,
    batch_size: int = 64,
) -> Path:
    """
    Stage 5A — BERT cross-encoder inference on eval (Colab GPU or local).

    Saves ``bert_eval_preds.parquet`` with eval columns + bert_pred / bert_proba1 /
    bert_max_proba. Run ``utils.stage5_agent.run_stage5_agent()`` locally afterwards.
    """
    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_path or BERT_EVAL_PREDS_PATH)

    if not Path(BERT_CALIBRATION_PATH).exists():
        raise FileNotFoundError(
            f"Missing {BERT_CALIBRATION_PATH}. Run Stage 2 calibration first."
        )
    if not overwrite and out_path.exists():
        raise FileExistsError(f"Refusing to overwrite {out_path}")

    eval_df = load_eval_df(sample_n=sample_n)

    logger.info("Stage 5A: BERT cross-encoder inference on %d eval rows", len(eval_df))
    bert_out = predict_bert(
        eval_df[COL_QUERY].astype(str).tolist(),
        eval_df[COL_ORG_TEXT].astype(str).tolist(),
        batch_size=batch_size,
    )
    eval_df[COL_BERT_PRED] = bert_out["pred"]
    eval_df[COL_BERT_PROBA1] = bert_out["proba1"]
    eval_df[COL_BERT_MAX_PROBA] = max_confidence_series(eval_df[COL_BERT_PROBA1])

    eval_df.to_parquet(out_path, index=False)
    logger.info("Saved %s (%d rows)", out_path, len(eval_df))
    return out_path
