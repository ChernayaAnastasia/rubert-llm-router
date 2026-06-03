"""
Stage 1 — TF-IDF + Logistic Regression reference baseline.

Trains on train_baseline.parquet, evaluates on val, writes metrics.json only.
Not used in the downstream pipeline (Stages 3–5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from utils.config import (
    COL_COMBINED_TEXT,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    STAGE1_REPORTS_DIR,
    TARGET,
    TARGET_NAMES_REPORT,
)
from utils.data_loader import attach_combined_text

logger = logging.getLogger(__name__)

METRICS_PATH = Path(STAGE1_REPORTS_DIR) / "metrics.json"


def load_baseline_splits(
    processed_dir: Optional[Path] = None,
    *,
    include_ood: bool = True,
) -> Tuple[pd.DataFrame, ...]:
    """
    Load baseline parquet splits and rebuild COL_COMBINED_TEXT on the fly.

    Returns (train_df, val_df) or (train_df, val_df, ood_df) when include_ood=True.
    """
    processed_dir = Path(processed_dir or PROCESSED_DATA_DIR)
    train_df = attach_combined_text(
        pd.read_parquet(processed_dir / "train_baseline.parquet")
    )
    val_df = attach_combined_text(
        pd.read_parquet(processed_dir / "val_baseline.parquet")
    )
    if not include_ood:
        return train_df, val_df

    ood_df = attach_combined_text(
        pd.read_parquet(processed_dir / "rel_minus_baseline.parquet")
    )
    return train_df, val_df, ood_df


def validate_train_data(train_df: pd.DataFrame) -> None:
    """Pre-training sanity checks from the project spec."""
    assert set(train_df[TARGET].unique()) == {0, 1}, (
        f"Expected labels {{0, 1}}, got {set(train_df[TARGET].unique())}"
    )
    assert len(train_df) > 20_000, f"Train too small: {len(train_df)}"


def build_tfidf_pipeline() -> Pipeline:
    """TF-IDF + LR — reference baseline (spec Stage 1.1)."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=3,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[Pipeline, float, float, str]:
    """
    Fit pipeline on train, predict on val.

    Returns (fitted pipe, accuracy, macro_f1, classification_report text).
    """
    pipe = build_tfidf_pipeline()
    pipe.fit(train_df[COL_COMBINED_TEXT].values, train_df[TARGET].values)

    y_pred = pipe.predict(val_df[COL_COMBINED_TEXT].values)
    y_true = val_df[TARGET].values

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=TARGET_NAMES_REPORT
    )
    return pipe, float(acc), float(macro_f1), report


def save_metrics_report(
    acc: float,
    macro_f1: float,
    *,
    output_path: Optional[Path] = None,
) -> Path:
    """Write reports/stage1_baseline/metrics.json (spec Stage 1.2)."""
    output_path = Path(output_path or METRICS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": "TF-IDF + LogisticRegression (reference only)",
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "note": "Lower bound. Not used in downstream pipeline.",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Metrics saved → %s", output_path)
    return output_path


def run_stage1(
    *,
    processed_dir: Optional[Path] = None,
) -> dict:
    """
    Execute Stage 1: TF-IDF reference metrics on val.

    Model and predictions are not persisted (spec v8).
    """
    Path(STAGE1_REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_baseline_splits(processed_dir, include_ood=False)
    validate_train_data(train_df)

    logger.info("Data loaded — train: %d, val: %d", len(train_df), len(val_df))

    _pipe, acc, macro_f1, report = train_and_evaluate(train_df, val_df)
    metrics_path = save_metrics_report(acc, macro_f1)

    print(f"TF-IDF  Val Accuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}")
    print(report)

    summary = {
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "metrics_path": str(metrics_path),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }
    logger.info(
        "Stage 1 complete — accuracy=%.4f, macro-F1=%.4f",
        summary["val_accuracy"],
        summary["val_macro_f1"],
    )
    return summary
