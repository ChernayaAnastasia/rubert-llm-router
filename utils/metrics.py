"""Shared evaluation metrics for all project stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils.config import TARGET_NAMES_REPORT


def eval_core(
    y_true,
    y_pred,
    *,
    zero_division: Union[int, str] = 0,
) -> dict[str, float]:
    """Return accuracy and macro-F1 rounded to 4 decimals."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=zero_division)
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
    }


def eval_subset(
    y_true,
    y_pred,
    *,
    zero_division: Union[int, str] = 0,
    include_class_dist: bool = False,
) -> dict[str, Any]:
    """
    Metrics for a subset; returns NaN when empty (Stage 3 confidence bands).
    """
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    if len(y_true_s) == 0:
        result: dict[str, Any] = {"accuracy": float("nan"), "macro_f1": float("nan")}
        if include_class_dist:
            result["class_dist"] = {}
        return result

    result = eval_core(y_true_s, y_pred_s, zero_division=zero_division)
    if include_class_dist:
        result["class_dist"] = {
            str(k): round(float(v), 4)
            for k, v in y_true_s.value_counts(normalize=True).items()
        }
    return result


def eval_binary(
    y_true,
    y_pred,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Evaluate binary classification.

    Primary:   accuracy
    Secondary: macro-F1 (class imbalance control)
    """
    core = eval_core(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_NAMES_REPORT,
        output_dict=True,
    )
    result = {
        "model": model_name,
        **core,
        "report": report,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def _comparison_category(baseline_correct: bool, final_correct: bool) -> str:
    if baseline_correct and final_correct:
        return "both_correct"
    if not baseline_correct and not final_correct:
        return "both_wrong"
    if baseline_correct and not final_correct:
        return "only_bert_correct"
    return "only_hybrid_correct"


def error_matrix(
    y_true,
    baseline_pred,
    final_pred,
) -> dict[str, Any]:
    """
    Compare baseline (BERT) vs final (hybrid) predictions on aligned rows.

    Used in Stage 5; category names match the hybrid-vs-BERT error matrix spec.
    """
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    baseline_s = pd.Series(baseline_pred).reset_index(drop=True)
    final_s = pd.Series(final_pred).reset_index(drop=True)

    baseline_correct = baseline_s == y_true_s
    final_correct = final_s == y_true_s
    categories = [
        _comparison_category(bool(b_ok), bool(h_ok))
        for b_ok, h_ok in zip(baseline_correct, final_correct)
    ]
    counts = pd.Series(categories).value_counts()
    shares = pd.Series(categories).value_counts(normalize=True).round(4)
    order = ["both_correct", "only_hybrid_correct", "only_bert_correct", "both_wrong"]
    return {
        "n_total": int(len(y_true_s)),
        "counts": {k: int(counts.get(k, 0)) for k in order},
        "shares": {k: float(shares.get(k, 0.0)) for k in order},
    }
