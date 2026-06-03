"""
Stage 3 — Error analysis on calibrated BERT predictions (spec v10).

Uses BERT val/OOD parquets + TF-IDF reference metrics from Stage 1 metrics.json.
Selects CONFIDENCE_THRESHOLD and error taxonomy for the agent architecture.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.config import (
    BERT_OOD_PREDS_PATH,
    BERT_VAL_PREDS_PATH,
    COL_BERT_CORRECT,
    COL_BERT_MAX_PROBA,
    COL_BERT_PRED,
    COL_BERT_PROBA1,
    COL_ID,
    COL_NAME,
    COL_QUERY,
    COL_RUBRIC,
    CONFIDENCE_THRESHOLD_DEFAULT,
    PREDICTIONS_DIR,
    BERT_CALIBRATION_PATH,
    STAGE1_REPORTS_DIR,
    STAGE3_ACCURACY_COVERAGE_FIG_PATH,
    STAGE3_BERT_ERRORS_SAMPLE_PATH,
    STAGE3_COMPARISON_PATH,
    RANDOM_STATE,
    TARGET,
    VAL_MERGED_PREDS_PATH,
)
from utils.metrics import eval_core, eval_subset

logger = logging.getLogger(__name__)

COMPARISON_PATH = Path(STAGE3_COMPARISON_PATH)
ACCURACY_COVERAGE_FIG = Path(STAGE3_ACCURACY_COVERAGE_FIG_PATH)
BERT_ERRORS_SAMPLE_PATH = Path(STAGE3_BERT_ERRORS_SAMPLE_PATH)

MIN_COVERAGE_TARGET = 0.70
ACCURACY_LIFT_PP = 0.05
SEARCH_ARCHITECTURE_BOUNDARY = 0.30

ERROR_TAXONOMY_KEYS = (
    "requires_search",
    "hard_semantic",
    "fact_verification",
    "label_noise",
    "other",
)


def max_confidence(proba1: float) -> float:
    """max(p, 1-p) for binary P(class=1) after temperature scaling."""
    return max(float(proba1), 1.0 - float(proba1))


def enrich_bert_predictions(bert_val: pd.DataFrame) -> pd.DataFrame:
    """Add COL_BERT_CORRECT and COL_BERT_MAX_PROBA if missing."""
    out = bert_val.copy()
    if COL_BERT_CORRECT not in out.columns:
        out[COL_BERT_CORRECT] = out[COL_BERT_PRED] == out[TARGET]
    out[COL_BERT_MAX_PROBA] = out[COL_BERT_PROBA1].map(max_confidence)
    return out


def load_bert_val_predictions(
    bert_path: Optional[Path] = None,
) -> pd.DataFrame:
    bert_path = Path(bert_path or BERT_VAL_PREDS_PATH)
    if not bert_path.exists():
        raise FileNotFoundError(
            f"BERT val preds not found: {bert_path}. Run Stage 2 first."
        )
    bert_val = pd.read_parquet(bert_path)
    for col in (COL_BERT_PRED, COL_BERT_PROBA1, TARGET):
        if col not in bert_val.columns:
            raise ValueError(f"BERT val preds missing column: {col}")
    return enrich_bert_predictions(bert_val)


def load_tfidf_reference_metrics(
    metrics_path: Optional[Path] = None,
) -> dict[str, float]:
    """TF-IDF reference metrics from Stage 1 (no per-row preds)."""
    metrics_path = Path(metrics_path or Path(STAGE1_REPORTS_DIR) / "metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Stage 1 metrics not found: {metrics_path}. Run scripts/run_stage1.py first."
        )
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "val_accuracy": float(data["val_accuracy"]),
        "val_macro_f1": float(data["val_macro_f1"]),
    }


def load_calibration_summary(
    calib_path: Optional[Path] = None,
) -> dict[str, Any]:
    calib_path = Path(calib_path or BERT_CALIBRATION_PATH)
    if not calib_path.exists():
        logger.warning("Calibration file not found: %s", calib_path)
        return {}
    try:
        # Prefer shared helper from calibration.py per spec.
        from utils.calibration import load_calibration  # local import: avoids hard torch dependency at module import time

        calib = load_calibration(calib_path)
    except ModuleNotFoundError:
        # Stage 3 should run on CPU-only env without torch installed.
        with open(calib_path, encoding="utf-8") as f:
            calib = json.load(f)
    return {
        "temperature": float(calib.get("temperature", 1.0)),
        "ece_before": calib.get("ece_before"),
        "ece_after": calib.get("ece_after"),
    }


def bert_errors(bert_val: pd.DataFrame) -> pd.DataFrame:
    return bert_val[~bert_val[COL_BERT_CORRECT]].copy()


def searchable_share(taxonomy: dict[str, int]) -> float:
    actionable = sum(v for k, v in taxonomy.items() if k != "label_noise")
    if actionable == 0:
        return 0.0
    return taxonomy.get("requires_search", 0) / actionable


def agent_architecture_recommendation(
    share: float, threshold: float = SEARCH_ARCHITECTURE_BOUNDARY
) -> str:
    if share > threshold:
        return "BERT -> LLM + Tavily search"
    return "BERT -> LLM (no search)"


def min_sample_size_for_taxonomy(
    *,
    boundary: float = SEARCH_ARCHITECTURE_BOUNDARY,
    margin: float = 0.10,
) -> int:
    """
    Minimum manual labels for error taxonomy (Cochran, 95% CI, ±margin pp).

    Uses boundary=0.30 (search vs no-search decision) and margin=0.10 → n≈96.
    Not the full val error count: we estimate shares in taxonomy categories, not
    re-label every misclassified row.
    """
    z = float(norm.ppf(0.975))
    p = max(boundary, 0.5)
    return int(np.ceil((z**2) * p * (1 - p) / (margin**2)))


def suggest_confidence_threshold(
    bert_val: pd.DataFrame,
    *,
    min_coverage: float = MIN_COVERAGE_TARGET,
    accuracy_lift: float = ACCURACY_LIFT_PP,
) -> float:
    """
    Spec 3.2: first threshold with coverage >= min_coverage and
    accuracy on confident examples >= overall accuracy + accuracy_lift.
    """
    bert_max = bert_val[COL_BERT_MAX_PROBA]
    overall_acc = float(bert_val[COL_BERT_CORRECT].mean())
    target_acc = overall_acc + accuracy_lift

    thresholds = np.arange(0.50, 0.96, 0.01)
    for thr in thresholds:
        mask = bert_max >= thr
        if mask.sum() == 0:
            break
        cov = float(mask.mean())
        if cov < min_coverage:
            continue
        acc = float(
            (bert_val.loc[mask, COL_BERT_PRED] == bert_val.loc[mask, TARGET]).mean()
        )
        if acc >= target_acc:
            return round(float(thr), 2)

    return float(CONFIDENCE_THRESHOLD_DEFAULT)


def accuracy_coverage_data(
    bert_val: pd.DataFrame,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    thresholds = (
        thresholds if thresholds is not None else np.arange(0.50, 0.96, 0.01)
    )
    bert_max = bert_val[COL_BERT_MAX_PROBA]
    accs: list[float] = []
    coverages: list[float] = []

    for thr in thresholds:
        mask = bert_max >= thr
        if mask.sum() == 0:
            break
        accs.append(
            float(
                (bert_val.loc[mask, COL_BERT_PRED] == bert_val.loc[mask, TARGET]).mean()
            )
        )
        coverages.append(float(mask.mean()))

    return thresholds[: len(accs)], accs, coverages


def plot_accuracy_coverage(
    bert_val: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Spec 3.2: accuracy vs coverage + coverage vs threshold."""
    output_path = Path(output_path or ACCURACY_COVERAGE_FIG)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall_acc = float(bert_val[COL_BERT_CORRECT].mean())
    thr_list, accs, coverages = accuracy_coverage_data(bert_val)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(coverages, accs, marker="o")
    ax1.axhline(
        overall_acc + ACCURACY_LIFT_PP,
        color="red",
        linestyle="--",
        label=f"overall + {ACCURACY_LIFT_PP:.0%}pp = {overall_acc + ACCURACY_LIFT_PP:.3f}",
    )
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Accuracy on confident examples")
    ax1.set_title("Cross-encoder (T-scaled): accuracy vs. coverage")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(thr_list, coverages, marker="o", color="orange")
    ax2.axhline(
        MIN_COVERAGE_TARGET,
        color="red",
        linestyle="--",
        label=f"coverage = {MIN_COVERAGE_TARGET:.0%}",
    )
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Coverage")
    ax2.set_title("Coverage vs. threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def low_confidence_summary(
    bert_val: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    mask_low = bert_val[COL_BERT_MAX_PROBA] < threshold
    low, high = bert_val[mask_low], bert_val[~mask_low]

    lm = eval_subset(
        low[TARGET], low[COL_BERT_PRED], zero_division=0, include_class_dist=True
    )
    hm = eval_subset(
        high[TARGET], high[COL_BERT_PRED], zero_division=0, include_class_dist=True
    )

    low_pos = lm["class_dist"].get("1", 0.0)
    high_pos = hm["class_dist"].get("1", 0.0)
    if abs(low_pos - high_pos) > 0.10:
        logger.warning(
            "Class distribution shift between confidence bands: "
            "low_conf pos_rate=%.3f, high_conf pos_rate=%.3f",
            low_pos,
            high_pos,
        )

    return {
        "threshold": threshold,
        "low_confidence_n": int(mask_low.sum()),
        "low_confidence_pct": round(float(mask_low.mean()) * 100, 2),
        "low_confidence_accuracy": lm["accuracy"],
        "low_confidence_macro_f1": lm["macro_f1"],
        "high_confidence_accuracy": hm["accuracy"],
        "high_confidence_macro_f1": hm["macro_f1"],
    }


def load_bert_ood_predictions(
    bert_path: Optional[Path] = None,
) -> pd.DataFrame:
    bert_path = Path(bert_path or BERT_OOD_PREDS_PATH)
    if not bert_path.exists():
        raise FileNotFoundError(
            f"BERT OOD preds not found: {bert_path}. Run Stage 2 first."
        )
    return pd.read_parquet(bert_path)


def ood_summary(
    bert_ood: pd.DataFrame,
    bert_val: pd.DataFrame,
) -> dict[str, Any]:
    """Spec 3.3: BERT-only OOD sanity check."""
    ood_median = float(bert_ood[COL_BERT_PROBA1].median())
    val_median = float(bert_val[COL_BERT_PROBA1].median())
    shift = round(ood_median - val_median, 4)

    if shift < -0.05:
        logger.warning(
            "OOD confidence shift=%.4f — possible distributional shift.",
            shift,
        )

    return {
        "n_ood": int(len(bert_ood)),
        "bert_pred_dist": {
            str(k): round(float(v), 4)
            for k, v in bert_ood[COL_BERT_PRED].value_counts(normalize=True).items()
        },
        "bert_proba1_median_ood": round(ood_median, 4),
        "bert_proba1_median_val": round(val_median, 4),
        "confidence_shift": shift,
    }


def build_error_taxonomy(
    errors: pd.DataFrame,
    *,
    manual_counts: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    if manual_counts is not None:
        return {k: int(manual_counts.get(k, 0)) for k in ERROR_TAXONOMY_KEYS}

    taxonomy = {k: 0 for k in ERROR_TAXONOMY_KEYS}
    taxonomy["other"] = int(len(errors))
    return taxonomy


def build_taxonomy_labeling_sample(
    errors: pd.DataFrame,
    *,
    n: Optional[int] = None,
    high_conf_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Stratified error sample for manual taxonomy (spec v10 §3.1).

    Size n defaults to min_sample_size_for_taxonomy() (~96): Cochran formula for
    estimating a proportion near the search/no-search boundary (30%) with ±10 pp
    at 95% CI. Per class: prioritize high-confidence errors (model confident but
    wrong), then fill to n//2 from the rest.
    """
    n = min_sample_size_for_taxonomy() if n is None else n
    n_per_class = max(1, n // 2)

    if COL_BERT_MAX_PROBA not in errors.columns:
        errors = errors.copy()
        errors[COL_BERT_MAX_PROBA] = errors[COL_BERT_PROBA1].map(max_confidence)

    high_conf_errors = errors[errors[COL_BERT_MAX_PROBA] >= high_conf_threshold]

    parts: list[pd.DataFrame] = []
    for label_val in errors[TARGET].unique():
        hc = high_conf_errors[high_conf_errors[TARGET] == label_val]
        rest = errors[
            (errors[TARGET] == label_val) & (~errors.index.isin(hc.index))
        ]
        hc_sample = hc.sample(min(n_per_class, len(hc)), random_state=random_state)
        rest_needed = n_per_class - len(hc_sample)
        rest_sample = (
            rest.sample(min(rest_needed, len(rest)), random_state=random_state)
            if rest_needed > 0 and len(rest)
            else rest.iloc[0:0]
        )
        parts.append(pd.concat([hc_sample, rest_sample]))

    sample = (
        pd.concat(parts)
        .drop_duplicates()
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    n_high = int((sample[COL_BERT_MAX_PROBA] >= high_conf_threshold).sum())
    logger.info(
        "Taxonomy sample: n=%d (target %d, Cochran), high-conf=%d, by class:\n%s",
        len(sample),
        n,
        n_high,
        sample[TARGET].value_counts().to_string(),
    )
    return sample


def save_bert_errors_sample(
    errors: pd.DataFrame,
    output_path: Optional[Path] = None,
    n: Optional[int] = None,
    random_state: int = RANDOM_STATE,
) -> Path:
    output_path = Path(output_path or BERT_ERRORS_SAMPLE_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_target = min_sample_size_for_taxonomy() if n is None else n
    if len(errors) < n_target:
        logger.warning(
            "BERT errors: %d rows, fewer than recommended taxonomy sample n=%d",
            len(errors),
            n_target,
        )

    cols = [
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
        ]
        if c in errors.columns
    ]
    cols = list(dict.fromkeys(cols))

    sample = build_taxonomy_labeling_sample(
        errors, n=n, random_state=random_state
    )[cols]
    sample.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %s (%d rows)", output_path, len(sample))
    return output_path


def build_comparison_report(
    *,
    confidence_threshold: float,
    temperature: Optional[float],
    error_taxonomy: dict[str, int],
    tfidf_metrics: dict[str, float],
    bert_val: pd.DataFrame,
    low_conf_stats: dict[str, Any],
    ood_stats: dict[str, Any],
) -> dict[str, Any]:
    share = searchable_share(error_taxonomy)
    bert_metrics = eval_core(bert_val[TARGET], bert_val[COL_BERT_PRED])

    report: dict[str, Any] = {
        "confidence_threshold": float(confidence_threshold),
        "error_taxonomy": error_taxonomy,
        "searchable_share": round(share, 3),
        "agent_architecture": agent_architecture_recommendation(share),
        "low_confidence": low_conf_stats,
        "ood": ood_stats,
        "TF-IDF + LR (reference)": {
            "val_accuracy": tfidf_metrics["val_accuracy"],
            "val_macro_f1": tfidf_metrics["val_macro_f1"],
        },
        "bert cross-encoder (T-scaled)": {
            "val_accuracy": bert_metrics["accuracy"],
            "val_macro_f1": bert_metrics["macro_f1"],
        },
    }
    if temperature is not None:
        report["temperature"] = float(temperature)
    return report


def run_stage3(
    *,
    confidence_threshold: Optional[float] = None,
    error_taxonomy: Optional[dict[str, int]] = None,
    bert_val_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Run full Stage 3 pipeline (spec v9).

    If ``error_taxonomy`` is None, sets all errors to ``other`` (refine manually
    in the notebook after reviewing bert_errors_sample.csv).
    """
    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)

    bert_val = load_bert_val_predictions(bert_val_path)
    tfidf_metrics = load_tfidf_reference_metrics()
    calib = load_calibration_summary()

    errors = bert_errors(bert_val)
    taxonomy = build_error_taxonomy(errors, manual_counts=error_taxonomy)

    suggested_thr = suggest_confidence_threshold(bert_val)
    threshold = (
        confidence_threshold if confidence_threshold is not None else suggested_thr
    )

    plot_accuracy_coverage(bert_val)
    save_bert_errors_sample(errors)

    bert_ood = load_bert_ood_predictions()
    ood_stats = ood_summary(bert_ood, bert_val)
    low_conf = low_confidence_summary(bert_val, threshold)

    bert_val.to_parquet(VAL_MERGED_PREDS_PATH, index=False)
    logger.info("Saved %s (BERT val for Stage 4 routing)", VAL_MERGED_PREDS_PATH)

    comparison = build_comparison_report(
        confidence_threshold=threshold,
        temperature=calib.get("temperature"),
        error_taxonomy=taxonomy,
        tfidf_metrics=tfidf_metrics,
        bert_val=bert_val,
        low_conf_stats=low_conf,
        ood_stats=ood_stats,
    )
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info(
        "Stage 3 complete — threshold=%.2f, agent=%s, bert_errors=%d",
        threshold,
        comparison["agent_architecture"],
        len(errors),
    )

    return {
        "confidence_threshold": threshold,
        "suggested_threshold": suggested_thr,
        "searchable_share": comparison["searchable_share"],
        "agent_architecture": comparison["agent_architecture"],
        "bert_errors_n": len(errors),
        "val_preds_path": str(VAL_MERGED_PREDS_PATH),
        "comparison_path": str(COMPARISON_PATH),
        "fig_path": str(ACCURACY_COVERAGE_FIG),
        "bert_errors_sample_path": str(BERT_ERRORS_SAMPLE_PATH),
        "stage1_reference_val_accuracy": comparison["TF-IDF + LR (reference)"][
            "val_accuracy"
        ],
        "bert_val_accuracy": comparison["bert cross-encoder (T-scaled)"]["val_accuracy"],
        "temperature": calib.get("temperature"),
    }
