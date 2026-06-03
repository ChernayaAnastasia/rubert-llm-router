"""
BERT confidence routing helpers (threshold + max_confidence).

Shared by Stage 3–5 without agent / langchain dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from utils.config import STAGE3_COMPARISON_PATH
from utils.stage3_error_analysis import max_confidence

STAGE3_COMPARISON_PATH = Path(STAGE3_COMPARISON_PATH)


def load_confidence_threshold(
    path: Optional[Union[str, Path]] = None,
) -> float:
    """Read ``confidence_threshold`` from Stage 3 ``comparison.json``."""
    cmp_path = Path(path or STAGE3_COMPARISON_PATH)
    if not cmp_path.exists():
        raise FileNotFoundError(
            f"Stage 3 comparison.json not found: {cmp_path}. Run scripts/run_stage3.py first."
        )
    with open(cmp_path, encoding="utf-8") as f:
        data = json.load(f)
    thr = data.get("confidence_threshold")
    if thr is None:
        raise ValueError(f"confidence_threshold missing in {cmp_path}")
    return float(thr)


def max_confidence_series(proba1: pd.Series) -> pd.Series:
    """Per-row max(p, 1-p) for calibrated P(class=1)."""
    return proba1.map(max_confidence)
