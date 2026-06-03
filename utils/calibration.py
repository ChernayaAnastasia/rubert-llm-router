"""Temperature scaling calibration for binary classifiers (Stage 2 BERT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 50,
) -> float:
    """
    Fit a single scalar temperature T by minimizing NLL on validation logits (LBFGS).

    Predictions (argmax) are unchanged; only softmax probabilities are rescaled.
    """
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    log_temp = nn.Parameter(torch.zeros(1))

    optimizer = torch.optim.LBFGS([log_temp], lr=0.01, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = log_temp.exp()
        loss = F.cross_entropy(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temp.exp().detach().cpu().item())


def apply_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return calibrated class probabilities (N, 2) via softmax(logits / T)."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    temp = max(float(temperature), 1e-6)
    return F.softmax(logits_t / temp, dim=-1).numpy()


def proba1_from_logits(logits: np.ndarray) -> np.ndarray:
    """P(class=1) from uncalibrated logits."""
    return apply_temperature(logits, temperature=1.0)[:, 1]


def negative_log_likelihood(
    y_true: np.ndarray,
    proba1: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean binary NLL from P(class=1)."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba1, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    y_true: np.ndarray,
    proba1: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Expected calibration error for binary labels and P(class=1)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba1, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)
        if not mask.any():
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    proba1: np.ndarray,
    output_path: Union[str, Path],
    *,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> Path:
    """Save reliability diagram (confidence vs. accuracy per bin)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba1, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers: list[float] = []
    bin_accs: list[float] = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)
        if not mask.any():
            continue
        bin_centers.append(0.5 * (low + high))
        bin_accs.append(float(y[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if bin_centers:
        ax.bar(
            bin_centers,
            bin_accs,
            width=1.0 / n_bins * 0.9,
            alpha=0.7,
            edgecolor="black",
            label="Model",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Fraction of positives (accuracy)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_calibration(
    path: Union[str, Path],
    *,
    temperature: float,
    nll_before: Optional[float] = None,
    nll_after: Optional[float] = None,
    ece_before: Optional[float] = None,
    ece_after: Optional[float] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Persist temperature and optional calibration metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "temperature_scaling",
        "temperature": float(temperature),
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_calibration(path: Union[str, Path]) -> dict:
    """Load calibration.json; returns dict with at least ``temperature``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "temperature" not in data:
        raise KeyError(f"Missing 'temperature' in {path}")
    return data
