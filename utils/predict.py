"""Prediction helpers for the Stage 2+ BERT cross-encoder model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from utils.calibration import apply_temperature, load_calibration
from utils.config import (
    BERT_BEST_CHECKPOINT_DIR,
    BERT_CALIBRATION_PATH,
    BERT_MAX_LENGTH,
    BERT_TRAINING_ARGS_PATH,
)

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, *, total=None, desc=None, unit=None, leave=True, **kwargs):
            self._iter = iter(iterable) if iterable is not None else None

        def __iter__(self):
            return self._iter if self._iter is not None else iter(())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            pass

        def set_description(self, desc=None, refresh=True):
            pass

        def close(self):
            pass


_bert_cache: dict = {}


def _as_list(texts: Union[str, List[str]]) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _load_bert_temperature() -> float:
    cal_path = Path(BERT_CALIBRATION_PATH)
    if cal_path.exists():
        return float(load_calibration(cal_path)["temperature"])
    return 1.0


def _load_bert_max_length(model_path: Path) -> int:
    meta_path = model_path.parent / "training_args.json"
    if meta_path.exists():
        import json

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("max_length", BERT_MAX_LENGTH))
    if Path(BERT_TRAINING_ARGS_PATH).exists():
        import json

        with open(BERT_TRAINING_ARGS_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("max_length", BERT_MAX_LENGTH))
    return BERT_MAX_LENGTH


def _get_bert_model_and_tokenizer(model_path: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_key = str(model_path.resolve())
    if cache_key in _bert_cache:
        return _bert_cache[cache_key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"BERT checkpoint not found: {model_path}. "
            "Run Stage 2 first (scripts/run_stage2.py or stage2 notebook)."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _bert_cache[cache_key] = (model, tokenizer, device)
    return _bert_cache[cache_key]


def predict_bert(
    queries: Union[str, List[str]],
    org_texts: Union[str, List[str]],
    *,
    model_path: Optional[Union[str, Path]] = None,
    max_length: Optional[int] = None,
    batch_size: int = 64,
    return_proba: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Predict with fine-tuned cross-encoder (Stage 2+).

    Tokenization matches training: [CLS] query [SEP] org_text [SEP],
    ``truncation='only_second'`` — query is never truncated.
    Batches use dynamic padding (longest in batch, up to ``max_length``).

    Returns
    -------
    dict with keys:
        pred   : np.ndarray of int labels {0, 1}
        proba1 : np.ndarray of calibrated P(class=1), if return_proba=True
    """
    import torch

    queries_list = _as_list(queries)
    org_list = _as_list(org_texts)
    if len(queries_list) != len(org_list):
        raise ValueError(
            f"queries and org_texts length mismatch: {len(queries_list)} vs {len(org_list)}"
        )

    model_path = Path(model_path or BERT_BEST_CHECKPOINT_DIR)
    n_samples = len(queries_list)
    n_batches = (n_samples + batch_size - 1) // batch_size if n_samples else 0
    use_progress = show_progress and n_samples > 0

    phase_pbar = (
        tqdm(total=3, desc="predict_bert", unit="phase", leave=False)
        if use_progress
        else None
    )

    def _set_phase(name: str) -> None:
        if phase_pbar is not None:
            phase_pbar.set_description(f"predict_bert: {name}")

    def _advance_phase() -> None:
        if phase_pbar is not None:
            phase_pbar.update(1)

    _set_phase("load model")
    model, tokenizer, device = _get_bert_model_and_tokenizer(model_path)
    _advance_phase()

    _set_phase("calibration & config")
    temperature = _load_bert_temperature()
    if max_length is None:
        max_length = _load_bert_max_length(model_path)
    _advance_phase()

    _set_phase("inference")
    _advance_phase()
    if phase_pbar is not None:
        phase_pbar.close()

    all_preds: list[int] = []
    all_proba1: list[float] = []

    batch_starts = range(0, n_samples, batch_size)
    if use_progress:
        batch_starts = tqdm(
            batch_starts,
            total=n_batches,
            desc="predict_bert: inference",
            unit="batch",
        )

    with torch.no_grad():
        for start in batch_starts:
            batch_q = queries_list[start : start + batch_size]
            batch_o = org_list[start : start + batch_size]
            encoded = tokenizer(
                batch_q,
                batch_o,
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            if return_proba:
                proba = apply_temperature(logits.cpu().numpy(), temperature)[:, 1]
                all_proba1.extend(proba.tolist())

    result: dict = {"pred": np.asarray(all_preds, dtype=int)}
    if return_proba:
        result["proba1"] = np.asarray(all_proba1, dtype=float)
    return result


def predict(
    queries: Union[str, List[str]],
    org_texts: Union[str, List[str]],
    **kwargs,
) -> dict:
    """Unified entry point for Stage 2+ (BERT cross-encoder)."""
    return predict_bert(queries, org_texts, **kwargs)
