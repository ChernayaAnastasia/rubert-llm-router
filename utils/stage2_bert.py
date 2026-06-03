"""
Stage 2 — Fine-tune deepvk/RuModernBERT-base as cross-encoder + temperature scaling.

Architecture: [CLS] COL_QUERY [SEP] org_text [SEP]
Trains on train_baseline.parquet, evaluates on val, predicts on rel_minus (OOD).
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score as sklearn_f1
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    load_calibration,
    negative_log_likelihood,
    plot_reliability_diagram,
    proba1_from_logits,
    save_calibration,
)
from utils.config import (
    BERT_OOD_PREDS_PATH,
    BERT_VAL_PREDS_PATH,
    COL_ADDRESS,
    COL_BERT_CORRECT,
    COL_BERT_PRED,
    COL_BERT_PROBA1,
    COL_NAME,
    COL_ORG_TEXT,
    COL_PRICELIST,
    COL_QUERY,
    COL_REVIEWS,
    COL_RUBRIC,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    BERT_BEST_CHECKPOINT_DIR,
    BERT_CALIBRATION_PATH,
    BERT_CHECKPOINTS_DIR,
    BERT_DIR,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_TRAINING_ARGS_PATH,
    STAGE2_RELIABILITY_AFTER_PATH,
    STAGE2_RELIABILITY_BEFORE_PATH,
    STAGE2_REPORTS_DIR,
    TARGET,
)
from utils.data_loader import attach_org_text
from utils.metrics import eval_binary

logger = logging.getLogger(__name__)

STAGE2_METRICS_PATH = Path(STAGE2_REPORTS_DIR) / "metrics.json"
VAL_PREDS_PATH = Path(BERT_VAL_PREDS_PATH)
OOD_PREDS_PATH = Path(BERT_OOD_PREDS_PATH)
BEST_CHECKPOINT_DIR = Path(BERT_BEST_CHECKPOINT_DIR)
TRAINING_ARGS_PATH = Path(BERT_TRAINING_ARGS_PATH)
CALIBRATION_PATH = Path(BERT_CALIBRATION_PATH)
RELIABILITY_BEFORE_PATH = Path(STAGE2_RELIABILITY_BEFORE_PATH)
RELIABILITY_AFTER_PATH = Path(STAGE2_RELIABILITY_AFTER_PATH)


def _require_transformers_stack() -> None:
    try:
        import evaluate  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Stage 2 requires: pip install transformers datasets accelerate "
            "evaluate torch"
        ) from exc


def load_stage2_splits(
    processed_dir: Optional[Path] = None,
    *,
    include_ood: bool = True,
) -> Tuple[pd.DataFrame, ...]:
    """Load baseline splits and attach COL_ORG_TEXT via make_org_text()."""
    processed_dir = Path(processed_dir or PROCESSED_DATA_DIR)
    train_df = attach_org_text(
        pd.read_parquet(processed_dir / "train_baseline.parquet")
    )
    val_df = attach_org_text(
        pd.read_parquet(processed_dir / "val_baseline.parquet")
    )
    if not include_ood:
        return train_df, val_df

    ood_df = attach_org_text(
        pd.read_parquet(processed_dir / "rel_minus_baseline.parquet")
    )
    return train_df, val_df, ood_df


def validate_train_data(train_df: pd.DataFrame) -> None:
    """Pre-training sanity checks from the project spec."""
    assert set(train_df[TARGET].unique()) == {0, 1}, (
        f"Expected labels {{0, 1}}, got {set(train_df[TARGET].unique())}"
    )
    assert len(train_df) > 20_000, f"Train too small: {len(train_df)}"


def build_tokenizer(model_name: str = BERT_MODEL_NAME):
    _require_transformers_stack()
    return AutoTokenizer.from_pretrained(model_name)


def _resolve_tokenizer(
    trainer: Trainer,
    tokenizer=None,
    *,
    model_name: str = BERT_MODEL_NAME,
):
    """Tokenizer from explicit arg, Trainer (4.40+), or base model."""
    if tokenizer is not None:
        return tokenizer
    tok = getattr(trainer, "processing_class", None) or getattr(
        trainer, "tokenizer", None
    )
    if tok is not None:
        return tok
    return build_tokenizer(model_name)


def _tokenizer_trainer_kwargs(tokenizer) -> dict:
    """Map tokenizer to the Trainer kwarg name for this transformers version."""
    if tokenizer is None:
        return {}
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


def _create_trainer(*, tokenizer=None, **kwargs) -> Trainer:
    kwargs.update(_tokenizer_trainer_kwargs(tokenizer))
    return Trainer(**kwargs)


def tokenize_cross_encoder_batch(
    batch,
    tokenizer,
    *,
    max_length: int = BERT_MAX_LENGTH,
):
    """
    Cross-encoder: [CLS] COL_QUERY [SEP] org_text [SEP].
    truncation='only_second' — org_text is truncated, query is never.
    """
    return tokenizer(
        batch[COL_QUERY],
        batch[COL_ORG_TEXT],
        padding="max_length",
        truncation="only_second",
        max_length=max_length,
    )


def log_token_length_stats(
    df: pd.DataFrame,
    tokenizer,
    *,
    sample_size: int = 500,
    max_length: int = BERT_MAX_LENGTH,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Log token-length percentiles for COL_QUERY + org_text (spec §2.1)."""
    sample = df.sample(min(sample_size, len(df)), random_state=random_state)
    lengths = [
        len(tokenizer(q, o)["input_ids"])
        for q, o in zip(sample[COL_QUERY], sample[COL_ORG_TEXT])
    ]
    arr = np.array(lengths)
    truncated_frac = float((arr > max_length).mean())
    stats = {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "truncated_frac_at_max_length": truncated_frac,
    }
    logger.info(
        "Token lengths (%s + %s): p50=%.0f p90=%.0f p95=%.0f max=%.0f "
        "(truncated at max_length=%d: %.1f%%)",
        COL_QUERY,
        COL_ORG_TEXT,
        stats["p50"],
        stats["p90"],
        stats["p95"],
        stats["max"],
        max_length,
        100 * truncated_frac,
    )
    return stats


def build_hf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    tokenizer=None,
    max_length: int = BERT_MAX_LENGTH,
) -> Tuple[Dataset, Dataset]:
    """Tokenize train/val for cross-encoder HuggingFace Trainer."""
    if tokenizer is None:
        tokenizer = build_tokenizer()

    train_raw = Dataset.from_pandas(
        train_df[[COL_QUERY, COL_ORG_TEXT, TARGET]].rename(columns={TARGET: "labels"})
    )
    val_raw = Dataset.from_pandas(
        val_df[[COL_QUERY, COL_ORG_TEXT, TARGET]].rename(columns={TARGET: "labels"})
    )

    def _tokenize(batch):
        return tokenize_cross_encoder_batch(
            batch, tokenizer, max_length=max_length
        )

    remove_cols = [COL_QUERY, COL_ORG_TEXT]
    train_ds = train_raw.map(_tokenize, batched=True, remove_columns=remove_cols)
    val_ds = val_raw.map(_tokenize, batched=True, remove_columns=remove_cols)
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    return train_ds, val_ds


def build_ood_dataset(
    ood_df: pd.DataFrame,
    *,
    tokenizer=None,
    max_length: int = BERT_MAX_LENGTH,
) -> Dataset:
    """OOD dataset — dummy labels column required by Trainer.predict."""
    if tokenizer is None:
        tokenizer = build_tokenizer()

    ood_raw = Dataset.from_pandas(
        ood_df[[COL_QUERY, COL_ORG_TEXT]].assign(labels=0)
    )

    def _tokenize(batch):
        return tokenize_cross_encoder_batch(
            batch, tokenizer, max_length=max_length
        )

    ood_ds = ood_raw.map(
        _tokenize, batched=True, remove_columns=[COL_QUERY, COL_ORG_TEXT]
    )
    ood_ds.set_format("torch")
    return ood_ds


def build_trainer(
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    tokenizer=None,
    model_name: str = BERT_MODEL_NAME,
    output_dir: Optional[Path] = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    fp16: Optional[bool] = None,
    early_stopping_patience: int = 1,
    early_stopping_threshold: float = 0.0,
) -> Trainer:
    """Create Trainer with accuracy + macro-F1 metrics and optional early stopping."""
    import evaluate

    _require_transformers_stack()

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    output_dir = Path(output_dir or BERT_CHECKPOINTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        macro_f1 = sklearn_f1(labels, preds, average="macro")
        return {"accuracy": acc["accuracy"], "macro_f1": macro_f1}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=RANDOM_STATE,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )
        logger.info(
            "Early stopping enabled — patience=%d on eval accuracy",
            early_stopping_patience,
        )

    return _create_trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )


def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    trainer: Optional[Trainer] = None,
    tokenizer=None,
    **trainer_kwargs,
) -> Trainer:
    """Fine-tune cross-encoder; returns fitted Trainer (best checkpoint loaded)."""
    if trainer is None:
        train_ds, val_ds = build_hf_datasets(train_df, val_df, tokenizer=tokenizer)
        trainer = build_trainer(
            train_ds, val_ds, tokenizer=tokenizer, **trainer_kwargs
        )

    trainer.train()
    return trainer


def predict_logits(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    """Run Trainer.predict and return raw logits."""
    output = trainer.predict(dataset)
    return output.predictions


def logits_to_pred_proba(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert logits to class predictions and P(class=1)."""
    proba = softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    return preds, proba[:, 1]


def save_best_checkpoint(
    trainer: Trainer,
    checkpoint_dir: Optional[Path] = None,
    *,
    tokenizer=None,
    model_name: str = BERT_MODEL_NAME,
    max_length: int = BERT_MAX_LENGTH,
    num_train_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    early_stopping_patience: Optional[int] = None,
    training_args_path: Optional[Path] = None,
) -> Path:
    """Save best model, tokenizer, and training_args.json metadata."""
    checkpoint_dir = Path(checkpoint_dir or BEST_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(checkpoint_dir))
    _resolve_tokenizer(trainer, tokenizer, model_name=model_name).save_pretrained(
        str(checkpoint_dir)
    )

    last_log = [
        entry
        for entry in trainer.state.log_history
        if "eval_macro_f1" in entry
    ]
    best_macro_f1 = last_log[-1]["eval_macro_f1"] if last_log else None

    training_meta = {
        "base_model": model_name,
        "architecture": f"cross-encoder: [CLS] {COL_QUERY} [SEP] {COL_ORG_TEXT} [SEP]",
        "org_text_cols": (
            f"{COL_NAME} | {COL_ADDRESS} | {COL_RUBRIC} | "
            f"{COL_REVIEWS} | {COL_PRICELIST}"
        ),
        "truncation": "only_second",
        "max_length": max_length,
        "epochs": num_train_epochs,
        "batch_size": batch_size,
        "lr": learning_rate,
        "early_stopping_patience": early_stopping_patience,
        "random_state": RANDOM_STATE,
        "best_val_acc": float(trainer.state.best_metric)
        if trainer.state.best_metric is not None
        else None,
        "last_macro_f1": float(best_macro_f1) if best_macro_f1 is not None else None,
        "calibration": "temperature scaling — see calibration.json",
    }

    training_args_path = Path(training_args_path or TRAINING_ARGS_PATH)
    training_args_path.parent.mkdir(parents=True, exist_ok=True)
    with open(training_args_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2, ensure_ascii=False)

    logger.info(
        "Checkpoint saved → %s (best val accuracy: %s)",
        checkpoint_dir,
        training_meta["best_val_acc"],
    )
    return checkpoint_dir


def save_val_predictions(
    val_df: pd.DataFrame,
    preds: np.ndarray,
    proba1: np.ndarray,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    output_path = Path(output_path or VAL_PREDS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val_preds_df = val_df.copy()
    val_preds_df[COL_BERT_PRED] = preds
    val_preds_df[COL_BERT_PROBA1] = proba1
    val_preds_df[COL_BERT_CORRECT] = val_preds_df[COL_BERT_PRED] == val_preds_df[TARGET]

    val_preds_df.to_parquet(output_path, index=False)
    logger.info("Saved %s — %d rows", output_path, len(val_preds_df))
    return val_preds_df


def save_ood_predictions(
    ood_df: pd.DataFrame,
    preds: np.ndarray,
    proba1: np.ndarray,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    output_path = Path(output_path or OOD_PREDS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ood_preds_df = ood_df.copy()
    ood_preds_df[COL_BERT_PRED] = preds
    ood_preds_df[COL_BERT_PROBA1] = proba1

    ood_preds_df.to_parquet(output_path, index=False)
    logger.info("Saved %s — %d rows", output_path, len(ood_preds_df))
    return ood_preds_df


def fit_and_save_temperature(
    val_logits: np.ndarray,
    y_val: np.ndarray,
    *,
    calibration_path: Optional[Path] = None,
    reliability_before_path: Optional[Path] = None,
    reliability_after_path: Optional[Path] = None,
) -> dict:
    """Fit T on val logits, save calibration.json and reliability diagrams."""
    calibration_path = Path(calibration_path or CALIBRATION_PATH)
    reliability_before_path = Path(
        reliability_before_path or RELIABILITY_BEFORE_PATH
    )
    reliability_after_path = Path(reliability_after_path or RELIABILITY_AFTER_PATH)

    raw_proba1 = proba1_from_logits(val_logits)
    temperature = fit_temperature(val_logits, y_val)

    cal_proba = apply_temperature(val_logits, temperature)
    cal_proba1 = cal_proba[:, 1]

    nll_before = negative_log_likelihood(y_val, raw_proba1)
    nll_after = negative_log_likelihood(y_val, cal_proba1)
    ece_before = expected_calibration_error(y_val, raw_proba1, n_bins=10)
    ece_after = expected_calibration_error(y_val, cal_proba1, n_bins=10)

    save_calibration(
        calibration_path,
        temperature=temperature,
        nll_before=round(nll_before, 6),
        nll_after=round(nll_after, 6),
        ece_before=round(ece_before, 6),
        ece_after=round(ece_after, 6),
        extra={"fitted_on": "val"},
    )
    plot_reliability_diagram(
        y_val,
        raw_proba1,
        reliability_before_path,
        n_bins=10,
        title="Cross-encoder (до temperature scaling)",
    )
    plot_reliability_diagram(
        y_val,
        cal_proba1,
        reliability_after_path,
        n_bins=10,
        title="Cross-encoder (после temperature scaling)",
    )

    logger.info(
        "Temperature scaling — T=%.4f, NLL %.4f→%.4f, ECE %.4f→%.4f",
        temperature,
        nll_before,
        nll_after,
        ece_before,
        ece_after,
    )
    return {
        "temperature": temperature,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "calibration_path": str(calibration_path),
        "reliability_before_path": str(reliability_before_path),
        "reliability_after_path": str(reliability_after_path),
    }


def calibrated_proba1(
    logits: np.ndarray,
    calibration_path: Optional[Path] = None,
) -> np.ndarray:
    """Apply saved temperature to logits; return P(class=1)."""
    path = Path(calibration_path or CALIBRATION_PATH)
    if path.exists():
        temperature = float(load_calibration(path)["temperature"])
    else:
        temperature = 1.0
    return apply_temperature(logits, temperature)[:, 1]


def load_inference_trainer(
    checkpoint_dir: Optional[Path] = None,
    *,
    tokenizer=None,
    model_name: str = BERT_MODEL_NAME,
) -> Trainer:
    """Load fine-tuned checkpoint for inference / calibration only."""
    checkpoint_dir = Path(checkpoint_dir or BEST_CHECKPOINT_DIR)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    return _create_trainer(model=model, tokenizer=tokenizer)


def run_calibration(
    trainer: Trainer,
    val_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    val_ds: Dataset,
    ood_ds: Dataset,
) -> tuple[dict, np.ndarray]:
    """
    Fit temperature on val logits, save calibration artifacts and predictions.

    ``bert_pred`` is unchanged (argmax invariant under T>0); ``bert_proba1`` is calibrated.

    Returns
    -------
    calibration_meta, val_preds
        JSON-serializable metadata and val class predictions (kept separate).
    """
    val_logits = predict_logits(trainer, val_ds)
    val_preds = np.argmax(val_logits, axis=-1)
    y_val = val_df[TARGET].values

    calibration_meta = fit_and_save_temperature(val_logits, y_val)
    val_proba1 = calibrated_proba1(val_logits)

    ood_logits = predict_logits(trainer, ood_ds)
    ood_preds = np.argmax(ood_logits, axis=-1)
    ood_proba1 = calibrated_proba1(ood_logits)

    save_val_predictions(val_df, val_preds, val_proba1)
    save_ood_predictions(ood_df, ood_preds, ood_proba1)

    val_accuracy = float((val_preds == y_val).mean())
    val_macro_f1 = float(sklearn_f1(y_val, val_preds, average="macro"))
    summary = {
        **calibration_meta,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
    }
    return summary, val_preds


def save_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    trainer: Optional[Trainer] = None,
    training_meta: Optional[dict] = None,
    calibration_meta: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> Path:
    output_path = Path(output_path or STAGE2_METRICS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = eval_binary(y_true, y_pred, model_name=BERT_MODEL_NAME)
    report = {
        "model": BERT_MODEL_NAME,
        "architecture": f"cross-encoder: [CLS] {COL_QUERY} [SEP] {COL_ORG_TEXT} [SEP]",
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_report": metrics["report"],
        "training": training_meta or {},
        "calibration": calibration_meta or {},
    }
    if trainer is not None and trainer.state.best_metric is not None:
        report["best_val_acc_trainer"] = round(float(trainer.state.best_metric), 4)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Metrics saved → %s", output_path)
    return output_path


def run_stage2(
    *,
    processed_dir: Optional[Path] = None,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    fp16: Optional[bool] = None,
    skip_train: bool = False,
    skip_calibration: bool = False,
    checkpoint_dir: Optional[Path] = None,
    early_stopping_patience: int = 1,
    early_stopping_threshold: float = 0.0,
    auto_resume: bool = True,
    resume_from_checkpoint: Optional[Path] = None,
) -> dict:
    """
    Execute full Stage 2 pipeline.

    Parameters
    ----------
    skip_train : bool
        If True, load existing best_checkpoint and only run inference.
    skip_calibration : bool
        If True, skip temperature scaling and save raw softmax probabilities.
    early_stopping_patience : int
        Stop when eval accuracy does not improve for this many epochs (0 = off).
    early_stopping_threshold : float
        Minimum improvement in eval accuracy to reset the patience counter.
    """
    _require_transformers_stack()

    def _find_latest_checkpoint(
        checkpoints_dir: Path, *, require_optimizer_state: bool = True
    ) -> Optional[Path]:
        if not checkpoints_dir.exists():
            return None

        ckpt_dirs: list[Path] = []
        for p in checkpoints_dir.glob("checkpoint-*"):
            if not p.is_dir():
                continue
            suffix = p.name.split("-")[-1]
            if not suffix.isdigit():
                continue
            ckpt_dirs.append(p)

        if not ckpt_dirs:
            return None

        ckpt_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        if not require_optimizer_state:
            return ckpt_dirs[-1]

        # Resume с optimizer/scheduler обычно стабильнее.
        for p in reversed(ckpt_dirs):
            if (p / "optimizer.pt").exists():
                return p
        return ckpt_dirs[-1]

    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(BERT_DIR).mkdir(parents=True, exist_ok=True)
    Path(STAGE2_REPORTS_DIR).mkdir(parents=True, exist_ok=True)

    train_df, val_df, ood_df = load_stage2_splits(processed_dir)
    validate_train_data(train_df)

    logger.info(
        "Data loaded — train: %d, val: %d, ood: %d",
        len(train_df),
        len(val_df),
        len(ood_df),
    )

    tokenizer = build_tokenizer()
    log_token_length_stats(train_df, tokenizer)

    train_ds, val_ds = build_hf_datasets(train_df, val_df, tokenizer=tokenizer)
    ood_ds = build_ood_dataset(ood_df, tokenizer=tokenizer)

    if skip_train and BEST_CHECKPOINT_DIR.exists():
        logger.info("Loading checkpoint from %s", BEST_CHECKPOINT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(BEST_CHECKPOINT_DIR))
        train_ds, val_ds = build_hf_datasets(
            train_df, val_df, tokenizer=tokenizer
        )
        ood_ds = build_ood_dataset(ood_df, tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(BEST_CHECKPOINT_DIR)
        )
        trainer = _create_trainer(model=model, tokenizer=tokenizer)
    else:
        trainer = build_trainer(
            train_ds,
            val_ds,
            tokenizer=tokenizer,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            fp16=fp16,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
        resume_ckpt = resume_from_checkpoint
        if resume_ckpt is None and auto_resume:
            resume_ckpt = _find_latest_checkpoint(
                Path(BERT_CHECKPOINTS_DIR), require_optimizer_state=True
            )

        if resume_ckpt is not None:
            logger.info("Resuming training from checkpoint: %s", resume_ckpt)
            trainer.train(resume_from_checkpoint=str(resume_ckpt))
        else:
            trainer.train()
        save_best_checkpoint(
            trainer,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
            num_train_epochs=num_train_epochs,
            batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
        )

    if skip_calibration:
        val_logits = predict_logits(trainer, val_ds)
        val_preds, val_proba1 = logits_to_pred_proba(val_logits)
        y_val = val_df[TARGET].values
        ood_logits = predict_logits(trainer, ood_ds)
        ood_preds, ood_proba1 = logits_to_pred_proba(ood_logits)
        save_val_predictions(val_df, val_preds, val_proba1)
        save_ood_predictions(ood_df, ood_preds, ood_proba1)
        calibration_summary = {}
        val_accuracy = float((val_preds == y_val).mean())
        val_macro_f1 = float(sklearn_f1(y_val, val_preds, average="macro"))
    else:
        calibration_summary, val_preds = run_calibration(
            trainer, val_df, ood_df, val_ds, ood_ds
        )
        y_val = val_df[TARGET].values
        val_accuracy = calibration_summary["val_accuracy"]
        val_macro_f1 = calibration_summary["val_macro_f1"]

    training_meta = {}
    if TRAINING_ARGS_PATH.exists():
        with open(TRAINING_ARGS_PATH, encoding="utf-8") as f:
            training_meta = json.load(f)

    metrics_path = save_metrics_report(
        y_val,
        val_preds,
        trainer=trainer,
        training_meta=training_meta,
        calibration_meta=calibration_summary or None,
    )

    summary = {
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "checkpoint_dir": str(checkpoint_dir or BEST_CHECKPOINT_DIR),
        "val_preds_path": str(VAL_PREDS_PATH),
        "ood_preds_path": str(OOD_PREDS_PATH),
        "metrics_path": str(metrics_path),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "ood_rows": len(ood_df),
    }
    if calibration_summary:
        summary.update(
            {
                "calibration_path": calibration_summary["calibration_path"],
                "temperature": calibration_summary["temperature"],
                "nll_before": calibration_summary["nll_before"],
                "nll_after": calibration_summary["nll_after"],
                "ece_before": calibration_summary["ece_before"],
                "ece_after": calibration_summary["ece_after"],
                "reliability_before_path": calibration_summary[
                    "reliability_before_path"
                ],
                "reliability_after_path": calibration_summary[
                    "reliability_after_path"
                ],
            }
        )
    logger.info(
        "Stage 2 complete — accuracy=%.4f, macro-F1=%.4f",
        summary["val_accuracy"],
        summary["val_macro_f1"],
    )
    return summary
