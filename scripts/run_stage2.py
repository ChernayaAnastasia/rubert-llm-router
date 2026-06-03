#!/usr/bin/env python
"""
Run Stage 2: fine-tune deepvk/RuModernBERT-base cross-encoder + temperature scaling.

Usage:
    python scripts/run_stage2.py
    python scripts/run_stage2.py --epochs 5 --batch-size 16
    python scripts/run_stage2.py --no-early-stopping   # run all epochs
    python scripts/run_stage2.py --skip-train   # inference only (checkpoint must exist)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import create_directories
from utils.stage2_bert import run_stage2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 RuModernBERT cross-encoder fine-tuning"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Train batch size per device"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=32, help="Eval/predict batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision even on GPU",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load best_checkpoint and run val/OOD predictions only",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip temperature scaling; save raw softmax probabilities",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Stop if eval accuracy does not improve for N epochs (0 = disabled)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum accuracy gain to count as improvement",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume training from explicit checkpoint path (e.g. models/bert/checkpoints/checkpoint-2659)",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable auto-resume from the latest checkpoint",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Train for all epochs (same as --early-stopping-patience 0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    early_patience = 0 if args.no_early_stopping else args.early_stopping_patience

    create_directories()
    summary = run_stage2(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        fp16=False if args.no_fp16 else None,
        skip_train=args.skip_train,
        skip_calibration=args.skip_calibration,
        early_stopping_patience=early_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        auto_resume=not args.no_auto_resume,
        resume_from_checkpoint=Path(args.resume_checkpoint)
        if args.resume_checkpoint
        else None,
    )

    print("\n=== Stage 2 complete ===")
    print(f"  Val Accuracy:  {summary['val_accuracy']:.4f}")
    print(f"  Val Macro-F1:  {summary['val_macro_f1']:.4f}")
    print(f"  Checkpoint:    {summary['checkpoint_dir']}")
    print(f"  Val preds:     {summary['val_preds_path']}")
    print(f"  OOD preds:     {summary['ood_preds_path']}")
    print(f"  Metrics:       {summary['metrics_path']}")
    if summary.get("temperature") is not None:
        print(
            f"  Calibration:   T={summary['temperature']:.4f}  "
            f"NLL {summary['nll_before']:.4f}→{summary['nll_after']:.4f}  "
            f"ECE {summary['ece_before']:.4f}→{summary['ece_after']:.4f}"
        )
        print(f"  Reliability:   {summary.get('reliability_before_path')}")
        print(f"                 {summary.get('reliability_after_path')}")


if __name__ == "__main__":
    main()
