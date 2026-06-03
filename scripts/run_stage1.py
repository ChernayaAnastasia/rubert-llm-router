#!/usr/bin/env python
"""
Run Stage 1: TF-IDF reference baseline (metrics only).

Usage:
    python scripts/run_stage1.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import create_directories
from utils.stage1_baseline import run_stage1


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    create_directories()
    summary = run_stage1()

    print("\n=== Stage 1 complete ===")
    print(f"  Val Accuracy:  {summary['val_accuracy']:.4f}")
    print(f"  Val Macro-F1:  {summary['val_macro_f1']:.4f}")
    print(f"  Metrics:       {summary['metrics_path']}")


if __name__ == "__main__":
    main()
