#!/usr/bin/env python
"""
Run Stage 3: BERT error analysis, confidence threshold, agent architecture.

Usage:
    python scripts/run_stage3.py
    python scripts/run_stage3.py --threshold 0.75
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import create_directories
from utils.stage3_error_analysis import run_stage3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 error analysis")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed CONFIDENCE_THRESHOLD (default: auto from val curve, spec 3.2)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    create_directories()
    summary = run_stage3(confidence_threshold=args.threshold)

    print("\n=== Stage 3 complete ===")
    print(
        f"  TF-IDF (reference): acc={summary['stage1_reference_val_accuracy']:.4f}  "
        "(from stage1 metrics.json)"
    )
    print(f"  BERT val acc:       {summary['bert_val_accuracy']:.4f}")
    if summary.get("temperature") is not None:
        print(f"  Temperature T:      {summary['temperature']:.4f}")
    print(f"  Threshold:          {summary['confidence_threshold']:.2f}")
    print(f"    (suggested):      {summary['suggested_threshold']:.2f}")
    print(f"  BERT errors:        {summary['bert_errors_n']}")
    print(f"  Searchable share:   {summary['searchable_share']:.1%}")
    print(f"  Agent architecture: {summary['agent_architecture']}")
    print(f"  Val preds:          {summary['val_preds_path']}")
    print(f"  Report:             {summary['comparison_path']}")
    print(f"  Coverage plot:      {summary['fig_path']}")
    print(f"  Errors sample:      {summary['bert_errors_sample_path']}")
    print("\n  Review bert_errors_sample.csv and fill error_taxonomy in")
    print("  notebooks/stage3_error_analysis.ipynb before Stage 4.")


if __name__ == "__main__":
    main()
