#!/usr/bin/env python

"""
Run Stage 4: LLM agent on low-confidence validation examples.

Usage:
    python scripts/run_stage4.py
    python scripts/run_stage4.py --sample 50
    python scripts/run_stage4.py --sleep 0.2

Requires environment variables (e.g. via .env in project root):
    VSEGPT_API_KEY=...
    TAVILY_API_KEY=...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import AGENT_LLM_MODEL, ENV_PATH, create_directories
from utils.stage4_agent import run_stage4


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 agent pipeline")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only on N low-confidence examples (debug). Default: full set.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Sleep between requests (sec). Default: 0.3",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=AGENT_LLM_MODEL,
        help=f"VseGPT model id. Default: {AGENT_LLM_MODEL}",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    create_directories()
    load_dotenv(ENV_PATH)

    summary = run_stage4(sample_n=args.sample, sleep_sec=args.sleep, llm_model=args.model)

    print("\n=== Stage 4 complete ===")
    print(f"  Threshold:            {summary['threshold']:.2f}")
    print(f"  Low-confidence rows:  {summary['low_conf_n']}")
    print(f"  Agent acc (low-conf): {summary['agent_acc_on_low_conf']:.4f}")
    print(f"  Hybrid acc (val):     {summary['hybrid_acc']:.4f}")
    print(f"  Hybrid macro-F1:      {summary['hybrid_macro_f1']:.4f}")
    print(f"  Agent preds:          {summary['paths'].agent_preds_path}")
    print(f"  Report:               {summary['paths'].report_path}")


if __name__ == "__main__":
    main()

