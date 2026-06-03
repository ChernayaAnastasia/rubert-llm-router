#!/usr/bin/env python

"""
Run Stage 5: final evaluation on locked eval_baseline.parquet.

Usage:
    python scripts/run_stage5.py
    python scripts/run_stage5.py --sample 50
    python scripts/run_stage5.py --sleep 0.1

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 final eval pipeline")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only on N eval examples (debug). Default: full set.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Sleep between agent requests (sec). Default: 0.3",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=AGENT_LLM_MODEL,
        help=f"VseGPT model id. Default: {AGENT_LLM_MODEL}",
    )
    parser.add_argument(
        "--bert-only",
        action="store_true",
        help="Run only BERT inference (Stage 5A), save bert_eval_preds.parquet",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Run only agent loop (Stage 5B), requires bert_eval_preds.parquet",
    )
    args = parser.parse_args()

    if args.bert_only and args.agent_only:
        parser.error("Use at most one of --bert-only and --agent-only")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    create_directories()
    load_dotenv(ENV_PATH)

    from utils.stage5_agent import run_stage5_agent
    from utils.stage5_bert import run_stage5_bert_inference

    if not args.bert_only and not args.agent_only:
        from utils.stage5_eval import run_stage5

    if args.bert_only:
        path = run_stage5_bert_inference(sample_n=args.sample)
        print(f"\n=== Stage 5A complete ===\n  BERT preds: {path}")
        return

    if args.agent_only:
        summary = run_stage5_agent(
            sample_n=args.sample, sleep_sec=args.sleep, llm_model=args.model
        )
    else:
        summary = run_stage5(sample_n=args.sample, sleep_sec=args.sleep, llm_model=args.model)

    print("\n=== Stage 5 complete ===")
    print(f"  Threshold:        {summary['threshold']:.2f}")
    print(f"  Eval size:        {summary['eval_size']}")
    print(f"  Low-conf share:   {summary['low_conf_share']:.1%}")
    print(f"  BERT-only acc:    {summary['bert_acc']:.4f}")
    print(f"  BERT-only F1:     {summary['bert_macro_f1']:.4f}")
    print(f"  Hybrid acc:       {summary['hybrid_acc']:.4f}")
    print(f"  Hybrid macro-F1:  {summary['hybrid_macro_f1']:.4f}")
    print(f"  Agent preds:      {summary['paths'].agent_preds_path}")
    print(f"  Metrics:          {summary['paths'].metrics_path}")
    print(f"  Error matrix:     {summary['paths'].error_matrix_path}")
    print(f"  Figure:           {summary['paths'].fig_path}")


if __name__ == "__main__":
    main()
