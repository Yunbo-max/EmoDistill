#!/usr/bin/env python3
"""
Run a random-emotion sweep: every creditor turn picks a random emotion from
the filtered subset. Produces mixed-emotion trajectories for IQL training.

Example (default 16-emotion subset, 20 scenarios × 10 iter):
  python experiments/run_random_emotion_sweep.py \
      --dataset_type debt \
      --scenarios 20 \
      --iterations 10 \
      --model_creditor qwen3.5-plus \
      --model_debtor qwen3.5-plus \
      --max_dialog_len 30 \
      --concurrency 6 \
      --seed 42
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


DATASET_PATHS = {
    "debt": "data/credit_recovery_scenarios.csv",
    "disaster": "data/disaster_survivor_scenarios.csv",
    "student": "data/education_sleep_scenarios.csv",
    "medical": "data/hospital_surgery_scenarios.csv",
}


def main():
    p = argparse.ArgumentParser(description="Random-emotion sweep for IQL training data")
    p.add_argument("--dataset_type", type=str, default="debt", choices=list(DATASET_PATHS.keys()))
    p.add_argument("--scenarios", type=int, default=20)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--offset", type=int, default=0,
                   help="Skip first N scenarios (use for held-out eval, e.g. --offset 80)")
    p.add_argument("--max_dialog_len", type=int, default=30)

    p.add_argument("--taxonomy", type=str, default="goemotions27",
                   choices=["ekman7", "izard10", "goemotions27"])
    p.add_argument("--emotion_subset", type=str, default="",
                   help="Comma-separated emotion subset (default: 16 with mean_v3 > 1.0)")

    p.add_argument("--model_creditor", type=str, default="qwen3.5-plus")
    p.add_argument("--model_debtor", type=str, default="qwen3.5-plus")
    p.add_argument("--debtor_emotion", type=str, default="neutral")

    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--out_dir", type=str, default="results/random_emotion_sweep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["EVOEMO_TAXONOMY"] = args.taxonomy
    from EmoDistill.emotions import set_active_taxonomy, get_emotions
    set_active_taxonomy(args.taxonomy)
    print(f"🎯 Active taxonomy: {args.taxonomy} ({len(get_emotions())} emotions)")

    from utils.preprocessing import preprocess_all_scenarios
    from EmoDistill.random_emotion_baseline import (
        run_random_emotion_sweep, DEFAULT_FILTERED_EMOTIONS_16,
    )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, DATASET_PATHS[args.dataset_type])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{args.dataset_type}_{args.taxonomy}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    scenarios_full = preprocess_all_scenarios(
        csv_path=csv_path,
        scenario_type=args.dataset_type,
        output_path=os.path.join(out_dir, "scenarios.json"),
        n_scenarios=args.scenarios + args.offset,
    )
    scenarios = scenarios_full[args.offset : args.offset + args.scenarios]
    print(f"✅ Loaded {len(scenarios)} scenarios (offset={args.offset})")

    # Resolve emotion subset
    if args.emotion_subset.strip():
        subset = [e.strip() for e in args.emotion_subset.split(",") if e.strip()]
    else:
        subset = DEFAULT_FILTERED_EMOTIONS_16
    print(f"🎲 Emotion subset ({len(subset)}): {subset}")

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        cfg_payload = vars(args).copy()
        cfg_payload["emotion_subset_resolved"] = subset
        json.dump(cfg_payload, f, indent=2)

    import random as _r
    import numpy as _np
    _r.seed(args.seed); _np.random.seed(args.seed)

    t0 = time.time()
    sweep = run_random_emotion_sweep(
        scenarios=scenarios,
        emotion_subset=subset,
        iterations=args.iterations,
        model_creditor=args.model_creditor,
        model_debtor=args.model_debtor,
        debtor_emotion=args.debtor_emotion,
        max_dialog_len=args.max_dialog_len,
        out_dir=out_dir,
        concurrency=args.concurrency,
        base_seed=args.seed,
    )

    elapsed = time.time() - t0
    print(f"\n⏱️  Total time: {elapsed/60:.1f} min")
    print(f"📁 Output dir: {out_dir}")


if __name__ == "__main__":
    main()
