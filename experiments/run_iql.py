#!/usr/bin/env python3
"""
Train IQL on a pre-collected offline dataset (NPZ produced by the
fixed-emotion sweep).

Example:
  # Train 50k steps on the dataset produced by the previous sweep
  python experiments/run_iql.py \\
      --dataset_path results/fixed_emotion_sweep/debt_goemotions27_20260512_xxxx/offline_trajectories.npz \\
      --n_steps 50000 \\
      --out_dir results/iql

  # Smaller smoke test (5k steps, lower expectile)
  python experiments/run_iql.py \\
      --dataset_path <path> --n_steps 5000 --expectile 0.7 --beta 3.0
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


def main():
    p = argparse.ArgumentParser(description="Offline IQL training on emotion-as-action dataset")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to .npz file produced by run_fixed_emotion_sweep.py")
    p.add_argument("--out_dir", type=str, default="results/iql")

    # Training
    p.add_argument("--n_steps", type=int, default=50000)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)

    # IQL specific
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7,
                   help="Upper expectile τ; higher = more optimistic V (default 0.7)")
    p.add_argument("--beta", type=float, default=3.0,
                   help="AWR inverse temperature; higher = sharper policy (default 3.0)")
    p.add_argument("--normalize_reward", type=str, default="scale",
                   choices=["none", "scale", "zscore"])
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")

    # Make sure metadata sidecar exists
    meta_path = args.dataset_path.rsplit(".", 1)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Dataset metadata sidecar missing: {meta_path}")

    # Pre-set taxonomy from metadata (before importing modules that lock onto it)
    with open(meta_path) as f:
        meta = json.load(f)
    os.environ["EVOEMO_TAXONOMY"] = meta.get("taxonomy", "ekman7")

    from EmoDistill.emotions import set_active_taxonomy, get_emotions
    set_active_taxonomy(meta.get("taxonomy", "ekman7"))
    print(f"🎯 Active taxonomy: {meta.get('taxonomy')} ({len(get_emotions())} emotions)")

    from EmoDistill.iql import run_iql_experiment

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{meta.get('taxonomy', 'taxonomy')}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n📦 Dataset: {args.dataset_path}")
    print(f"   {meta.get('n_transitions')} transitions, {meta.get('n_episodes')} episodes")
    print(f"   state_dim={meta.get('state_dim')}, n_emotions={meta.get('n_emotions')}")

    t0 = time.time()
    summary = run_iql_experiment(
        dataset_path=args.dataset_path,
        out_dir=out_dir,
        n_steps=args.n_steps,
        log_every=args.log_every,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        expectile=args.expectile,
        beta=args.beta,
        batch_size=args.batch_size,
        normalize_reward=args.normalize_reward,
        seed=args.seed,
    )

    elapsed = time.time() - t0
    print(f"\n🏁 IQL training done in {elapsed/60:.1f} min")
    print(f"   Final losses: {summary['final_metrics']}")
    print(f"   Checkpoint:   {summary['ckpt_path']}")


if __name__ == "__main__":
    main()
