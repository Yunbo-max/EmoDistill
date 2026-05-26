#!/usr/bin/env python3
"""
End-to-end offline-GRPO training pipeline:

1. Extract GRPO training pairs from a random_emotion_sweep dir
   (advantage = z-scored episode reward per scenario; see EmoDistill/grpo_data.py)
2. Fine-tune Qwen2.5-7B-Instruct with offline GRPO
   (see EmoDistill/grpo_train.py for the loss)

Typical usage (after Step 1 + Step 4 of PIPELINE.md):

  python experiments/run_grpo_train.py \\
      --sweep_dir results/random_emotion_sweep/debt_goemotions27_<TS>/ \\
      --sft_adapter results/lora/qwen2.5-7b-creditor/run_<TS>/adapter_final \\
      --base_model Qwen/Qwen2.5-7B-Instruct \\
      --epochs 1 --batch_size 1 --grad_accum 16

Cold start (no SFT adapter — GRPO directly on base Qwen):

  python experiments/run_grpo_train.py \\
      --sweep_dir results/random_emotion_sweep/debt_goemotions27_<TS>/ \\
      --base_model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--scenario_type", default="debt")
    ap.add_argument("--min_group_size", type=int, default=4)
    ap.add_argument("--advantage_clip", type=float, default=10.0)
    ap.add_argument("--reward_field", default="total_episode_reward")
    # Model
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_adapter", default=None,
                    help="Optional path to SFT LoRA adapter to warm-start GRPO")
    ap.add_argument("--reference_mode", choices=["base", "init_snapshot"], default="base",
                    help="Reference policy: 'base' or 'init_snapshot' (frozen SFT, ρ≈1 at start)")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--output_dir", default="results/grpo/qwen2.5-7b-creditor")
    # LoRA (fresh-init only)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,v_proj,k_proj,o_proj")
    # GRPO
    ap.add_argument("--kl_beta", type=float, default=0.04)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--advantage_batch_norm", action="store_true")
    # Training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable

    # ===== Step 1: extract GRPO training data =====
    print("\n" + "=" * 70)
    print(f"📊 Step 1: Extract GRPO training pairs from {args.sweep_dir}")
    print("=" * 70)
    jsonl_path = os.path.join(args.sweep_dir, f"grpo_train_{timestamp}.jsonl")
    cmd = [
        py, "-m", "EmoDistill.grpo_data",
        "--sweep_dir", args.sweep_dir,
        "--out", jsonl_path,
        "--scenario_type", args.scenario_type,
        "--min_group_size", str(args.min_group_size),
        "--advantage_clip", str(args.advantage_clip),
        "--reward_field", args.reward_field,
    ]
    print("CMD:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=repo_root)
    if rc != 0:
        print("❌ Data extraction failed")
        sys.exit(rc)

    # ===== Step 2: offline GRPO training =====
    print("\n" + "=" * 70)
    print(f"🏋️  Step 2: Offline GRPO training")
    print("=" * 70)
    train_cmd = [
        py, "-m", "EmoDistill.grpo_train",
        "--train_jsonl", jsonl_path,
        "--base_model", args.base_model,
        "--output_dir", args.output_dir,
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--target_modules", args.target_modules,
        "--kl_beta", str(args.kl_beta),
        "--clip_eps", str(args.clip_eps),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--max_seq_len", str(args.max_seq_len),
        "--seed", str(args.seed),
    ]
    if args.sft_adapter:
        train_cmd += ["--sft_adapter", args.sft_adapter]
    if args.reference_mode != "base":
        train_cmd += ["--reference_mode", args.reference_mode]
    if args.load_in_8bit:
        train_cmd.append("--load_in_8bit")
    if args.advantage_batch_norm:
        train_cmd.append("--advantage_batch_norm")
    if args.max_samples:
        train_cmd += ["--max_samples", str(args.max_samples)]
    print("CMD:", " ".join(train_cmd))
    rc = subprocess.call(train_cmd, cwd=repo_root)
    if rc != 0:
        print("❌ GRPO training failed")
        sys.exit(rc)

    print("\n" + "=" * 70)
    print("🎉 Offline GRPO training complete!")
    print("=" * 70)
    print(f"   Adapter saved under: {args.output_dir}/run_*/adapter_final/")
    print("\nNext step (hierarchical eval, IQL meta + GRPO expression):")
    print(f"  python experiments/run_hierarchical_eval.py \\")
    print(f"      --iql_ckpt <path/to/iql.pt> \\")
    print(f"      --lora_adapter <path/to/grpo/adapter_final> \\")
    print(f"      --base_model {args.base_model} \\")
    print(f"      --dataset_type {args.scenario_type} \\")
    print(f"      --scenarios 20 --iterations 1 --offset 80")


if __name__ == "__main__":
    main()
