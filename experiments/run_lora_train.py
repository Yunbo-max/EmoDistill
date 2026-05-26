#!/usr/bin/env python3
"""
End-to-end LoRA training pipeline:

1. Extract training pairs from random sweep
2. Train LoRA adapter on Qwen2.5-7B
3. Save adapter for hierarchical eval

Example:
  python experiments/run_lora_train.py \\
      --sweep_dir results/random_emotion_sweep/debt_goemotions27_<TS>/ \\
      --base_model Qwen/Qwen2.5-7B-Instruct \\
      --top_k_percent 0.5 \\
      --epochs 2 --batch_size 2 --grad_accum 8
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
    ap.add_argument("--top_k_percent", type=float, default=0.5)
    ap.add_argument("--scenario_type", default="debt")
    # Model
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load_in_8bit", action="store_true")
    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,v_proj,k_proj,o_proj")
    # Training
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/lora/qwen2.5-7b-creditor")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = sys.executable

    # ===== Step 1: extract training pairs =====
    print("\n" + "=" * 70)
    print(f"📊 Step 1: Extract LoRA training pairs from {args.sweep_dir}")
    print("=" * 70)
    jsonl_path = os.path.join(args.sweep_dir, f"lora_train_{timestamp}.jsonl")
    cmd = [
        py, "-m", "EmoDistill.lora_data",
        "--sweep_dir", args.sweep_dir,
        "--out", jsonl_path,
        "--top_k_percent", str(args.top_k_percent),
        "--scenario_type", args.scenario_type,
    ]
    print("CMD:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=repo_root)
    if rc != 0:
        print("❌ Data extraction failed")
        sys.exit(rc)

    # ===== Step 2: LoRA SFT training =====
    print("\n" + "=" * 70)
    print(f"🏋️  Step 2: LoRA SFT training")
    print("=" * 70)
    train_cmd = [
        py, "-m", "EmoDistill.lora_train",
        "--train_jsonl", jsonl_path,
        "--base_model", args.base_model,
        "--output_dir", args.output_dir,
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--target_modules", args.target_modules,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--max_seq_len", str(args.max_seq_len),
        "--seed", str(args.seed),
    ]
    if args.load_in_8bit:
        train_cmd.append("--load_in_8bit")
    if args.max_samples:
        train_cmd += ["--max_samples", str(args.max_samples)]
    print("CMD:", " ".join(train_cmd))
    rc = subprocess.call(train_cmd, cwd=repo_root)
    if rc != 0:
        print("❌ Training failed")
        sys.exit(rc)

    print("\n" + "=" * 70)
    print("🎉 LoRA training complete!")
    print("=" * 70)
    print(f"   Adapter saved under: {args.output_dir}/run_*/adapter_final/")
    print("\nNext steps:")
    print("  Hierarchical eval on held-out:")
    print(f"  python experiments/run_hierarchical_eval.py \\")
    print(f"      --iql_ckpt <path/to/iql.pt> \\")
    print(f"      --lora_adapter <path/to/adapter_final> \\")
    print(f"      --base_model {args.base_model} \\")
    print(f"      --dataset_type {args.scenario_type} \\")
    print(f"      --scenarios 20 --iterations 1 --offset 80")


if __name__ == "__main__":
    main()
