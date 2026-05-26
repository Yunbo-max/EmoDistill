"""
LoRA SFT trainer for the Qwen-7B creditor model.

Reads JSONL training pairs from lora_data.extract_training_pairs() and
fine-tunes Qwen2.5-7B-Instruct with a small low-rank adapter.

Pipeline:
1. Load Qwen2.5-7B-Instruct in BF16
2. Wrap with LoRA(r=16, alpha=32, target q/v projections)
3. Format each pair as chat-template:
   [system: creditor role prompt][user: scenario+history+emotion][assistant: target response]
4. Cross-entropy on assistant tokens only
5. Save LoRA adapter to disk

Memory(5090 24GB):
  Base Qwen2.5-7B BF16:        ~14 GB
  LoRA params + gradients:     ~2 GB
  Optimizer states (Adam):     ~4 GB
  Activations / batch:         ~4 GB
  ───────────────────────────────────
  Total:                       ~24 GB(刚好)

If OOM,降到 8-bit:bitsandbytes load_in_8bit=True,then LoRA on top → 10-12 GB.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True, help="JSONL from lora_data.py")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output_dir", default="results/lora/qwen2.5-7b-creditor")
    ap.add_argument("--load_in_8bit", action="store_true",
                    help="Use 8-bit quantization (saves ~50% VRAM)")
    # LoRA hyperparameters
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,v_proj,k_proj,o_proj",
                    help="Comma-separated module names")
    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Per-device batch size; effective = batch_size × grad_accum")
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--save_every", type=int, default=500, help="Save every N steps")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    # Quality of life
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap training set size (debug)")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "lora_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Imports here to avoid import cost when just using --help
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import trl
    from trl import SFTTrainer
    # SFTConfig name varies by trl version
    try:
        from trl import SFTConfig
        _USE_SFT_CONFIG = True
    except ImportError:
        _USE_SFT_CONFIG = False

    torch.manual_seed(args.seed)

    # ===== Load tokenizer =====
    print(f"📦 Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ===== Load training data =====
    print(f"📂 Loading training data: {args.train_jsonl}")
    pairs: List[Dict] = []
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    print(f"   {len(pairs)} training pairs")

    # Format as chat-template strings expected by SFTTrainer
    def format_pair(p: Dict) -> Dict:
        messages = [
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": p["response"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    formatted = [format_pair(p) for p in pairs]
    train_ds = Dataset.from_list(formatted)

    # ===== Load base model =====
    print(f"📦 Loading base model: {args.base_model}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # ===== Apply LoRA =====
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {n_trainable:,} ({n_trainable/n_total:.2%} of total)")

    # ===== Trainer =====
    if _USE_SFT_CONFIG:
        # Newer trl (>=0.12) — use SFTConfig directly
        sft_cfg = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            bf16=True,
            logging_steps=args.log_every,
            save_steps=args.save_every,
            save_total_limit=3,
            report_to="none",
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
            packing=False,
            remove_unused_columns=False,
            seed=args.seed,
        )
        try:
            trainer = SFTTrainer(
                model=model,
                args=sft_cfg,
                train_dataset=train_ds,
                processing_class=tokenizer,
            )
        except TypeError:
            # Older API uses `tokenizer`
            trainer = SFTTrainer(
                model=model,
                args=sft_cfg,
                train_dataset=train_ds,
                tokenizer=tokenizer,
            )
    else:
        # Old trl (<0.12) — use TrainingArguments + SFTTrainer params
        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            bf16=True,
            logging_steps=args.log_every,
            save_steps=args.save_every,
            save_total_limit=3,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=False,
        )

    print(f"\n🚀 Training start. Output → {out_dir}")
    trainer.train()
    print(f"\n🏁 Training done.")

    # Save final adapter
    final_dir = os.path.join(out_dir, "adapter_final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"💾 Saved LoRA adapter → {final_dir}")


if __name__ == "__main__":
    main()
