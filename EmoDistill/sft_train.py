"""
Custom SFT trainer for the Qwen-7B creditor LoRA — does NOT use TRL or HF
Trainer. Mirrors EmoDistill.grpo_train architecture (data loader, model
setup, custom loop) so it works in environments where TRL/HF Trainer are
fragile (RTX 4090 + transformers 4.44 + accelerate compatibility issues).

Reads JSONL produced by `EmoDistill.lora_data` (top-K% sweep responses) and
applies plain cross-entropy on response tokens only (prompt tokens masked).

Loss
----
    L = -E[ (1/|y|) Σ_t log π_θ(y_t | x, y_<t) ]    response-only mean-CE

No advantages, no KL anchor, no reference forward. Just imitation of the
top-K% high-quality (prompt, response) pairs.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train_jsonl", required=True, help="JSONL from EmoDistill.lora_data")
    ap.add_argument("--max_samples", type=int, default=None)
    # Model
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--output_dir", default="results/lora_sft/qwen2.5-7b-creditor")
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
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sft_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(args.seed)

    print(f"📦 Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    class SFTDataset(Dataset):
        def __init__(self, pairs, tok, max_len):
            self.pairs = pairs; self.tok = tok; self.max_len = max_len
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            p = self.pairs[idx]
            prompt_text = self.tok.apply_chat_template(
                [{"role": "user", "content": p["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            response_text = p["response"]
            if not response_text.endswith(self.tok.eos_token):
                response_text = response_text + self.tok.eos_token
            full_text = prompt_text + response_text
            prompt_ids = self.tok(prompt_text, add_special_tokens=False).input_ids
            full_enc = self.tok(full_text, add_special_tokens=False,
                                truncation=True, max_length=self.max_len)
            input_ids = full_enc.input_ids
            n_prompt = min(len(prompt_ids), len(input_ids))
            n_resp = len(input_ids) - n_prompt
            if n_resp <= 0:
                return self.__getitem__((idx + 1) % len(self.pairs))
            response_mask = [0] * n_prompt + [1] * n_resp
            return {"input_ids": input_ids, "response_mask": response_mask}

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros(len(batch), max_len, dtype=torch.long)
        resp = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            attn[i, :L] = 1
            resp[i, :L] = torch.tensor(b["response_mask"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "response_mask": resp}

    dataset = SFTDataset(pairs, tokenizer, args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2, drop_last=True)

    print(f"📦 Loading base model: {args.base_model}")
    model_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_8bit:
        base = prepare_model_for_kbit_training(base)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=target_modules,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {n_train:,} ({n_train/max(1,n_total):.2%})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = next(model.parameters()).device

    print(f"\n🚀 SFT training")
    print(f"   {len(loader)} mini-batches/epoch × {args.epochs} epochs")
    print(f"   grad_accum={args.grad_accum} → {total_steps} optimizer steps")
    print(f"   lr={args.lr}")
    model.train()

    micro_step = 0
    optim_step = 0
    log_buf = {"loss": 0.0, "n_resp": 0.0, "n": 0}

    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            resp_mask = batch["response_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = outputs.logits[:, :-1, :]                           # [B, T-1, V]
            shift_labels = input_ids[:, 1:]                              # [B, T-1]
            shift_resp = resp_mask[:, 1:].to(logits.dtype)               # [B, T-1]

            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            gathered = gathered * shift_resp
            n_resp = shift_resp.sum(dim=-1).clamp(min=1.0)
            # mean CE over response tokens, then mean over batch
            loss = -(gathered.sum(dim=-1) / n_resp).mean()

            (loss / args.grad_accum).backward()
            log_buf["loss"] += float(loss.item())
            log_buf["n_resp"] += float(n_resp.mean().item())
            log_buf["n"] += 1
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                if optim_step % args.log_every == 0:
                    n = max(1, log_buf["n"])
                    row = {
                        "epoch": epoch,
                        "optim_step": optim_step,
                        "loss": log_buf["loss"] / n,
                        "mean_resp_tokens": log_buf["n_resp"] / n,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    print(f"[ep{epoch} step {optim_step}/{total_steps}] "
                          f"loss={row['loss']:+.4f}  resp_tok={row['mean_resp_tokens']:.0f}  "
                          f"lr={row['lr']:.2e}", flush=True)
                    with open(metrics_path, "a") as fmet:
                        fmet.write(json.dumps(row) + "\n")
                    log_buf = {"loss": 0.0, "n_resp": 0.0, "n": 0}

                if args.save_every > 0 and optim_step % args.save_every == 0:
                    ckpt = os.path.join(out_dir, f"checkpoint-{optim_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"💾 checkpoint → {ckpt}", flush=True)

    print("\n🏁 SFT training done.", flush=True)
    final_dir = os.path.join(out_dir, "adapter_final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"💾 Final adapter → {final_dir}", flush=True)


if __name__ == "__main__":
    main()
