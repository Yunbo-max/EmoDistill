"""
Offline GRPO (Group-Relative Policy Optimization) trainer for the Qwen-7B creditor.

Reads JSONL produced by `EmoDistill.grpo_data` (each line carries a prompt,
the actually-uttered creditor response, and the group-relative advantage A
for that response) and fine-tunes a LoRA adapter on Qwen2.5-7B-Instruct.

GRPO loss (per response, mean over response tokens):

    L_pg = -E[ min( ρ · A , clip(ρ, 1-ε, 1+ε) · A ) ]
    L_kl = β · KL_K3(π_θ ‖ π_ref)
    L    = L_pg + L_kl

  ρ        = exp( mean_t [ log π_θ(y_t | x, y_<t) - log π_ref(y_t | x, y_<t) ] )
  KL_K3    = mean_t [ exp(Δ_t) - Δ_t - 1 ],   Δ_t = log π_ref - log π_θ
             (Schulman's K3 estimator — unbiased, always ≥ 0)
  A        = scalar advantage per response, precomputed by grpo_data.py

Why this is *offline* GRPO
--------------------------
Standard GRPO is on-policy: at each gradient step you sample G responses from
the current policy, score them with the reward model, normalise within the
group, and update. We can't afford live DashScope rollouts during training
(50× slower, costs real ¥), so we reuse the random_emotion_sweep — which
already contains 50 rollouts per scenario, each with a v4 reward. The
sweep is treated as the behaviour-policy buffer; the importance ratio ρ
keeps the update conservative against the policy that generated it.

Reference policy
----------------
We use the PEFT trick of disabling the LoRA adapter to get the base-model
log-probs — saves one full 14 GB copy in VRAM. The reference is therefore
the BASE Qwen2.5-7B (regardless of whether we warm-start the policy from an
SFT adapter). At step 0, with no SFT adapter, π_θ = π_ref exactly, so ρ = 1
and KL = 0 — clean. With an SFT adapter loaded, ρ starts off-1 but bounded;
the clip + KL keep updates safe.

Memory (24 GB GPU, BF16, bs=1, seq=2048, grad-checkpointing on)
---------------------------------------------------------------
  Base Qwen2.5-7B               ~14.0 GB
  LoRA (r=16) params + grads      ~1.5 GB
  Adam states (LoRA only)         ~0.5 GB
  Activations (policy + ref fwd)  ~3-4 GB
  ─────────────────────────────────────
  Total                          ~19-20 GB
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _response_token_logp(
    model,
    input_ids,
    attention_mask,
    response_mask,
):
    """Per-token log p(y_t | x, y_<t) at response positions only.

    Returns:
      gathered_logp : [B, T-1] float — log-prob at every position, masked to 0
                                       outside response tokens
      shift_resp    : [B, T-1] float — 1.0 at response token positions
      n_resp        : [B]      float — number of response tokens per row, ≥1
    """
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, T, V]
    # Position-t logits predict token t+1
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_resp = response_mask[:, 1:].to(shift_logits.dtype)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    gathered = gathered * shift_resp
    n_resp = shift_resp.sum(dim=-1).clamp(min=1.0)
    return gathered, shift_resp, n_resp


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    # Model
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_adapter", default=None,
                    help="Path to SFT LoRA adapter to warm-start GRPO (recommended)")
    ap.add_argument("--reference_mode", choices=["base", "init_snapshot"], default="base",
                    help="Reference policy for the importance ratio and KL anchor. "
                         "'base' = base model (LoRA disabled). 'init_snapshot' = a frozen "
                         "copy of the LoRA at training start (recommended when warm-starting "
                         "from --sft_adapter, so ρ ≈ 1 at step 0). 'init_snapshot' requires "
                         "--sft_adapter.")
    ap.add_argument("--output_dir", default="results/grpo/qwen2.5-7b-creditor")
    ap.add_argument("--load_in_8bit", action="store_true")
    # LoRA (used only if --sft_adapter not given)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,v_proj,k_proj,o_proj")
    # GRPO
    ap.add_argument("--kl_beta", type=float, default=0.04, help="KL penalty coefficient β")
    ap.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip range ε")
    ap.add_argument("--advantage_batch_norm", action="store_true",
                    help="Re-normalise advantages within each mini-batch (defensive — "
                         "data is already z-scored per scenario at data-prep time)")
    ap.add_argument("--kappa_neg", type=float, default=1.0,
                    help="Asymmetric advantage scaling for negative samples. "
                         "kappa=1.0 (default) = standard JPO; kappa=0.0 ≈ A-LoL (drops "
                         "negative-advantage pressure); intermediate kappa interpolates "
                         "the success-vs-per-deal-value frontier.")
    # Training
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
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
    with open(os.path.join(out_dir, "grpo_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    # Imports here to keep --help cheap
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

    torch.manual_seed(args.seed)

    # ===== Tokenizer =====
    print(f"📦 Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ===== Data =====
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

    class GRPODataset(Dataset):
        def __init__(self, pairs, tokenizer, max_seq_len):
            self.pairs = pairs
            self.tok = tokenizer
            self.max_len = max_seq_len

        def __len__(self):
            return len(self.pairs)

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
                # Truncation killed the response; skip by reusing index 0
                return self.__getitem__((idx + 1) % len(self.pairs))
            response_mask = [0] * n_prompt + [1] * n_resp
            return {
                "input_ids": input_ids,
                "response_mask": response_mask,
                "advantage": float(p["advantage"]),
            }

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn = torch.zeros(len(batch), max_len, dtype=torch.long)
        resp = torch.zeros(len(batch), max_len, dtype=torch.long)
        adv = torch.zeros(len(batch), dtype=torch.float32)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            attn[i, :L] = 1
            resp[i, :L] = torch.tensor(b["response_mask"], dtype=torch.long)
            adv[i] = b["advantage"]
        return {"input_ids": ids, "attention_mask": attn,
                "response_mask": resp, "advantage": adv}

    dataset = GRPODataset(pairs, tokenizer, args.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )

    # ===== Model =====
    print(f"📦 Loading base model: {args.base_model}")
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_8bit:
        base = prepare_model_for_kbit_training(base)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()

    use_reference_adapter = (args.reference_mode == "init_snapshot")
    if use_reference_adapter and not args.sft_adapter:
        raise ValueError("--reference_mode=init_snapshot requires --sft_adapter")

    if args.sft_adapter:
        print(f"📦 Warm-starting from SFT adapter: {args.sft_adapter}")
        model = PeftModel.from_pretrained(
            base, args.sft_adapter,
            is_trainable=True, adapter_name="default",
        )
        if use_reference_adapter:
            model.load_adapter(args.sft_adapter, adapter_name="reference", is_trainable=False)
            model.set_adapter("default")  # trainable adapter is the active one
            print(f"   Loaded SECOND copy as frozen reference adapter (init_snapshot mode)")
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

    # Required so PEFT registers the parameters for gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {n_trainable:,} ({n_trainable / max(1, n_total):.2%})")

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_optim_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)

    device = next(model.parameters()).device

    # ===== Training loop =====
    print(f"\n🚀 Offline GRPO training")
    print(f"   {len(loader)} mini-batches/epoch × {args.epochs} epochs")
    print(f"   grad_accum={args.grad_accum} → {total_optim_steps} optimizer steps")
    print(f"   kl_beta={args.kl_beta}  clip_eps={args.clip_eps}  lr={args.lr}")
    model.train()

    micro_step = 0
    optim_step = 0
    log_buf = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "ratio": 0.0,
               "adv_abs": 0.0, "n": 0}

    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            resp_mask = batch["response_mask"].to(device)
            adv = batch["advantage"].to(device)
            if args.advantage_batch_norm and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            if args.kappa_neg != 1.0:
                # Asymmetric scaling: keep positive samples at full weight,
                # scale negative samples by kappa_neg.
                #   kappa=0   → ≈ A-LoL (no negative-sample push, KL anchor still active)
                #   kappa=1   → standard JPO (default)
                #   kappa>1   → risk-seeking JPO (over-weight negative samples)
                adv = torch.where(adv > 0, adv, args.kappa_neg * adv)

            # Policy log-probs (LoRA active)
            logp_pol, shift_resp, n_resp = _response_token_logp(
                model, input_ids, attn, resp_mask,
            )
            # Reference log-probs. No grad.
            #   reference_mode=base           → disable LoRA → base-model forward
            #   reference_mode=init_snapshot  → switch to frozen 'reference' adapter
            with torch.no_grad():
                if use_reference_adapter:
                    model.set_adapter("reference")
                    logp_ref, _, _ = _response_token_logp(
                        model, input_ids, attn, resp_mask,
                    )
                    model.set_adapter("default")
                else:
                    with model.disable_adapter():
                        logp_ref, _, _ = _response_token_logp(
                            model, input_ids, attn, resp_mask,
                        )

            # log ratio per response token, masked
            log_ratio_tok = (logp_pol - logp_ref) * shift_resp  # [B, T-1]
            # mean over response tokens → per-response log-ratio
            mean_log_ratio = log_ratio_tok.sum(dim=-1) / n_resp  # [B]
            ratio = torch.exp(mean_log_ratio)                    # [B]

            # GRPO clipped PG (per-response — same A broadcast to all its tokens
            # is equivalent to mean-token ratio · A under length normalisation)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # K3 KL estimator (token-level mean, then batch mean)
            delta_tok = (logp_ref - logp_pol) * shift_resp
            kl_tok = (torch.exp(delta_tok) - delta_tok - 1.0) * shift_resp
            kl = (kl_tok.sum(dim=-1) / n_resp).mean()

            loss = pg_loss + args.kl_beta * kl
            (loss / args.grad_accum).backward()

            log_buf["loss"] += float(loss.item())
            log_buf["pg"] += float(pg_loss.item())
            log_buf["kl"] += float(kl.item())
            log_buf["ratio"] += float(ratio.mean().item())
            log_buf["adv_abs"] += float(adv.abs().mean().item())
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
                        "pg": log_buf["pg"] / n,
                        "kl": log_buf["kl"] / n,
                        "ratio": log_buf["ratio"] / n,
                        "adv_abs": log_buf["adv_abs"] / n,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    print(f"[ep{epoch} step {optim_step}/{total_optim_steps}] "
                          f"loss={row['loss']:+.4f}  pg={row['pg']:+.4f}  "
                          f"kl={row['kl']:.5f}  ρ={row['ratio']:.4f}  "
                          f"|A|={row['adv_abs']:.3f}  lr={row['lr']:.2e}")
                    with open(metrics_path, "a") as fmet:
                        fmet.write(json.dumps(row) + "\n")
                    log_buf = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "ratio": 0.0,
                               "adv_abs": 0.0, "n": 0}

                if args.save_every > 0 and optim_step % args.save_every == 0:
                    ckpt = os.path.join(out_dir, f"checkpoint-{optim_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    print(f"💾 checkpoint → {ckpt}")

    print("\n🏁 Offline GRPO training done.")
    final_dir = os.path.join(out_dir, "adapter_final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"💾 Final adapter → {final_dir}")
    print(f"📈 Metrics      → {metrics_path}")


if __name__ == "__main__":
    main()
