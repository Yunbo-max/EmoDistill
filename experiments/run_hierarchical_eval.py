#!/usr/bin/env python3
"""
Hierarchical eval: IQL meta-policy + LoRA-fine-tuned creditor LLM.

For each held-out scenario:
  Turn t:
    state_t = build_state(dialog up to t)
    emotion_t = IQL.select_emotion(state_t)          ← meta level
    creditor_utterance = LoRA-Qwen.generate(         ← expression level
        scenario, history, emotion=emotion_t
    )
    debtor responds (could use API or local Qwen)

Outputs:
  - per-episode success / reward / emotion sequence
  - per-scenario aggregate
  - comparison-ready format (vs IQL alone, vs fixed emotion, etc.)

Example:
  python experiments/run_hierarchical_eval.py \\
      --iql_ckpt results/iql/.../iql_xxx.pt \\
      --lora_adapter results/lora/.../adapter_final \\
      --base_model Qwen/Qwen2.5-7B-Instruct \\
      --dataset_type debt \\
      --scenarios 20 --iterations 1 --offset 80 \\
      --debtor_model qwen-plus \\
      --max_dialog_len 30 --concurrency 1
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dotenv import load_dotenv

load_dotenv()


DATASET_PATHS = {
    "debt": "data/credit_recovery_scenarios.csv",
    "disaster": "data/disaster_survivor_scenarios.csv",
    "student": "data/education_sleep_scenarios.csv",
    "medical": "data/hospital_surgery_scenarios.csv",
}


def main():
    p = argparse.ArgumentParser(description="Hierarchical IQL + LoRA-Qwen eval")
    p.add_argument("--iql_ckpt", default=None, help="IQL checkpoint .pt (or use --meta_grpo_ckpt)")
    p.add_argument("--meta_grpo_ckpt", default=None,
                   help="Meta-GRPO checkpoint .pt (alternative to --iql_ckpt)")
    p.add_argument("--lora_adapter", default=None,
                   help="LoRA adapter dir (omit for base-model-only ablation)")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--load_in_8bit", action="store_true")

    p.add_argument("--dataset_type", default="debt", choices=list(DATASET_PATHS.keys()))
    p.add_argument("--scenarios", type=int, default=20)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--offset", type=int, default=80, help="Skip first N scenarios (train set)")
    p.add_argument("--max_dialog_len", type=int, default=30)

    # Debtor can be local (same Qwen) or API
    p.add_argument("--debtor_model", default="qwen-plus",
                   help="Debtor LLM (API model name); use 'local' to share creditor's Qwen")

    p.add_argument("--concurrency", type=int, default=1,
                   help="LoRA inference is GPU-bound; usually 1")
    p.add_argument("--out_dir", default="results/hierarchical_eval")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ===== Set up taxonomy =====
    if not args.iql_ckpt and not args.meta_grpo_ckpt:
        raise SystemExit("Must pass --iql_ckpt or --meta_grpo_ckpt")
    if args.iql_ckpt and args.meta_grpo_ckpt:
        raise SystemExit("Pass exactly one of --iql_ckpt or --meta_grpo_ckpt")
    meta_ckpt = args.iql_ckpt or args.meta_grpo_ckpt
    use_meta_grpo = bool(args.meta_grpo_ckpt)
    import torch as _torch
    ckpt = _torch.load(meta_ckpt, map_location="cpu", weights_only=False)
    taxonomy = ckpt.get("training", {}).get("taxonomy", "goemotions27")
    os.environ["EVOEMO_TAXONOMY"] = taxonomy

    from EmoDistill.emotions import set_active_taxonomy, get_emotions
    set_active_taxonomy(taxonomy)
    print(f"🎯 Active taxonomy: {taxonomy} ({len(get_emotions())} emotions)")

    # ===== Load scenarios =====
    from utils.preprocessing import preprocess_all_scenarios

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, DATASET_PATHS[args.dataset_type])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"{args.dataset_type}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = preprocess_all_scenarios(
        csv_path=csv_path,
        scenario_type=args.dataset_type,
        output_path=os.path.join(out_dir, "scenarios.json"),
        n_scenarios=args.scenarios + args.offset,
    )
    scenarios = scenarios[args.offset : args.offset + args.scenarios]
    print(f"✅ Held-out: {len(scenarios)} scenarios (offset={args.offset})")

    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ===== Load LoRA-wrapped creditor =====
    from EmoDistill.lora_negotiator import LoRAWrapper
    print(f"\n📦 Loading LoRA creditor:")
    print(f"   Base:    {args.base_model}")
    print(f"   Adapter: {args.lora_adapter}")
    creditor_llm = LoRAWrapper(
        base_model=args.base_model,
        adapter_path=args.lora_adapter,
        load_in_8bit=args.load_in_8bit,
        role="creditor",
    )
    # Neuter cleanup on the SHARED wrapper so per-negotiation Negotiator.__del__
    # (which calls llm_creditor.cleanup() and would `del self.model`) cannot
    # poison the wrapper for subsequent threads. We keep a real-cleanup handle.
    creditor_llm._real_cleanup = creditor_llm.cleanup
    creditor_llm.cleanup = lambda: None

    # ===== Run negotiations =====
    from EmoDistill.iql import IQLPolicy
    import random as _r
    _r.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n🧪 Hierarchical eval | {len(scenarios) * args.iterations} negotiations")

    tasks = [(s, it) for s in scenarios for it in range(args.iterations)]
    results = []
    t0 = time.time()

    def run_one(scenario, iteration):
        return _eval_single(
            scenario=scenario,
            iteration=iteration,
            iql_ckpt=args.iql_ckpt,
            taxonomy=taxonomy,
            creditor_llm=creditor_llm,            # shared GPU model
            debtor_model_name=args.debtor_model,
            max_dialog_len=args.max_dialog_len,
        )

    if args.concurrency <= 1:
        # Sequential (LoRA inference on single GPU)
        for i, (sc, it) in enumerate(tasks):
            r = run_one(sc, it)
            results.append(r)
            print(f"   [{i+1}/{len(tasks)}] sc={r['scenario']} it={r['iteration']} "
                  f"success={r['success']} R={r['total_episode_reward']:.3f} "
                  f"rounds={r['rounds']} unique_emos={len(set(r['emotion_sequence']))}")
    else:
        # Parallel — only safe if creditor_llm supports it (usually not for GPU)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(run_one, sc, it) for sc, it in tasks]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                results.append(r)
                print(f"   [{i+1}/{len(tasks)}] sc={r['scenario']} it={r['iteration']} "
                      f"success={r['success']} R={r['total_episode_reward']:.3f}")

    # ===== Aggregate =====
    successes = [r for r in results if r["success"]]
    success_rate = len(successes) / max(1, len(results))
    mean_R = float(np.mean([r["total_episode_reward"] for r in results])) if results else 0.0
    sv = [r["savings_ratio"] for r in results if r["savings_ratio"] is not None]
    mean_savings = float(np.mean(sv)) if sv else 0.0
    mean_total_conc = float(np.mean([r["total_debtor_concession_norm"] for r in results])) if results else 0.0
    unique_emos = [len(set(r["emotion_sequence"])) for r in results if r["emotion_sequence"]]
    mean_unique = float(np.mean(unique_emos)) if unique_emos else 0.0

    summary = {
        "iql_ckpt": args.iql_ckpt,
        "lora_adapter": args.lora_adapter,
        "base_model": args.base_model,
        "taxonomy": taxonomy,
        "n_episodes": len(results),
        "success_rate": success_rate,
        "mean_episode_reward": mean_R,
        "mean_total_concession": mean_total_conc,
        "mean_savings_ratio": mean_savings,
        "mean_unique_emotions_per_episode": mean_unique,
        "config": vars(args),
        "per_episode": results,
    }
    out_path = os.path.join(out_dir, f"hierarchical_eval_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"🏁 HIERARCHICAL EVAL — {len(results)} negotiations in {elapsed/60:.1f} min")
    print("=" * 80)
    print(f"  Success rate:                    {success_rate:.1%}")
    print(f"  Mean episode reward (v4):        {mean_R:.3f}")
    print(f"  Mean total concession:           {mean_total_conc:.3f}")
    print(f"  Mean savings ratio:              {mean_savings:.3f}")
    print(f"  Mean unique emotions / episode:  {mean_unique:.2f}")
    print(f"💾 Saved → {out_path}")


def _eval_single(
    scenario,
    iteration,
    iql_ckpt,
    taxonomy,
    creditor_llm,
    debtor_model_name,
    max_dialog_len,
):
    """Run one hierarchical negotiation episode."""
    from EmoDistill.iql import IQLPolicy
    from EmoDistill.negotiator_new import NegotiatorNew
    from EmoDistill.fixed_emotion_baseline import _compute_reward_metrics

    # Build IQL policy (meta-level)
    policy = IQLPolicy(
        ckpt_path=iql_ckpt,
        taxonomy=taxonomy,
        greedy=False,
        temperature=1.0,
    )

    # Build NegotiatorNew with LoRA-wrapped creditor and API debtor
    negotiator = NegotiatorNew(
        config=scenario,
        emotion_model=policy,
        model_creditor=debtor_model_name,   # placeholder; we replace below
        model_debtor=debtor_model_name,
        debtor_emotion="neutral",
        use_observer=False,
        max_dialog_len=max_dialog_len,
    )

    # OVERRIDE creditor LLM with shared LoRA wrapper (avoid reloading per episode)
    if hasattr(negotiator.llm_creditor, "cleanup"):
        try:
            negotiator.llm_creditor.cleanup()
        except Exception:
            pass
    negotiator.llm_creditor = creditor_llm  # shared GPU model

    try:
        result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
    except Exception as e:
        print(f"   ⚠️  scenario={scenario.get('id')} iter={iteration} failed: {e}")
        result = {
            "final_state": "breakdown", "final_days": None,
            "dialog": [], "emotion_sequence": [], "negotiation_rounds": 0,
            "error": str(e),
        }

    creditor_target = int(scenario.get("seller", {}).get("target_price", 30))
    debtor_initial = int(scenario.get("buyer", {}).get("target_price", creditor_target * 3))
    rmetrics = _compute_reward_metrics(
        result, creditor_target, debtor_initial, max_turn=max_dialog_len
    )

    return {
        "scenario": scenario.get("id"),
        "iteration": iteration,
        "success": result.get("final_state") == "accept",
        "final_days": result.get("final_days"),
        "creditor_target_days": result.get("creditor_target_days"),
        "rounds": result.get("negotiation_rounds", 0),
        "emotion_sequence": result.get("emotion_sequence", []),
        "dialog": result.get("dialog", []),
        "total_episode_reward": rmetrics["total_episode_reward"],
        "first_step_concession_norm": rmetrics["first_step_concession_norm"],
        "total_debtor_concession_norm": rmetrics["total_debtor_concession_norm"],
        "savings_ratio": rmetrics["savings_ratio"],
    }


if __name__ == "__main__":
    main()
