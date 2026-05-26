"""
LoRA training data extraction from random-emotion sweep results.

Pipeline:
1. Read random sweep JSON(episode_results with full dialog + step_rewards)
2. For each creditor turn, build (input_prompt, target_response) pair
   where input_prompt = system + scenario + history + emotion instruction
3. Compute per-turn quality score from v4 step reward + future reward
4. Filter to top-K% high-quality pairs
5. Output JSONL ready for HuggingFace SFTTrainer
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from llm.prompt_templates import PromptTemplates
from EmoDistill.emotions import prompt_for, parse_emotion_str, get_emotions


def _build_creditor_prompt(
    scenario: Dict[str, Any],
    dialog_history: List[Tuple[str, str]],
    emotion: str,
    scenario_type: str = "debt",
    time_unit: str = "days",
) -> str:
    """Reconstruct the creditor prompt that was used during this turn.

    This must match what NegotiatorNew / DebtNegotiator builds at runtime, so
    the LoRA-finetuned model sees the exact same input format during training
    and at inference.
    """
    # Build full transcript like negotiator does
    if not dialog_history:
        timeline_text = "This is the start of the negotiation."
    else:
        lines = ["NEGOTIATION HISTORY (full transcript so far):"]
        for speaker, message in dialog_history:
            label = "You (Creditor)" if speaker == "seller" else "Debtor"
            lines.append(f"{label}: {(message or '').strip()}")
        timeline_text = "\n".join(lines)

    # Build emotion_config for prompt template
    emotion_config = {
        "emotion": emotion,
        "emotion_text": prompt_for(emotion),
        "temperature": 0.7,
    }

    config = scenario.get("seller_config", scenario.get("seller", {}))
    debt_info = scenario.get("metadata", {})

    prompt = PromptTemplates.get_creditor_prompt(
        scenario_type=scenario_type,
        config=config,
        emotion_config=emotion_config,
        timeline_text=timeline_text,
        debt_info=debt_info,
    )
    return prompt


def extract_training_pairs(
    sweep_path: str,
    scenarios_path: str,
    reward_threshold: Optional[float] = None,
    top_k_percent: float = 0.5,
    scenario_type: str = "debt",
) -> List[Dict[str, Any]]:
    """Extract (input_prompt, target_response) pairs from a sweep dir.

    Args:
      sweep_path: random_sweep_*.json or partial_results.json
      scenarios_path: scenarios.json from same sweep dir
      reward_threshold: drop turns with step_reward < this; if None, use top_k_percent
      top_k_percent: keep top fraction by per-turn quality

    Returns: list of dicts {prompt, response, emotion, scenario, reward, quality}
    """
    print(f"📂 Loading sweep: {sweep_path}")
    with open(sweep_path) as f:
        sweep = json.load(f)

    print(f"📂 Loading scenarios: {scenarios_path}")
    with open(scenarios_path) as f:
        scenarios_list = json.load(f)
    scenarios_by_id = {s.get("id"): s for s in scenarios_list}

    episodes = sweep.get("episode_results") or sweep.get("episodes") or []
    print(f"   {len(episodes)} episodes loaded")

    pairs = []
    for ep in episodes:
        dialog = ep.get("dialog", [])
        emotion_seq = ep.get("emotion_sequence", [])
        step_rewards = ep.get("step_rewards", [])
        total_R = ep.get("total_episode_reward", 0.0)
        success = ep.get("success", False)
        scenario_id = ep.get("scenario")
        scenario = scenarios_by_id.get(scenario_id)
        if scenario is None or not dialog:
            continue

        # Find creditor turn positions in dialog
        creditor_turns = [i for i, d in enumerate(dialog) if d.get("speaker") == "seller"]

        for k, ci in enumerate(creditor_turns):
            if k >= len(emotion_seq):
                break
            emotion = emotion_seq[k]
            # Dialog history BEFORE this creditor turn
            dialog_before = [(d["speaker"], d["message"]) for d in dialog[:ci]]
            creditor_msg = dialog[ci].get("message", "").strip()
            if not creditor_msg:
                continue

            step_r = step_rewards[k] if k < len(step_rewards) else 0.0

            # Quality score: per-step reward + 0.5 × episode-level reward (so high-success
            # episodes' steps get boost even if individual step is mid).
            quality = float(step_r) + 0.5 * float(total_R)

            prompt = _build_creditor_prompt(
                scenario=scenario,
                dialog_history=dialog_before,
                emotion=emotion,
                scenario_type=scenario_type,
            )

            pairs.append({
                "prompt": prompt,
                "response": creditor_msg,
                "emotion": emotion,
                "scenario": scenario_id,
                "iteration": ep.get("iteration"),
                "turn_idx": k,
                "step_reward": float(step_r),
                "episode_reward": float(total_R),
                "episode_success": bool(success),
                "quality": float(quality),
            })

    print(f"✅ Extracted {len(pairs)} (prompt, response) pairs")

    # Filter
    if reward_threshold is not None:
        before = len(pairs)
        pairs = [p for p in pairs if p["quality"] >= reward_threshold]
        print(f"   Filter quality >= {reward_threshold}: {before} → {len(pairs)}")
    elif top_k_percent is not None and top_k_percent < 1.0:
        pairs.sort(key=lambda p: -p["quality"])
        cutoff = int(len(pairs) * top_k_percent)
        before = len(pairs)
        pairs = pairs[:cutoff]
        print(f"   Top {top_k_percent:.0%} by quality: {before} → {len(pairs)}")
        if pairs:
            print(f"   Quality range kept: [{pairs[-1]['quality']:.3f}, {pairs[0]['quality']:.3f}]")

    # Stats
    by_emo: Dict[str, int] = {}
    for p in pairs:
        emo = p["emotion"] if p["emotion"] is not None else "<none>"
        by_emo[emo] = by_emo.get(emo, 0) + 1
    print(f"   Per-emotion pair count:")
    for e, c in sorted(by_emo.items(), key=lambda x: -x[1]):
        print(f"     {str(e):<16} {c}")

    return pairs


def write_jsonl(pairs: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"💾 Wrote {len(pairs)} pairs → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True, help="Random sweep output dir")
    ap.add_argument("--out", default=None, help="Output JSONL path (default: alongside sweep)")
    ap.add_argument("--reward_threshold", type=float, default=None,
                    help="Filter quality >= this; takes precedence over --top_k_percent")
    ap.add_argument("--top_k_percent", type=float, default=0.5,
                    help="Keep top fraction by quality (default 0.5)")
    ap.add_argument("--scenario_type", type=str, default="debt")
    args = ap.parse_args()

    # Find sweep JSON: support random_sweep_*, sweep_iql_*, sweep_self_* (exclude _judged variants)
    import glob
    candidates = []
    for pat in ("random_sweep_*.json", "sweep_iql_*.json", "sweep_self_*.json"):
        for p in glob.glob(os.path.join(args.sweep_dir, pat)):
            if "_judged" in os.path.basename(p):
                continue
            candidates.append(p)
    candidates.sort()
    sweep_path = candidates[-1] if candidates else os.path.join(args.sweep_dir, "partial_results.json")
    scenarios_path = os.path.join(args.sweep_dir, "scenarios.json")

    pairs = extract_training_pairs(
        sweep_path=sweep_path,
        scenarios_path=scenarios_path,
        reward_threshold=args.reward_threshold,
        top_k_percent=args.top_k_percent,
        scenario_type=args.scenario_type,
    )

    out = args.out
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.sweep_dir, f"lora_train_{ts}.jsonl")
    write_jsonl(pairs, out)


if __name__ == "__main__":
    main()
