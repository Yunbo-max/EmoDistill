"""
GRPO (Group-Relative Policy Optimization) offline-RL training-data extraction.

Pipeline:
1. Read random-emotion sweep JSON (same source as lora_data.py)
2. For every creditor turn, build the same (prompt, response) pair the LoRA
   model is trained on
3. Group episodes by scenario_id — each scenario was rolled out N≈50 times
   with different random emotion sequences, giving a natural GRPO group
4. Compute group-relative advantage  A = (R_episode − μ_group) / (σ_group + ε)
   and attach the same advantage to every creditor turn in that episode
   (DeepSeek-R1-style: one scalar advantage per sampled response, broadcast
   to all of its tokens)
5. Write JSONL ready for grpo_train.py

Why scenario-level groups
-------------------------
- Same scenario  →  same starting context, same debtor type, same target
  amount  →  rollouts are comparable
- ~50 iterations per scenario  →  stable group mean/std
- Across scenarios, the achievable reward varies (easy vs hard debtors).
  Normalising per-scenario removes that task-difficulty confound from the
  policy-gradient signal.

Why episode-level (not turn-level) advantage
--------------------------------------------
- After turn 0 the prompt is dialog-conditioned, so the SAME (prompt, action)
  group does not exist across iterations.
- The reward signal is sparse-ish: the meaningful information lives in the
  whole-episode outcome. Broadcasting one advantage to all turns of an
  episode is exactly what GRPO does for whole-response sampling in DeepSeek-R1.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from llm.prompt_templates import PromptTemplates
from EmoDistill.emotions import prompt_for


def _build_creditor_prompt(
    scenario: Dict[str, Any],
    dialog_history: List[Tuple[str, str]],
    emotion: str,
    scenario_type: str = "debt",
) -> str:
    """Reconstruct the creditor prompt exactly as the negotiator built it at sweep time.

    Mirrors lora_data._build_creditor_prompt so SFT and GRPO see identical inputs.
    """
    if not dialog_history:
        timeline_text = "This is the start of the negotiation."
    else:
        lines = ["NEGOTIATION HISTORY (full transcript so far):"]
        for speaker, message in dialog_history:
            label = "You (Creditor)" if speaker == "seller" else "Debtor"
            lines.append(f"{label}: {(message or '').strip()}")
        timeline_text = "\n".join(lines)

    emotion_config = {
        "emotion": emotion,
        "emotion_text": prompt_for(emotion),
        "temperature": 0.7,
    }

    config = scenario.get("seller_config", scenario.get("seller", {}))
    debt_info = scenario.get("metadata", {})
    return PromptTemplates.get_creditor_prompt(
        scenario_type=scenario_type,
        config=config,
        emotion_config=emotion_config,
        timeline_text=timeline_text,
        debt_info=debt_info,
    )


def extract_grpo_pairs(
    sweep_path: str,
    scenarios_path: str,
    scenario_type: str = "debt",
    min_group_size: int = 4,
    advantage_clip: float = 10.0,
    reward_field: str = "total_episode_reward",
    per_turn_reward_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract (prompt, response, group_advantage) tuples for offline GRPO.

    Args:
      sweep_path:      random_sweep_*.json (or partial_results.json) from a sweep dir
      scenarios_path:  scenarios.json from the same sweep dir
      min_group_size:  drop scenarios with fewer than this many iterations
                       (need ≥2 for meaningful std, ≥4 for robust z-score)
      advantage_clip:  clip |A| to this magnitude (after z-score). Prevents the
                       few-outlier rollouts from dominating gradients.
      reward_field:    which per-episode reward to use as the group statistic
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

    # --- Step A: compute per-scenario group statistics ----------------------
    # If per_turn_reward_field is set, we collect ALL per-turn scores into the
    # scenario pool (turn-level z-score). Otherwise we use episode-level rewards.
    by_scenario: Dict[Any, List[float]] = defaultdict(list)
    for ep in episodes:
        sid = ep.get("scenario")
        if sid is None:
            continue
        if per_turn_reward_field:
            turn_scores = ep.get(per_turn_reward_field, []) or []
            by_scenario[sid].extend(float(x) for x in turn_scores)
        else:
            by_scenario[sid].append(float(ep.get(reward_field, 0.0)))

    group_stats: Dict[Any, Tuple[float, float]] = {}
    for sid, rs in by_scenario.items():
        if len(rs) < min_group_size:
            continue
        mu = float(np.mean(rs))
        sigma = float(np.std(rs))
        group_stats[sid] = (mu, max(sigma, 1e-6))

    print(f"   {len(group_stats)} / {len(by_scenario)} scenarios "
          f"retained (≥{min_group_size} iterations)")

    # --- Step B: build per-turn pairs with the broadcast advantage ----------
    pairs: List[Dict[str, Any]] = []
    skipped_no_group = 0
    skipped_empty = 0
    for ep in episodes:
        sid = ep.get("scenario")
        if sid not in group_stats:
            skipped_no_group += 1
            continue
        mu, sigma = group_stats[sid]
        ep_reward = float(ep.get(reward_field, 0.0))
        # Episode-level fallback advantage (used if per_turn_reward_field is None
        # or the per-turn score list is short).
        ep_advantage = float(np.clip((ep_reward - mu) / sigma, -advantage_clip, advantage_clip))

        dialog = ep.get("dialog", [])
        emotion_seq = ep.get("emotion_sequence", [])
        per_turn_scores = ep.get(per_turn_reward_field, []) if per_turn_reward_field else []
        scenario = scenarios_by_id.get(sid)
        if scenario is None or not dialog:
            skipped_empty += 1
            continue

        creditor_turns = [i for i, d in enumerate(dialog) if d.get("speaker") == "seller"]
        for k, ci in enumerate(creditor_turns):
            if k >= len(emotion_seq):
                break
            emotion = emotion_seq[k]
            dialog_before = [(d["speaker"], d["message"]) for d in dialog[:ci]]
            creditor_msg = dialog[ci].get("message", "").strip()
            if not creditor_msg:
                continue
            prompt = _build_creditor_prompt(scenario, dialog_before, emotion, scenario_type)

            # Decide which advantage to attach to this turn
            if per_turn_reward_field and k < len(per_turn_scores):
                raw = float(per_turn_scores[k])
                advantage = float(np.clip((raw - mu) / sigma, -advantage_clip, advantage_clip))
            else:
                advantage = ep_advantage

            pairs.append({
                "prompt": prompt,
                "response": creditor_msg,
                "emotion": emotion,
                "scenario": sid,
                "iteration": ep.get("iteration"),
                "turn_idx": k,
                "episode_reward": ep_reward,
                "group_mean": float(mu),
                "group_std": float(sigma),
                "advantage": advantage,
                "success": bool(ep.get("success", False)),
            })

    if skipped_no_group:
        print(f"   Skipped {skipped_no_group} episodes (scenario group too small)")
    if skipped_empty:
        print(f"   Skipped {skipped_empty} episodes (no scenario or empty dialog)")

    print(f"✅ Extracted {len(pairs)} (prompt, response) pairs")

    # --- Step C: diagnostics ------------------------------------------------
    if pairs:
        adv_arr = np.array([p["advantage"] for p in pairs])
        print(f"   Advantage range: [{adv_arr.min():+.2f}, {adv_arr.max():+.2f}]")
        print(f"   Advantage mean : {adv_arr.mean():+.3f}  (≈0 by construction)")
        print(f"   Advantage std  : {adv_arr.std():.3f}   (≈1 by construction)")
        pos = int((adv_arr > 0).sum())
        neg = int((adv_arr < 0).sum())
        zero = int((adv_arr == 0).sum())
        N = len(pairs)
        print(f"   Sign split: +{pos} ({pos/N:.1%}) / -{neg} ({neg/N:.1%}) / 0={zero}")

        by_emo: Dict[str, int] = {}
        by_emo_adv: Dict[str, List[float]] = defaultdict(list)
        for p in pairs:
            emo = p["emotion"] if p["emotion"] is not None else "<none>"
            by_emo[emo] = by_emo.get(emo, 0) + 1
            by_emo_adv[emo].append(p["advantage"])
        print("   Per-emotion (count, mean advantage):")
        for e, c in sorted(by_emo.items(), key=lambda x: -x[1]):
            print(f"     {str(e):<16} {c:>6}   meanA={np.mean(by_emo_adv[e]):+.3f}")

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
    ap.add_argument("--sweep_json", default=None,
                    help="Override sweep JSON path (e.g. judge-rewarded variant); "
                         "default: latest random_sweep_*.json in --sweep_dir")
    ap.add_argument("--out", default=None, help="Output JSONL (default: alongside sweep)")
    ap.add_argument("--scenario_type", default="debt")
    ap.add_argument("--min_group_size", type=int, default=4)
    ap.add_argument("--advantage_clip", type=float, default=10.0)
    ap.add_argument("--reward_field", default="total_episode_reward",
                    help="Episode-level reward key (e.g. total_episode_reward, total_episode_reward_v4)")
    ap.add_argument("--per_turn_reward_field", default=None,
                    help="If set, use per-turn scores from this list field (e.g. "
                         "judge_turn_scores) for turn-level advantages instead of "
                         "broadcasting the episode reward")
    args = ap.parse_args()

    import glob
    if args.sweep_json:
        sweep_path = args.sweep_json
    else:
        finals = sorted(glob.glob(os.path.join(args.sweep_dir, "random_sweep_*.json")))
        sweep_path = finals[-1] if finals else os.path.join(args.sweep_dir, "partial_results.json")
    scenarios_path = os.path.join(args.sweep_dir, "scenarios.json")

    pairs = extract_grpo_pairs(
        sweep_path=sweep_path,
        scenarios_path=scenarios_path,
        scenario_type=args.scenario_type,
        min_group_size=args.min_group_size,
        advantage_clip=args.advantage_clip,
        reward_field=args.reward_field,
        per_turn_reward_field=args.per_turn_reward_field,
    )

    out = args.out
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.sweep_dir, f"grpo_train_{ts}.jsonl")
    write_jsonl(pairs, out)


if __name__ == "__main__":
    main()
