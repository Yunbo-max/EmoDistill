"""
DPO preference-pair extractor from the judge-scored random_emotion_sweep.

Pairing strategy
----------------
Each scenario was rolled out ~50 times with different random emotion sequences.
The judge gave each episode a `judge_episode_score` (and also per-turn scores
`judge_turn_scores`).

For each scenario, we:
  1. Sort the iterations by episode_judge_score
  2. Take top-K iterations as "chosen" and bottom-K as "rejected"
  3. For every (chosen_ep, rejected_ep) pair, align them by turn-index
     and emit per-turn preference triplets:
         (prompt, chosen_response, rejected_response)
  4. The prompt used is the one from the *chosen* episode's turn (the chosen
     and rejected dialogs diverge after turn 0 because of different emotion
     sequences, so the prompts at turn>0 are not identical — we use the
     chosen's prompt for both sides, which is the standard DPO approximation
     when within-pair prompts differ slightly).

Output JSONL: one dict per pair, with keys
  prompt, chosen, rejected, scenario, turn_idx, chosen_score, rejected_score
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from llm.prompt_templates import PromptTemplates
from EmoDistill.emotions import prompt_for


def _build_creditor_prompt(scenario, dialog_history, emotion, scenario_type="debt"):
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
        scenario_type=scenario_type, config=config,
        emotion_config=emotion_config, timeline_text=timeline_text, debt_info=debt_info,
    )


def _creditor_turns_with_prompts(ep, scenario, scenario_type):
    """Return list of (prompt, response, emotion) for each creditor turn in episode."""
    out = []
    dialog = ep.get("dialog", []) or []
    emotion_seq = ep.get("emotion_sequence", []) or []
    creditor_idxs = [i for i, d in enumerate(dialog) if d.get("speaker") == "seller"]
    for k, ci in enumerate(creditor_idxs):
        if k >= len(emotion_seq):
            break
        emotion = emotion_seq[k]
        dialog_before = [(d["speaker"], d["message"]) for d in dialog[:ci]]
        msg = (dialog[ci].get("message") or "").strip()
        if not msg:
            continue
        prompt = _build_creditor_prompt(scenario, dialog_before, emotion, scenario_type)
        out.append((prompt, msg, emotion))
    return out


def extract_dpo_pairs(
    sweep_path: str,
    scenarios_path: str,
    scenario_type: str = "debt",
    top_k: int = 15,
    bot_k: int = 15,
    score_field: str = "judge_episode_score",
    min_score_gap: float = 0.5,
) -> List[Dict[str, Any]]:
    print(f"📂 Sweep: {sweep_path}")
    with open(sweep_path) as f: sweep = json.load(f)
    with open(scenarios_path) as f: scenarios_list = json.load(f)
    scenarios_by_id = {s.get("id"): s for s in scenarios_list}
    episodes = sweep.get("episode_results", [])
    print(f"   {len(episodes)} episodes loaded")

    # Group episodes by scenario
    by_scen: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        sid = ep.get("scenario")
        if sid is None or score_field not in ep:
            continue
        by_scen[sid].append(ep)
    print(f"   {len(by_scen)} scenarios with judge scores")

    pairs: List[Dict[str, Any]] = []
    n_skipped_gap = 0
    for sid, eps in by_scen.items():
        if len(eps) < top_k + bot_k:
            continue
        scenario = scenarios_by_id.get(sid)
        if scenario is None:
            continue
        sorted_eps = sorted(eps, key=lambda e: -e[score_field])
        top = sorted_eps[:top_k]
        bot = sorted_eps[-bot_k:]

        for ch in top:
            for rj in bot:
                if ch is rj:
                    continue
                if (ch[score_field] - rj[score_field]) < min_score_gap:
                    n_skipped_gap += 1
                    continue
                chs = _creditor_turns_with_prompts(ch, scenario, scenario_type)
                rjs = _creditor_turns_with_prompts(rj, scenario, scenario_type)
                n_turns = min(len(chs), len(rjs))
                for k in range(n_turns):
                    prompt, ch_msg, ch_emo = chs[k]
                    _, rj_msg, _ = rjs[k]
                    if not ch_msg or not rj_msg:
                        continue
                    pairs.append({
                        "prompt": prompt,
                        "chosen": ch_msg,
                        "rejected": rj_msg,
                        "scenario": sid,
                        "turn_idx": k,
                        "emotion": ch_emo,
                        "chosen_score": float(ch[score_field]),
                        "rejected_score": float(rj[score_field]),
                        "score_gap": float(ch[score_field] - rj[score_field]),
                    })
    if n_skipped_gap:
        print(f"   skipped {n_skipped_gap} pairs with score gap < {min_score_gap}")
    print(f"✅ Built {len(pairs)} preference pairs")
    if pairs:
        gaps = np.array([p["score_gap"] for p in pairs])
        print(f"   score gap distribution: mean={gaps.mean():.2f}  "
              f"min={gaps.min():.2f}  max={gaps.max():.2f}")
    return pairs


def write_jsonl(pairs, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"💾 Wrote {len(pairs)} pairs → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--sweep_json", default=None,
                    help="Override sweep JSON (e.g. judged_qwen_plus_v2.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario_type", default="debt")
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--bot_k", type=int, default=15)
    ap.add_argument("--score_field", default="judge_episode_score")
    ap.add_argument("--min_score_gap", type=float, default=0.5)
    args = ap.parse_args()

    import glob
    sweep_path = args.sweep_json or sorted(
        glob.glob(os.path.join(args.sweep_dir, "random_sweep_*.json"))
    )[-1]
    scenarios_path = os.path.join(args.sweep_dir, "scenarios.json")

    pairs = extract_dpo_pairs(
        sweep_path=sweep_path, scenarios_path=scenarios_path,
        scenario_type=args.scenario_type,
        top_k=args.top_k, bot_k=args.bot_k,
        score_field=args.score_field, min_score_gap=args.min_score_gap,
    )

    out = args.out
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(args.sweep_dir, f"dpo_pairs_{ts}.jsonl")
    write_jsonl(pairs, out)


if __name__ == "__main__":
    main()
