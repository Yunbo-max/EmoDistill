"""
LLM-judge per-turn reward scorer for the random_emotion_sweep.

For every creditor turn in every episode, query a strong LLM (qwen-plus by
default) to rate the utterance on a 1–10 scale of negotiation quality:

  10  Strong, strategic, firm but professional, advances toward deal closure
   5  Average — neither helping nor hurting the creditor
   1  Weak, capitulating, off-topic, or counter-productive

The judge sees the FULL preceding dialog history, the creditor utterance to
score, and the scenario context. It responds with a single integer 1-10
(and a one-line rationale, which we log but don't use for training).

Output
------
For each episode in the sweep:
  episode["judge_turn_scores"]  : list[int] of length n_creditor_turns
  episode["judge_episode_score"]: mean of the per-turn scores (per-episode summary)

The output sweep JSON has the same shape as the input but with these two
fields added per episode. Downstream `grpo_data.py` can be pointed at the
new file via --reward_field=judge_episode_score.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from EmoDistill.dashscope_wrapper import DashScopeWrapper

JUDGE_SYSTEM = (
    "You are an expert negotiation analyst evaluating debt-collection dialogs. "
    "Rate each creditor utterance on a 1-10 integer scale, judging how well it "
    "BALANCES FIRMNESS (extracting value) with REALISM (closing the deal at all).\n\n"
    "REWARD THE FOLLOWING (higher scores):\n"
    "  + STRATEGIC anchoring near the creditor's target days WITH realistic chance of acceptance\n"
    "  + Specific, dated demands tied to legitimate LEVERAGE (escalation, escrow, secured-priority)\n"
    "  + Calibrated CONCESSIONS — moves enough to keep the deal alive, anchors enough to extract value\n"
    "  + Maintains firm position WHILE acknowledging the debtor's genuine constraints\n"
    "  + READS THE ROOM — recognizes when the debtor is at their limit vs stalling\n"
    "  + Pushes toward A SUCCESSFUL CLOSE at favourable (but realistically achievable) terms\n\n"
    "PENALIZE THE FOLLOWING (lower scores):\n"
    "  - UNREALISTIC anchors that risk negotiation BREAKDOWN (e.g., demanding 3 days on a 30-day debt)\n"
    "  - Aggressive ULTIMATUMS or breakdown-inducing threats when not strategically justified\n"
    "  - Excessive concessions when the debtor still has bargaining room\n"
    "  - Repeating same offer without advancing OR closing\n"
    "  - Long-winded, over-empathetic, capitulating language\n"
    "  - Posturing without leverage — empty threats that the debtor will recognize as bluff\n\n"
    "Anchor points:\n"
    "  10 - Excellent: firm-and-realistic anchor, clear leverage, pushes toward fast SUCCESSFUL close\n"
    "   8 - Strong: small calibrated concession, firm tone, deal-closing momentum\n"
    "   6 - Average: maintains position but not advancing closure\n"
    "   4 - Weak: large concession OR unrealistic anchor that risks breakdown\n"
    "   2 - Poor: capitulates OR pushes for impossible terms — deal will break\n"
    "   1 - Terrible: completely undermines the negotiation in either direction\n\n"
    "CRITICAL: A turn that causes the debtor to walk away SCORES LOWER than a turn that closes "
    "at a moderate concession. Closing the deal IS the primary objective.\n\n"
    "Use the scenario context (creditor's target days, current creditor offer, dialog history) "
    "to judge whether this utterance EFFECTIVELY ADVANCES toward a SUCCESSFUL favourable close — "
    "not just toward firmness for its own sake.\n\n"
    "RESPONSE FORMAT (strict): one line containing exactly:\n"
    "  SCORE: <int 1-10>\n"
    "Optionally a second line with a one-sentence rationale.\n"
)


def _build_judge_prompt(
    scenario: Dict[str, Any],
    dialog_so_far: List[Dict[str, Any]],
    creditor_utterance: str,
) -> str:
    seller = scenario.get("seller_config", scenario.get("seller", {}))
    debt_info = scenario.get("metadata", {})
    target_days = seller.get("target_price", "?")
    amount = debt_info.get("outstanding_balance_usd", debt_info.get("amount", "?"))
    overdue = debt_info.get("days_overdue", "?")

    if not dialog_so_far:
        history = "(This is the creditor's opening turn.)"
    else:
        lines = []
        for d in dialog_so_far:
            label = "Creditor" if d.get("speaker") == "seller" else "Debtor"
            lines.append(f"{label}: {(d.get('message') or '').strip()}")
        history = "\n".join(lines)

    return (
        f"DEBT NEGOTIATION CONTEXT\n"
        f"  Outstanding balance: ${amount}\n"
        f"  Days overdue: {overdue}\n"
        f"  Creditor's target settlement: {target_days} days\n\n"
        f"DIALOG HISTORY\n{history}\n\n"
        f"CREDITOR UTTERANCE TO SCORE\n{creditor_utterance.strip()}\n\n"
        f"Provide your 1-10 score on the next line in the form 'SCORE: N'."
    )


def _parse_score(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"SCORE:\s*(\d{1,2})", text, re.IGNORECASE)
    if not m:
        # fallback: any standalone integer 1..10
        m = re.search(r"\b([1-9]|10)\b", text)
    if not m:
        return None
    try:
        s = int(m.group(1))
    except ValueError:
        return None
    return max(1, min(10, s))


def score_one_turn(
    judge: DashScopeWrapper,
    scenario: Dict[str, Any],
    dialog_so_far: List[Dict[str, Any]],
    creditor_utterance: str,
) -> Tuple[Optional[int], str]:
    prompt = _build_judge_prompt(scenario, dialog_so_far, creditor_utterance)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        msg = judge.invoke(messages, temperature=0.0, max_tokens=64)
        text = getattr(msg, "content", str(msg))
        return _parse_score(text), text
    except Exception as e:
        return None, f"ERROR: {e}"


def score_episode(
    judge: DashScopeWrapper,
    episode: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """Score every creditor turn in an episode. Returns the augmented episode."""
    dialog = episode.get("dialog", []) or []
    if not dialog:
        episode["judge_turn_scores"] = []
        episode["judge_episode_score"] = 5.0
        return episode

    creditor_idxs = [i for i, d in enumerate(dialog) if d.get("speaker") == "seller"]
    scores: List[int] = []
    for ci in creditor_idxs:
        creditor_msg = (dialog[ci].get("message") or "").strip()
        if not creditor_msg:
            scores.append(5)
            continue
        s, _ = score_one_turn(
            judge=judge,
            scenario=scenario,
            dialog_so_far=dialog[:ci],
            creditor_utterance=creditor_msg,
        )
        scores.append(int(s) if s is not None else 5)

    episode["judge_turn_scores"] = scores
    episode["judge_episode_score"] = float(sum(scores)) / max(1, len(scores))
    return episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_json", required=True, help="random_sweep_*.json input")
    ap.add_argument("--scenarios_json", required=True)
    ap.add_argument("--out_path", default=None,
                    help="Output JSON; default: <input>_judged.json next to it")
    ap.add_argument("--judge_model", default="qwen-plus",
                    help="DashScope model (qwen-plus, qwen3-32b, qwen-max...)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=None,
                    help="Cap episodes to score (debug)")
    ap.add_argument("--checkpoint_every", type=int, default=200,
                    help="Save partial output every N episodes")
    args = ap.parse_args()

    with open(args.sweep_json) as f:
        sweep = json.load(f)
    with open(args.scenarios_json) as f:
        scenarios_list = json.load(f)
    scenarios_by_id = {s.get("id"): s for s in scenarios_list}

    episodes = sweep.get("episode_results") or sweep.get("episodes") or []
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    print(f"📂 {len(episodes)} episodes to judge ({args.judge_model})")

    out_path = args.out_path
    if out_path is None:
        base = args.sweep_json.rsplit(".", 1)[0]
        out_path = f"{base}_judged.json"

    judge = DashScopeWrapper(model=args.judge_model, max_tokens=64)

    t0 = time.time()
    done = 0

    def task(idx_ep):
        idx, ep = idx_ep
        sid = ep.get("scenario")
        scen = scenarios_by_id.get(sid, {})
        return idx, score_episode(judge, ep, scen)

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(task, (i, ep)): i for i, ep in enumerate(episodes)}
        for fut in as_completed(futures):
            idx, ep = fut.result()
            episodes[idx] = ep
            done += 1
            if done % 20 == 0:
                rate = done / (time.time() - t0)
                eta = (len(episodes) - done) / max(rate, 1e-9) / 60
                print(f"   [{done}/{len(episodes)}]  rate={rate:.1f}/s  ETA≈{eta:.1f} min  "
                      f"last_episode_score={ep.get('judge_episode_score'):.2f}")
            if done % args.checkpoint_every == 0:
                # Partial save
                sweep["episode_results"] = episodes
                with open(out_path + ".partial", "w") as f:
                    json.dump(sweep, f)
                print(f"   💾 partial → {out_path}.partial")

    sweep["episode_results"] = episodes
    sweep["judge_meta"] = {
        "judge_model": args.judge_model,
        "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_episodes_scored": done,
        "concurrency": args.concurrency,
    }
    with open(out_path, "w") as f:
        json.dump(sweep, f)
    elapsed_min = (time.time() - t0) / 60
    print(f"\n✅ Done. {done} episodes, {elapsed_min:.1f} min total.")
    print(f"💾 → {out_path}")

    # Distribution diagnostic
    import numpy as np
    epsc = [e.get("judge_episode_score") for e in episodes if e.get("judge_episode_score") is not None]
    if epsc:
        print(f"   Judge episode-score distribution:")
        print(f"     mean={np.mean(epsc):.3f}  std={np.std(epsc):.3f}  "
              f"min={min(epsc):.2f}  max={max(epsc):.2f}")


if __name__ == "__main__":
    main()
