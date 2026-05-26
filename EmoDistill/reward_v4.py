"""
Reward design v4 — Time-Weighted Symmetric Net Concession.

Builds on v3 (symmetric net concession + ±2 outcome anchor) by adding a
LINEAR time penalty to each per-step reward:

    r_t = (Δ_debtor - Δ_creditor) × (1 - t / max_t)
    R_final = +2 if success else -2     (unchanged from v3)

Effect:
  - Turn 1:  step reward × ~0.97  (almost full weight)
  - Turn 15: step reward × ~0.50  (half weight)
  - Turn 28: step reward × ~0.07  (almost zero)
  - Turn max: step reward × 0     (no contribution)

Why this is more accurate for negotiation:
  - Real debt collection prefers EARLY closing (less interest accrual,
    less default risk, faster pipeline throughput).
  - v3 treats all step rewards equally, so a 0.1 concession at turn 1
    and at turn 28 count the same. v4 correctly says the early one is
    much more valuable.

Why we DON'T weight R_final:
  - A closed deal at turn 30 is still a closed deal — should still get the
    +2 anchor. Weighting R_final by (1 - t/max) would make late deals
    indistinguishable from breakdown.
  - The "close earlier is better" signal is already in the step reward
    weighting: early steps accumulate more, so closing earlier means MORE
    total weighted step reward AND the same +2 anchor.

Comparison vs v3:
  - v3 total = net_advantage + ±2
  - v4 total = weighted_net_advantage + ±2
  - For same net advantage, v4 prefers trajectories that achieve concessions
    EARLY rather than late.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


W_DEBTOR_CONCESSION = 1.0
W_CREDITOR_CAPITULATION = 1.0
W_DEAL_REACHED = 2.0
W_FAILURE = -2.0


def _time_weight(turn: int, max_turn: int) -> float:
    """Linear decay weight: (1 - t/max_t), clipped to [0, 1]."""
    if max_turn <= 0:
        return 1.0
    w = 1.0 - float(turn) / float(max_turn)
    return max(0.0, min(1.0, w))


def compute_step_reward_v4(
    prev_debtor_offer: Optional[int],
    new_debtor_offer: Optional[int],
    prev_creditor_offer: Optional[int],
    new_creditor_offer: Optional[int],
    initial_gap: int,
    turn: int,
    max_turn: int,
) -> Tuple[float, Dict[str, float]]:
    """Per-step reward = symmetric net concession × linear time weight.

    Args:
      turn: 1-indexed turn number for this creditor decision
      max_turn: maximum episode length (typically 30)
    """
    gap_norm = max(1, initial_gap)

    debtor_concession = 0.0
    if prev_debtor_offer is not None and new_debtor_offer is not None:
        debtor_concession = float(prev_debtor_offer - new_debtor_offer) / gap_norm
        debtor_concession = float(np.clip(debtor_concession, -2.0, 2.0))

    creditor_capitulation = 0.0
    if prev_creditor_offer is not None and new_creditor_offer is not None:
        creditor_capitulation = float(new_creditor_offer - prev_creditor_offer) / gap_norm
        creditor_capitulation = float(np.clip(creditor_capitulation, -2.0, 2.0))

    net = W_DEBTOR_CONCESSION * debtor_concession - W_CREDITOR_CAPITULATION * creditor_capitulation
    w = _time_weight(turn, max_turn)
    reward = net * w

    return reward, {
        "delta_debtor_concession": debtor_concession,
        "delta_creditor_capitulation": creditor_capitulation,
        "net_raw": net,
        "time_weight": w,
        "step_reward_v4": float(reward),
    }


def compute_final_reward_v4(success: bool) -> Tuple[float, Dict[str, float]]:
    """Terminal anchor: ±2, NOT time-weighted (late deals still close)."""
    r = W_DEAL_REACHED if success else W_FAILURE
    return float(r), {"final_v4": float(r), "success": bool(success)}


def recompute_episode_reward_v4(
    dialog: List[Dict],
    creditor_target: int,
    debtor_initial: int,
    success: bool,
    max_turn: int = 30,
) -> Tuple[Dict, List[float]]:
    """Walk a saved dialog and recompute reward under v4 (time-weighted)."""
    initial_gap = max(1, abs(debtor_initial - creditor_target))

    prev_debtor = debtor_initial
    prev_creditor = creditor_target
    step_rewards: List[float] = []

    last_seen_creditor = creditor_target
    last_seen_debtor = debtor_initial
    creditor_turn_count = 0

    for i, entry in enumerate(dialog):
        if entry.get("speaker") != "seller":
            continue
        creditor_turn_count += 1

        new_creditor_offer = entry.get("requested_days")
        new_creditor_offer = int(new_creditor_offer) if new_creditor_offer is not None else last_seen_creditor

        new_debtor_offer = last_seen_debtor
        if i + 1 < len(dialog) and dialog[i + 1].get("speaker") == "buyer":
            rd = dialog[i + 1].get("requested_days")
            if rd is not None:
                new_debtor_offer = int(rd)

        r, _ = compute_step_reward_v4(
            prev_debtor_offer=prev_debtor,
            new_debtor_offer=new_debtor_offer,
            prev_creditor_offer=prev_creditor,
            new_creditor_offer=new_creditor_offer,
            initial_gap=initial_gap,
            turn=creditor_turn_count,
            max_turn=max_turn,
        )
        step_rewards.append(r)
        prev_creditor = new_creditor_offer
        prev_debtor = new_debtor_offer
        last_seen_creditor = new_creditor_offer
        last_seen_debtor = new_debtor_offer

    final_r, _ = compute_final_reward_v4(success)
    if step_rewards:
        step_rewards[-1] += final_r

    weighted_net_advantage = float(sum(r for r in step_rewards) - final_r) if step_rewards else 0.0
    total_episode_reward = float(sum(step_rewards)) if step_rewards else float(final_r)

    return {
        "total_step_reward_v4": weighted_net_advantage,
        "final_outcome_reward_v4": float(final_r),
        "total_episode_reward_v4": total_episode_reward,
        "weighted_net_creditor_advantage": float(weighted_net_advantage),
        "final_creditor_offer": int(last_seen_creditor),
        "final_debtor_offer": int(last_seen_debtor),
        "creditor_target": int(creditor_target),
        "debtor_initial": int(debtor_initial),
        "n_creditor_turns": len(step_rewards),
        "max_turn": max_turn,
    }, step_rewards
