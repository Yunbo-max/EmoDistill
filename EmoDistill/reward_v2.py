"""
Reward design v2 — Symmetric Net Concession (Option D).

Motivation
----------
Option B (the v1 reward in EmoDistill/reward.py) measures only the debtor's
concession. Empirically the trajectory analysis showed many "successful" deals
in v1 came from the CREDITOR caving to the debtor — the debtor never moved.
v1 rewards those deals with +1 anchor, which is misleading and would lead IQL
to learn capitulation policies.

v2 fixes this by penalizing creditor capitulation symmetrically:

    r_t = α · Δ_debtor_concession_t  −  β · Δ_creditor_capitulation_t

with α = β = 1.0. The episode-summed step reward telescopes to the net
gap-closure toward the creditor's side, which IS the creditor's true utility.

A small ±1 final anchor distinguishes deal vs breakdown.

This module is additive — it does NOT modify reward.py. Use it via:

    from EmoDistill.reward_v2 import recompute_episode_reward_v2
    new_total, components = recompute_episode_reward_v2(dialog, creditor_target, debtor_initial)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# Weights (symmetric, no tuning at this stage)
W_DEBTOR_CONCESSION = 1.0
W_CREDITOR_CAPITULATION = 1.0

W_DEAL_REACHED = 1.0
W_FAILURE = -1.0


def compute_step_reward_v2(
    prev_debtor_offer: Optional[int],
    new_debtor_offer: Optional[int],
    prev_creditor_offer: Optional[int],
    new_creditor_offer: Optional[int],
    initial_gap: int,
) -> Tuple[float, Dict[str, float]]:
    """Symmetric per-step reward: debtor concession minus creditor capitulation.

    All deltas normalized by `initial_gap` (= |debtor_init - creditor_init|).

    Sign convention:
      - debtor reducing their days  →  positive Δ_debtor_concession
      - creditor raising their days →  positive Δ_creditor_capitulation (BAD)
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

    reward = (
        W_DEBTOR_CONCESSION * debtor_concession
        - W_CREDITOR_CAPITULATION * creditor_capitulation
    )

    return reward, {
        "delta_debtor_concession": debtor_concession,
        "delta_creditor_capitulation": creditor_capitulation,
        "step_reward_v2": float(reward),
    }


def compute_final_reward_v2(success: bool) -> Tuple[float, Dict[str, float]]:
    """Binary terminal anchor: +1 deal, −1 breakdown."""
    r = W_DEAL_REACHED if success else W_FAILURE
    return float(r), {"final_v2": float(r), "success": bool(success)}


def recompute_episode_reward_v2(
    dialog: List[Dict],
    creditor_target: int,
    debtor_initial: int,
    success: bool,
) -> Tuple[Dict, List[float]]:
    """Walk a saved dialog and recompute reward under v2.

    `dialog` is the per-episode list of dicts saved by run_fixed_emotion_sweep.
    Each entry has keys: speaker ('seller'|'buyer'), message, requested_days.

    Returns:
      summary: dict of aggregate metrics
      step_rewards: list of per-creditor-turn rewards (last one includes final anchor)
    """
    initial_gap = max(1, abs(debtor_initial - creditor_target))

    # Walk dialog and track current offers on both sides.
    prev_debtor = debtor_initial      # seed
    prev_creditor = creditor_target   # seed
    step_rewards: List[float] = []
    step_components: List[Dict[str, float]] = []

    last_seen_creditor = creditor_target
    last_seen_debtor = debtor_initial

    # Decision points = creditor turns. After creditor turn, look at debtor reply
    # in the next dialog entry to determine the step reward for THAT action.
    for i, entry in enumerate(dialog):
        if entry.get("speaker") != "seller":
            continue
        new_creditor_offer = entry.get("requested_days")
        if new_creditor_offer is not None:
            new_creditor_offer = int(new_creditor_offer)
        else:
            new_creditor_offer = last_seen_creditor

        # Find the debtor response after this creditor turn (next entry)
        new_debtor_offer = last_seen_debtor  # default = no change observed
        if i + 1 < len(dialog) and dialog[i + 1].get("speaker") == "buyer":
            rd = dialog[i + 1].get("requested_days")
            if rd is not None:
                new_debtor_offer = int(rd)

        r, comp = compute_step_reward_v2(
            prev_debtor_offer=prev_debtor,
            new_debtor_offer=new_debtor_offer,
            prev_creditor_offer=prev_creditor,
            new_creditor_offer=new_creditor_offer,
            initial_gap=initial_gap,
        )
        step_rewards.append(r)
        step_components.append(comp)

        prev_creditor = new_creditor_offer
        prev_debtor = new_debtor_offer
        last_seen_creditor = new_creditor_offer
        last_seen_debtor = new_debtor_offer

    # Final anchor on last step
    final_r, final_comp = compute_final_reward_v2(success)
    if step_rewards:
        step_rewards[-1] += final_r

    total_step_reward = float(sum(r for r in step_rewards) - final_r) if step_rewards else 0.0
    total_episode_reward = float(sum(step_rewards)) if step_rewards else float(final_r)

    # Compute net advantage metric directly from initial vs final offers
    final_creditor = last_seen_creditor
    final_debtor = last_seen_debtor
    debtor_total_concession = (debtor_initial - final_debtor) / initial_gap
    creditor_total_capitulation = (final_creditor - creditor_target) / initial_gap
    net_advantage = debtor_total_concession - creditor_total_capitulation

    summary = {
        "total_step_reward_v2": total_step_reward,
        "final_outcome_reward_v2": float(final_r),
        "total_episode_reward_v2": total_episode_reward,
        "debtor_total_concession_norm": float(debtor_total_concession),
        "creditor_total_capitulation_norm": float(creditor_total_capitulation),
        "net_creditor_advantage": float(net_advantage),
        "final_creditor_offer": int(final_creditor),
        "final_debtor_offer": int(final_debtor),
        "creditor_target": int(creditor_target),
        "debtor_initial": int(debtor_initial),
        "n_creditor_turns": len(step_rewards),
    }
    return summary, step_rewards
