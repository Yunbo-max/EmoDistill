"""
Dense per-step reward grounded in objective offer behavior.

r_t = w_concession * Δopponent_offer_normalized
    + w_speed       * concession_acceleration
    - w_stalemate   * stalemate_indicator
    - w_breakdown   * breakdown_risk
    + (only at episode end) final outcome term

All terms are computed from offer trajectory and a small set of regex/heuristic
signals — no LLM judge. The only LLM-derived term is breakdown_risk from the
observer, kept low-weight so it can't dominate the reward.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# Reward weights (Option B: pure Δ_concession + ±1 final anchor).
# Per-step reward = pure debtor concession signal.
# Final = ±1 binary anchor distinguishing accept vs breakdown.
#
# Time pressure is INTENTIONALLY OMITTED in this iteration: the first sweep is
# for signal discovery (does emotion conditioning change debtor behavior?), not
# policy efficiency. A time penalty would confound the emotion effect with
# episode-length effects. IQL's gamma discount (0.99) already provides implicit
# time preference if needed. If a later experiment shows the policy dragging
# episodes out, re-enable W_TIME then.
W_CONCESSION = 1.0
W_SPEED = 0.0       # disabled
W_STALEMATE = 0.0   # disabled
W_BREAKDOWN = 0.0   # disabled
W_TIME = 0.0        # disabled — signal discovery first, efficiency later

# Final outcome: ±1 binary anchor
W_FINAL_UTILITY = 0.0  # disabled (was 2.0)
W_DEAL_REACHED = 1.0   # success bonus
W_FAILURE = -1.0       # failure penalty

STALEMATE_EPS = 1.0
BREAKDOWN_KEYWORDS = (
    "we're done",
    "i'm done",
    "no deal",
    "cannot continue",
    "walk away",
    "ending this",
    "this won't work",
    "forget it",
)


def compute_step_reward(
    prev_debtor_offer: Optional[int],
    new_debtor_offer: Optional[int],
    debtor_offer_history: List[int],
    initial_gap: int,
    debtor_message: str,
    observer_breakdown_risk: float = 0.0,
    turn: Optional[int] = None,
    max_turn: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Per-step reward after observing the debtor's reply for this turn.

    A concession is the debtor REDUCING their requested days toward the creditor.
    If `turn` and `max_turn` are provided, a small time-pressure penalty
    (W_TIME · turn/max_turn) is subtracted from the step reward — this gently
    pushes the policy toward earlier deals without dominating Δ_concession.
    """
    components = {}
    gap_norm = max(1, initial_gap)

    # Δopponent_offer: positive = concession toward us
    delta_concession = 0.0
    if prev_debtor_offer is not None and new_debtor_offer is not None:
        delta_concession = float(prev_debtor_offer - new_debtor_offer) / gap_norm
        delta_concession = float(np.clip(delta_concession, -2.0, 2.0))
    components["delta_concession"] = delta_concession

    # Concession acceleration: this-step delta minus previous-step delta
    accel = 0.0
    if len(debtor_offer_history) >= 3:
        d_now = debtor_offer_history[-2] - debtor_offer_history[-1]
        d_prev = debtor_offer_history[-3] - debtor_offer_history[-2]
        accel = float((d_now - d_prev) / gap_norm)
        accel = float(np.clip(accel, -2.0, 2.0))
    components["accel"] = accel

    # Stalemate: 1 if last two debtor offers barely moved
    stalemate = 0.0
    if prev_debtor_offer is not None and new_debtor_offer is not None:
        if abs(new_debtor_offer - prev_debtor_offer) < STALEMATE_EPS:
            stalemate = 1.0
    components["stalemate"] = stalemate

    # Breakdown signal: keyword + observer
    kw_hit = 0.0
    if debtor_message:
        msg_lower = debtor_message.lower()
        if any(k in msg_lower for k in BREAKDOWN_KEYWORDS):
            kw_hit = 1.0
    breakdown_signal = max(kw_hit, float(observer_breakdown_risk))
    components["breakdown_signal"] = breakdown_signal

    # Time pressure: gentle penalty proportional to turn/max_turn. Larger when
    # we're deep into the dialog; near-zero at the start.
    time_penalty = 0.0
    if turn is not None and max_turn is not None and max_turn > 0:
        time_penalty = W_TIME * (float(turn) / float(max_turn))
    components["time_penalty"] = time_penalty

    reward = (
        W_CONCESSION * delta_concession
        + W_SPEED * accel
        - W_STALEMATE * stalemate
        - W_BREAKDOWN * breakdown_signal
        - time_penalty
    )
    components["step_reward"] = float(reward)
    return float(reward), components


def compute_final_reward(
    success: bool,
    final_days: Optional[int],
    creditor_target: int,
    debtor_initial: int,
) -> Tuple[float, Dict[str, float]]:
    """Terminal-step bonus/penalty layered on top of the dense signal."""
    components = {}
    if not success or final_days is None:
        components["final"] = float(W_FAILURE)
        return float(W_FAILURE), components

    span = max(1, debtor_initial - creditor_target)
    # utility = how close to creditor's target (lower days = better for creditor)
    utility = (debtor_initial - final_days) / span
    utility = float(np.clip(utility, 0.0, 1.0))
    final = W_FINAL_UTILITY * utility + W_DEAL_REACHED
    components["utility"] = utility
    components["final"] = float(final)
    return float(final), components
