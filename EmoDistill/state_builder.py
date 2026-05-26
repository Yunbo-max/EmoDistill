"""
Build 10-dim state vector for IQL training (compact design v2).

The previous design used a 28-dim one-hot for `last_emotion`, blowing the
state up to 37 dims even though only ~6-7 dims carried real information.
The new design represents the previous action as a single normalized scalar,
collapsing the state to 10 dims regardless of taxonomy size.

State layout (10 dim):

    [0]  Δ_debtor_offer       — debtor's last-step concession toward creditor
                                = (debtor_prev - debtor_new) / initial_gap
                                Positive when debtor moved our way.
                                Range ≈ [-2, +2] (clipped).

    [1]  offer_gap_norm        — current spread between the two sides
                                = (debtor_cur - creditor_cur) / initial_gap
                                Range ≈ [-2, +2] (clipped). 0 means they meet.

    [2]  turn_progress         — t / max_turn ∈ [0, 1].

    [3]  opponent_offer_norm   — debtor's current price / initial debtor target
                                ∈ [0, 2]. Captures HOW MUCH debtor moved.

    [4]  our_offer_norm        — (creditor_cur - creditor_target) / initial_gap
                                ∈ [0, ~1]. Captures HOW MUCH creditor caved.

    [5]  concession_speed      — mean Δ over last 3 debtor offers / initial_gap.
                                Positive = debtor accelerating toward us.

    [6]  stalemate_counter     — consecutive trailing turns with |Δ_debtor| < 1.
                                Normalized to [0, 1] (cap at 10).

    [7]  deal_probability      — observer LLM output, default 0.5 if not used.

    [8]  breakdown_risk        — observer LLM output, default 0.3 if not used.

    [9]  last_emotion_idx_norm — index of last emotion / (n_emotions - 1).
                                Scalar in [0, 1]. Replaces the 28-dim one-hot.

Sign convention:
  - Δ_debtor positive  → debtor moved toward us (good)
  - our_offer_norm > 0 → creditor capitulated (bad; reward_v3 penalizes via
                          the symmetric step reward)

The scalar encoding for last_emotion implies an ordinal relationship that
doesn't truly exist. With ~12K transitions across 16 actions, a small MLP
should still learn to disregard misleading magnitude comparisons (each
emotion appears in many state contexts). If empirical learning is poor,
the next iteration should replace this with an nn.Embedding lookup; see
plan in EmoDistill/iql.py docstring.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from EmoDistill.emotions import (
    emotion_to_idx,
    get_emotions,
    n_emotions,
    parse_emotion_str,
)


STATE_DIM = 10            # Fixed regardless of taxonomy size
LEVEL1_DIM = 7            # idx 0-6: behavioral / offer-derived
LEVEL2_DIM = 2            # idx 7-8: observer signals (placeholders when off)
EMOTION_IDX_DIM = 1       # idx 9: last action scalar

STALEMATE_EPS = 1.0       # |Δoffer| < 1 day counts as no movement
STALEMATE_NORM = 10.0     # divide stalemate counter by this for normalization

# Default observer values (used when observer is disabled — keeps state stable)
DEFAULT_DEAL_PROBABILITY = 0.5
DEFAULT_BREAKDOWN_RISK = 0.3


def state_dim() -> int:
    """Active state dimension. Always 10 in this v2 design."""
    return STATE_DIM


# Legacy module-level constants kept for backward compatibility with imports.
# Some older modules import EMOTIONS / N_EMOTIONS / EMOTION_TO_IDX directly.
def _refresh_legacy_constants():
    global EMOTIONS, N_EMOTIONS, EMOTION_TO_IDX
    EMOTIONS = list(get_emotions())
    N_EMOTIONS = n_emotions()
    EMOTION_TO_IDX = emotion_to_idx()


_refresh_legacy_constants()


def extract_offer_trajectory(
    dialog: List[Tuple[str, str]],
    days_extractor,
) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """Walk dialog and return (creditor_offers, debtor_offers) aligned to each turn.

    Each list has one entry per dialog turn, holding the latest known offer
    from that side up to and including that turn (None until first offer).
    """
    creditor_offers: List[Optional[int]] = []
    debtor_offers: List[Optional[int]] = []
    last_creditor = None
    last_debtor = None
    for speaker, msg in dialog:
        days = days_extractor(msg)
        if speaker == "seller":
            if days is not None:
                last_creditor = days
        else:
            if days is not None:
                last_debtor = days
        creditor_offers.append(last_creditor)
        debtor_offers.append(last_debtor)
    return creditor_offers, debtor_offers


def build_state(
    dialog: List[Tuple[str, str]],
    days_extractor,
    last_emotion: str,
    turn: int,
    max_turn: int,
    creditor_target: int,
    debtor_target: int,
    observer_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Construct a 10-dim state vector.

    Parameters mirror the v1 builder, but the output is always 10 dim and
    `last_emotion` is encoded as a single normalized scalar rather than a
    one-hot block.
    """
    _refresh_legacy_constants()
    state = np.zeros(STATE_DIM, dtype=np.float32)

    creditor_offers, debtor_offers = extract_offer_trajectory(dialog, days_extractor)
    initial_gap = max(1, abs(debtor_target - creditor_target))
    debtor_initial = max(1, debtor_target)

    cur_creditor = (
        creditor_offers[-1] if creditor_offers and creditor_offers[-1] is not None else creditor_target
    )
    cur_debtor = (
        debtor_offers[-1] if debtor_offers and debtor_offers[-1] is not None else debtor_target
    )

    # [0] Δ_debtor_offer — last-step debtor concession toward creditor
    delta_opp = 0.0
    debtor_seq = [d for d in debtor_offers if d is not None]
    if len(debtor_seq) >= 2:
        delta_opp = float(debtor_seq[-2] - debtor_seq[-1])  # positive = concession
    state[0] = np.clip(delta_opp / initial_gap, -2.0, 2.0)

    # [1] offer_gap_norm
    state[1] = np.clip((cur_debtor - cur_creditor) / initial_gap, -2.0, 2.0)

    # [2] turn_progress
    state[2] = min(1.0, turn / max(1, max_turn))

    # [3] opponent_offer_norm (debtor's current price / their initial target)
    state[3] = np.clip(cur_debtor / debtor_initial, 0.0, 2.0)

    # [4] our_offer_norm — fraction creditor has moved AWAY from its target
    #     0 = held at target, ~1 = caved all the way to debtor's initial
    state[4] = np.clip((cur_creditor - creditor_target) / initial_gap, 0.0, 2.0)

    # [5] concession_speed — avg Δ over last 3 debtor offers
    speed = 0.0
    if len(debtor_seq) >= 2:
        deltas = []
        for i in range(max(0, len(debtor_seq) - 4), len(debtor_seq) - 1):
            deltas.append(debtor_seq[i] - debtor_seq[i + 1])  # positive = concession
        if deltas:
            speed = float(np.mean(deltas)) / initial_gap
    state[5] = np.clip(speed, -2.0, 2.0)

    # [6] stalemate_counter — trailing turns where debtor barely moved
    stalemate = 0
    if len(debtor_seq) >= 2:
        for i in range(len(debtor_seq) - 1, 0, -1):
            if abs(debtor_seq[i] - debtor_seq[i - 1]) < STALEMATE_EPS:
                stalemate += 1
            else:
                break
    state[6] = min(1.0, stalemate / STALEMATE_NORM)

    # [7] deal_probability — observer or default
    if observer_features is not None and "deal_probability" in observer_features:
        state[7] = float(observer_features["deal_probability"])
    else:
        state[7] = DEFAULT_DEAL_PROBABILITY

    # [8] breakdown_risk — observer or default
    if observer_features is not None and "breakdown_risk" in observer_features:
        state[8] = float(observer_features["breakdown_risk"])
    else:
        state[8] = DEFAULT_BREAKDOWN_RISK

    # [9] last_emotion as a single normalized scalar
    e2i = EMOTION_TO_IDX
    n_emo = N_EMOTIONS
    emo_idx = e2i.get(last_emotion)
    if emo_idx is None:
        emo_idx = parse_emotion_str(last_emotion)
    state[9] = float(emo_idx) / max(1.0, n_emo - 1)

    return state
