"""
Offline RL dataset for IQL training.

Workflow:
1. fixed_emotion_baseline.py runs negotiations and calls `append_episode()` for
   each completed dialog → in-memory list of episodes.
2. After collection, `save_npz()` flattens episodes into (s, a, r, s', done)
   tuples and writes a compact NPZ + a sidecar JSON with metadata.
3. IQL training loads the NPZ via `OfflineDataset.load()` and samples minibatches.

States are computed by walking the dialog with the same `build_state()` used by
DQN-new — so dataset format and IQL training-time state are identical.

If `observer_features_per_turn` is provided, Level-2 observer features are
included; otherwise Level-2 defaults (deal=0.5, breakdown=0.3, opp=neutral)
are filled in by build_state itself.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from EmoDistill.emotions import emotion_to_idx, get_active_taxonomy_name, get_emotions, n_emotions, parse_emotion_str
# v4 reward: v3 with a per-step linear time weight (1 - t/max_t).
# Early concessions weighted more, late ones near zero. Final ±2 anchor
# unchanged. See EmoDistill/reward_v4.py for details.
from EmoDistill.reward_v4 import (
    compute_step_reward_v4 as compute_step_reward,
    compute_final_reward_v4 as compute_final_reward,
)
from EmoDistill.state_builder import build_state, state_dim


@dataclass
class EpisodeTransitions:
    """Per-episode tuples ready for IQL training."""

    emotion_label: str
    scenario_id: Any
    iteration: int
    success: bool

    states: np.ndarray            # (T, state_dim) float32
    actions: np.ndarray           # (T,) int64
    rewards: np.ndarray           # (T,) float32 — dense step reward + final on last
    next_states: np.ndarray       # (T, state_dim) float32
    dones: np.ndarray             # (T,) bool

    debtor_offer_trajectory: List[int] = field(default_factory=list)
    final_days: Optional[int] = None
    creditor_target: int = 0
    debtor_initial: int = 0


def _fallback_extractor(message: str) -> Optional[int]:
    """Rule-based extractor used during offline state reconstruction.

    We deliberately avoid invoking an LLM here — the dialog dicts saved during
    sweep already carry `requested_days` field, so the lookup is just a regex
    fallback if a turn lacked the field.
    """
    if not message:
        return None
    import re
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*days?", 1),
        (r"(\d+(?:\.\d+)?)\s*weeks?", 7),
        (r"(\d+(?:\.\d+)?)\s*months?", 30),
        (r"(\d+(?:\.\d+)?)\s*hours?", 1.0 / 24),
        (r"(\d+(?:\.\d+)?)\s*minutes?", 1.0 / 1440),
    ]
    for pat, mul in patterns:
        m = re.findall(pat, message.lower())
        if m:
            return int(float(m[-1]) * mul) or 1
    return None


def build_transitions_from_episode(
    emotion_label: str,
    scenario: Dict[str, Any],
    neg_result: Dict[str, Any],
    iteration: int = 0,
    observer_features_per_turn: Optional[List[Dict[str, float]]] = None,
) -> Optional[EpisodeTransitions]:
    """Convert a completed negotiation into per-step (s, a, r, s', done) tuples.

    Returns None if the dialog has too few creditor turns to form even one
    (state, next_state) pair.
    """
    dialog = neg_result.get("dialog", [])
    if not dialog:
        return None

    creditor_target = int(scenario.get("seller", {}).get("target_price", 30))
    debtor_initial = int(scenario.get("buyer", {}).get("target_price", creditor_target * 3))
    initial_gap = max(1, abs(debtor_initial - creditor_target))
    max_turn = neg_result.get("max_dialog_len") or len(dialog)

    fallback_action_idx = parse_emotion_str(emotion_label)

    # Per-turn emotion sequence (if recorded). For random-emotion sweep this
    # gives the actual emotion picked each turn; for fixed-emotion sweep it's
    # the same emotion repeated. Falls back to `emotion_label` if missing.
    emotion_seq = neg_result.get("emotion_sequence", []) or []

    # Indices of seller (creditor) turns
    creditor_idxs = [i for i, e in enumerate(dialog) if e.get("speaker") == "seller"]
    if len(creditor_idxs) < 1:
        return None

    # Per-message extractor: prefer pre-parsed `requested_days` else regex.
    def extract_days(msg: str) -> Optional[int]:
        for entry in dialog:
            if entry.get("message") == msg:
                rd = entry.get("requested_days")
                if rd is not None:
                    return int(rd)
        return _fallback_extractor(msg)

    states_list: List[np.ndarray] = []
    actions_list: List[int] = []
    rewards_list: List[float] = []
    next_states_list: List[np.ndarray] = []
    dones_list: List[bool] = []

    prev_debtor_offer: Optional[int] = debtor_initial
    prev_creditor_offer: Optional[int] = creditor_target
    rolling_debtor_hist: List[int] = [debtor_initial]
    last_emotion = "neutral"

    for k, ci in enumerate(creditor_idxs):
        # Resolve THIS turn's actual emotion (per-turn for random sweep,
        # constant for fixed-emotion sweep). Falls back to `emotion_label`
        # if emotion_sequence wasn't recorded.
        this_emotion = emotion_seq[k] if k < len(emotion_seq) else emotion_label
        this_action_idx = parse_emotion_str(this_emotion)

        # State at decision time: dialog up to but NOT including this creditor turn
        dialog_before = [(e["speaker"], e["message"]) for e in dialog[:ci]]
        obs_now = observer_features_per_turn[k] if observer_features_per_turn and k < len(observer_features_per_turn) else None

        s_t = build_state(
            dialog=dialog_before,
            days_extractor=extract_days,
            last_emotion=last_emotion,         # ← uses prev-turn actual emotion
            turn=k + 1,
            max_turn=max_turn,
            creditor_target=creditor_target,
            debtor_target=debtor_initial,
            observer_features=obs_now,
        )

        # Action: this turn's actual chosen emotion (per-turn, not episode-level)
        a_t = this_action_idx

        # Capture creditor's new offer from this turn (for v3 symmetric reward)
        new_creditor_offer: Optional[int] = None
        cred_entry = dialog[ci]
        rd_c = cred_entry.get("requested_days")
        if rd_c is not None:
            new_creditor_offer = int(rd_c)

        # Reward: from debtor's response to this creditor turn
        debtor_idx = ci + 1
        new_debtor_offer: Optional[int] = None
        if debtor_idx < len(dialog) and dialog[debtor_idx].get("speaker") == "buyer":
            debtor_entry = dialog[debtor_idx]
            rd = debtor_entry.get("requested_days")
            if rd is not None:
                new_debtor_offer = int(rd)
                rolling_debtor_hist.append(new_debtor_offer)

        # v4 reward: r_t = (Δ_debtor - Δ_creditor) × (1 - turn/max_turn)
        r_t, _ = compute_step_reward(
            prev_debtor_offer=prev_debtor_offer,
            new_debtor_offer=new_debtor_offer,
            prev_creditor_offer=prev_creditor_offer,
            new_creditor_offer=new_creditor_offer,
            initial_gap=initial_gap,
            turn=k + 1,
            max_turn=max_turn,
        )

        # Next state: dialog up through (and including) the debtor's reply
        is_last = (k == len(creditor_idxs) - 1)
        dialog_after = [(e["speaker"], e["message"]) for e in dialog[: debtor_idx + 1]] if debtor_idx < len(dialog) else dialog_before
        s_next = build_state(
            dialog=dialog_after,
            days_extractor=extract_days,
            last_emotion=this_emotion,         # ← uses THIS turn's emotion (now becomes prev)
            turn=k + 2,
            max_turn=max_turn,
            creditor_target=creditor_target,
            debtor_target=debtor_initial,
            observer_features=obs_now,
        )

        states_list.append(s_t)
        actions_list.append(a_t)
        rewards_list.append(r_t)
        next_states_list.append(s_next)
        dones_list.append(is_last)

        prev_debtor_offer = new_debtor_offer if new_debtor_offer is not None else prev_debtor_offer
        prev_creditor_offer = new_creditor_offer if new_creditor_offer is not None else prev_creditor_offer
        last_emotion = this_emotion  # ← carry forward THIS turn's emotion as next turn's "last"

    if not states_list:
        return None

    # Add final-outcome bonus to the last transition's reward (v3: ±2 anchor)
    success = neg_result.get("final_state") == "accept"
    final_r, _ = compute_final_reward(success)
    rewards_list[-1] += float(final_r)
    dones_list[-1] = True

    debtor_offer_trajectory = [d for d in rolling_debtor_hist[1:]]  # drop seed

    return EpisodeTransitions(
        emotion_label=emotion_label,
        scenario_id=scenario.get("id"),
        iteration=iteration,
        success=success,
        states=np.asarray(states_list, dtype=np.float32),
        actions=np.asarray(actions_list, dtype=np.int64),
        rewards=np.asarray(rewards_list, dtype=np.float32),
        next_states=np.asarray(next_states_list, dtype=np.float32),
        dones=np.asarray(dones_list, dtype=bool),
        debtor_offer_trajectory=debtor_offer_trajectory,
        final_days=neg_result.get("final_days"),
        creditor_target=creditor_target,
        debtor_initial=debtor_initial,
    )


class OfflineDataset:
    """In-memory accumulator of episodes that can be written to / loaded from NPZ."""

    def __init__(self, taxonomy: Optional[str] = None):
        self.taxonomy = taxonomy or get_active_taxonomy_name()
        self.n_emotions = n_emotions()
        self.state_dim = state_dim()
        self.episodes: List[EpisodeTransitions] = []

    def append_episode(self, ep: Optional[EpisodeTransitions]) -> None:
        if ep is None:
            return
        if ep.states.shape[1] != self.state_dim:
            raise ValueError(
                f"State dim mismatch: dataset expects {self.state_dim}, got {ep.states.shape[1]}"
            )
        self.episodes.append(ep)

    def n_transitions(self) -> int:
        return sum(int(e.states.shape[0]) for e in self.episodes)

    def n_episodes(self) -> int:
        return len(self.episodes)

    def save_npz(self, path: str, scenarios_used: Optional[List[Any]] = None) -> str:
        """Flatten episodes into NPZ + write a JSON sidecar with metadata."""
        if not self.episodes:
            raise ValueError("No episodes to save")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []
        all_episode_ids, all_step_ids = [], []
        all_emotion_ids = []  # behavior-policy emotion (the fixed one used to generate this transition)

        for eid, ep in enumerate(self.episodes):
            n = ep.states.shape[0]
            all_states.append(ep.states)
            all_actions.append(ep.actions)
            all_rewards.append(ep.rewards)
            all_next_states.append(ep.next_states)
            all_dones.append(ep.dones)
            all_episode_ids.append(np.full(n, eid, dtype=np.int64))
            all_step_ids.append(np.arange(n, dtype=np.int64))
            all_emotion_ids.append(np.full(n, ep.actions[0], dtype=np.int64))  # fixed-emotion = first action

        np.savez_compressed(
            path,
            states=np.concatenate(all_states, axis=0),
            actions=np.concatenate(all_actions, axis=0),
            rewards=np.concatenate(all_rewards, axis=0),
            next_states=np.concatenate(all_next_states, axis=0),
            dones=np.concatenate(all_dones, axis=0),
            episode_ids=np.concatenate(all_episode_ids, axis=0),
            step_ids=np.concatenate(all_step_ids, axis=0),
            behavior_emotion_ids=np.concatenate(all_emotion_ids, axis=0),
        )

        meta = {
            "taxonomy": self.taxonomy,
            "n_emotions": self.n_emotions,
            "state_dim": self.state_dim,
            "n_episodes": self.n_episodes(),
            "n_transitions": self.n_transitions(),
            "created_at": datetime.now().isoformat(),
            "scenarios_used": scenarios_used or [],
            "emotion_labels": get_emotions(),
            "per_emotion_episode_count": self._per_emotion_episode_count(),
            "success_rate_overall": self._overall_success_rate(),
        }
        meta_path = path.rsplit(".", 1)[0] + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"💾 Saved offline dataset → {path}  ({meta['n_transitions']} transitions, {meta['n_episodes']} episodes)")
        print(f"📝 Metadata sidecar      → {meta_path}")
        return path

    def _per_emotion_episode_count(self) -> Dict[str, int]:
        emos = get_emotions()
        counts: Dict[str, int] = {e: 0 for e in emos}
        for ep in self.episodes:
            counts[ep.emotion_label] = counts.get(ep.emotion_label, 0) + 1
        return counts

    def _overall_success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)

    # -------- Loading --------

    @staticmethod
    def load(path: str) -> "LoadedOfflineDataset":
        meta_path = path.rsplit(".", 1)[0] + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        npz = np.load(path)
        return LoadedOfflineDataset(
            states=npz["states"],
            actions=npz["actions"],
            rewards=npz["rewards"],
            next_states=npz["next_states"],
            dones=npz["dones"],
            episode_ids=npz["episode_ids"],
            step_ids=npz["step_ids"],
            behavior_emotion_ids=npz["behavior_emotion_ids"],
            meta=meta,
        )


@dataclass
class LoadedOfflineDataset:
    """In-memory dataset, ready for minibatch sampling during IQL training."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray
    behavior_emotion_ids: np.ndarray
    meta: Dict[str, Any]

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.meta.get("n_emotions", self.actions.max() + 1))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }

    def normalize_rewards(self, mode: str = "scale") -> None:
        """Normalize rewards in-place (helps IQL stability)."""
        r = self.rewards
        if mode == "scale":
            scale = float(np.max(np.abs(r))) or 1.0
            self.rewards = r / scale
        elif mode == "zscore":
            mu = float(r.mean())
            sd = float(r.std()) or 1.0
            self.rewards = (r - mu) / sd
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown reward normalize mode: {mode}")

    def summary(self) -> str:
        return (
            f"OfflineDataset: {len(self)} transitions over {self.meta.get('n_episodes')} episodes\n"
            f"  state_dim={self.state_dim}, n_actions={self.n_actions}, taxonomy={self.meta.get('taxonomy')}\n"
            f"  reward stats: min={self.rewards.min():.3f}, max={self.rewards.max():.3f}, "
            f"mean={self.rewards.mean():.3f}, std={self.rewards.std():.3f}\n"
            f"  done count: {int(self.dones.sum())} / {len(self)}\n"
            f"  per-emotion episodes: {self.meta.get('per_emotion_episode_count')}"
        )
