"""
Random-emotion sweep: every creditor turn picks a random emotion from a
filtered subset of effective emotions.

Purpose
-------
The fixed-emotion sweep gave us per-episode commitment data — useful for
signal detection but it teaches IQL nothing about WITHIN-EPISODE switching.
This sweep produces the missing data: mixed-emotion trajectories that show
IQL "what happens if you change emotion mid-negotiation".

Filtering
---------
By default uses the 16 emotions whose mean_v3 reward exceeded 1.0 in the
qwen3.5-plus sweep — the "effective" set. The user can override.

Output
------
- partial_results.json / final sweep JSON: same schema as fixed-emotion sweep
  (so existing merge / recompute / dataset builders all work unchanged)
- offline_trajectories.npz: per-step (s, a, r, s', done) tuples with v3 reward,
  ready for IQL training.

The key behavioral difference vs FixedEmotionModel: emotion_sequence within
one episode is a RANDOM mix from the filtered subset.
"""

import os
import json
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from baselines.base_model import BaseEmotionModel
from EmoDistill.emotions import emotion_to_idx, get_emotions, prompt_for
from EmoDistill.offline_dataset import OfflineDataset, build_transitions_from_episode
from EmoDistill.reward import compute_step_reward, compute_final_reward
from EmoDistill.fixed_emotion_baseline import _compute_reward_metrics, _signal_diagnostic


# 16 emotions with mean_v3 > 1.0 from qwen3.5-plus sweep
DEFAULT_FILTERED_EMOTIONS_16 = [
    "annoyance", "realization", "surprise", "disgust",
    "fear", "sadness", "anger", "approval",
    "nervousness", "curiosity", "embarrassment", "grief",
    "confusion", "disapproval", "neutral", "disappointment",
]


class RandomEmotionModel(BaseEmotionModel):
    """Picks a uniformly random emotion from a filtered subset each turn."""

    def __init__(self, emotion_subset: Optional[List[str]] = None, seed: Optional[int] = None):
        all_emotions = get_emotions()
        if emotion_subset is None:
            emotion_subset = DEFAULT_FILTERED_EMOTIONS_16
        # validate
        unknown = [e for e in emotion_subset if e not in all_emotions]
        if unknown:
            raise ValueError(f"Unknown emotions for active taxonomy: {unknown}")
        self.emotion_subset = list(emotion_subset)
        self.subset_indices = [emotion_to_idx()[e] for e in self.emotion_subset]
        self.rng = random.Random(seed)
        self.history: List[str] = []

    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        idx_in_subset = self.rng.randrange(len(self.emotion_subset))
        emo = self.emotion_subset[idx_in_subset]
        action_idx = self.subset_indices[idx_in_subset]
        self.history.append(emo)
        return {
            "emotion": emo,
            "emotion_text": prompt_for(emo),
            "confidence": 1.0 / len(self.emotion_subset),  # uniform random
            "exploration_rate": 1.0,                       # fully exploratory
            "action_idx": action_idx,
            "temperature": 0.7,
            "use_emotion": True,
            "strategy": f"random_from_{len(self.emotion_subset)}_emotions",
        }

    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        pass  # no learning

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_type": "random_emotion",
            "n_emotions_in_subset": len(self.emotion_subset),
            "emotions": self.emotion_subset,
            "history_last_5": self.history[-5:],
        }

    def reset(self) -> None:
        self.history = []


def _run_single_random_negotiation(
    scenario: Dict[str, Any],
    iteration: int,
    emotion_subset: List[str],
    model_creditor: str,
    model_debtor: str,
    debtor_emotion: str,
    max_dialog_len: int,
    seed_offset: int,
) -> Dict[str, Any]:
    """Run one negotiation with random-emotion-per-turn creditor."""
    from llm.negotiator import DebtNegotiator

    model = RandomEmotionModel(emotion_subset=emotion_subset, seed=seed_offset)
    negotiator = DebtNegotiator(
        config=scenario,
        emotion_model=model,
        model_creditor=model_creditor,
        model_debtor=model_debtor,
        debtor_emotion=debtor_emotion,
        debtor_model_type="vanilla",
    )
    try:
        result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
    except Exception as e:
        print(f"      ⚠️  [random|{scenario.get('id')}|it{iteration}] failed: {e}")
        result = {
            "final_state": "breakdown", "final_days": None, "negotiation_rounds": 0,
            "dialog": [], "emotion_sequence": [], "scenario_id": scenario.get("id"),
            "error": str(e),
        }

    creditor_target = int(scenario.get("seller", {}).get("target_price", 30))
    debtor_initial = int(scenario.get("buyer", {}).get("target_price", creditor_target * 3))
    result["max_dialog_len"] = max_dialog_len
    reward_metrics = _compute_reward_metrics(result, creditor_target, debtor_initial, max_turn=max_dialog_len)

    # Build (s, a, r, s', done) transitions using v3 reward
    transitions = build_transitions_from_episode(
        emotion_label="random_mixed",       # placeholder — actions vary per turn
        scenario=scenario,
        neg_result=result,
        iteration=iteration,
        observer_features_per_turn=None,
    )
    # Critical: replace actions with the REAL random emotion per turn
    if transitions is not None and transitions.actions.size:
        seq = result.get("emotion_sequence", [])
        from EmoDistill.emotions import parse_emotion_str
        per_turn_actions = np.array(
            [parse_emotion_str(e) for e in seq[: transitions.actions.size]],
            dtype=np.int64,
        )
        if per_turn_actions.size == transitions.actions.size:
            transitions.actions = per_turn_actions

    return {
        "scenario": scenario.get("id"),
        "iteration": iteration,
        "success": result.get("final_state") == "accept",
        "final_state": result.get("final_state"),
        "final_days": result.get("final_days"),
        "creditor_target_days": result.get("creditor_target_days"),
        "debtor_initial_days": debtor_initial,
        "rounds": result.get("negotiation_rounds", 0),
        "emotion_sequence": result.get("emotion_sequence", []),
        "dialog": result.get("dialog", []),
        "total_episode_reward": reward_metrics["total_episode_reward"],
        "total_step_reward": reward_metrics["total_step_reward"],
        "final_outcome_reward": reward_metrics["final_outcome_reward"],
        "step_rewards": reward_metrics["step_rewards"],
        "total_debtor_concession_norm": reward_metrics["total_debtor_concession_norm"],
        "first_step_concession_norm": reward_metrics["first_step_concession_norm"],
        "savings_ratio": reward_metrics["savings_ratio"],
        "debtor_offer_trajectory": reward_metrics["debtor_offer_trajectory"],
        "_iql_transitions": transitions,
    }


def run_random_emotion_sweep(
    scenarios: List[Dict[str, Any]],
    emotion_subset: Optional[List[str]] = None,
    iterations: int = 10,
    model_creditor: str = "qwen3.5-plus",
    model_debtor: str = "qwen3.5-plus",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results/random_emotion_sweep",
    concurrency: int = 6,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Run scenarios × iterations negotiations with random-emotion creditor.

    Produces IQL training data containing within-episode emotion switches.
    """
    os.makedirs(out_dir, exist_ok=True)
    subset = list(emotion_subset) if emotion_subset else DEFAULT_FILTERED_EMOTIONS_16
    dataset = OfflineDataset()

    sweep: Dict[str, Any] = {
        "experiment_type": "random_emotion_sweep",
        "emotion_subset": subset,
        "n_subset": len(subset),
        "scenarios": [s.get("id") for s in scenarios],
        "iterations_per_scenario": iterations,
        "config": {
            "model_creditor": model_creditor,
            "model_debtor": model_debtor,
            "debtor_emotion": debtor_emotion,
            "max_dialog_len": max_dialog_len,
            "concurrency": concurrency,
            "base_seed": base_seed,
            "setup": "creditor picks random emotion each turn from filtered subset; debtor vanilla",
        },
        "episode_results": [],
    }

    tasks = [(s, it) for s in scenarios for it in range(iterations)]
    n_total = len(tasks)
    print(f"\n🎲 Random-Emotion Sweep")
    print(f"   Emotion subset ({len(subset)}): {subset}")
    print(f"   {len(scenarios)} scenarios × {iterations} iter = {n_total} episodes")
    print(f"   Concurrency: {concurrency}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for i, (scenario, it) in enumerate(tasks):
            futures.append(ex.submit(
                _run_single_random_negotiation,
                scenario=scenario,
                iteration=it,
                emotion_subset=subset,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                max_dialog_len=max_dialog_len,
                seed_offset=base_seed + i,
            ))
        completed = 0
        for fut in as_completed(futures):
            r = fut.result()
            completed += 1
            dataset.append_episode(r.pop("_iql_transitions", None))
            sweep["episode_results"].append(r)

            print(f"   [{completed}/{n_total}] sc={r['scenario']} it={r['iteration']} "
                  f"success={r['success']} rounds={r['rounds']} "
                  f"R={r['total_episode_reward']:.3f} "
                  f"unique_emos={len(set(r['emotion_sequence']))}")

            # Periodic snapshot
            if completed % 20 == 0 or completed == n_total:
                snap_path = os.path.join(out_dir, "partial_results.json")
                with open(snap_path, "w") as f:
                    json.dump(sweep, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    # Aggregate
    eps = sweep["episode_results"]
    successes = [e for e in eps if e["success"]]
    success_rate = len(successes) / max(1, len(eps))
    all_rewards = [e["total_episode_reward"] for e in eps]

    sweep["aggregate"] = {
        "n_episodes": len(eps),
        "n_success": len(successes),
        "success_rate": success_rate,
        "mean_episode_reward": float(np.mean(all_rewards)),
        "std_episode_reward": float(np.std(all_rewards)),
        "mean_first_step_concession": float(np.mean([e["first_step_concession_norm"] for e in eps])),
        "mean_total_concession": float(np.mean([e["total_debtor_concession_norm"] for e in eps])),
        "mean_savings_ratio_on_success": (
            float(np.mean([e["savings_ratio"] for e in eps if e["savings_ratio"] is not None]))
            if successes else 0.0
        ),
        "mean_unique_emotions_per_episode": float(np.mean([len(set(e["emotion_sequence"])) for e in eps if e["emotion_sequence"]])),
        "mean_emotion_seq_len": float(np.mean([len(e["emotion_sequence"]) for e in eps if e["emotion_sequence"]])),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"random_sweep_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
    print(f"\n💾 Saved sweep JSON → {out_path}")

    if dataset.n_episodes() > 0:
        npz_path = os.path.join(out_dir, "offline_trajectories.npz")
        dataset.save_npz(npz_path, scenarios_used=[s.get("id") for s in scenarios])
        sweep["offline_dataset_path"] = npz_path
        with open(out_path, "w") as f:
            json.dump(sweep, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
        print(f"💾 Saved IQL dataset → {npz_path}  ({dataset.n_transitions()} transitions)")

    print("\n" + "=" * 70)
    print(f"🏁 RANDOM-EMOTION SWEEP COMPLETE")
    print("=" * 70)
    a = sweep["aggregate"]
    print(f"  Episodes: {a['n_episodes']} ({a['n_success']} success, {a['success_rate']:.1%})")
    print(f"  Mean reward (v3):                {a['mean_episode_reward']:.3f} ± {a['std_episode_reward']:.3f}")
    print(f"  Mean unique emotions/episode:    {a['mean_unique_emotions_per_episode']:.2f}")
    print(f"  Mean emotion seq length:         {a['mean_emotion_seq_len']:.1f}")
    print(f"  Mean total debtor concession:    {a['mean_total_concession']:.3f}")
    print(f"  Mean savings (on success):       {a['mean_savings_ratio_on_success']:.3f}")
    print("=" * 70)
    return sweep
