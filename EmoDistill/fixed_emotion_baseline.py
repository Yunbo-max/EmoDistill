"""
Fixed-emotion baseline: sanity check before RL.

For each emotion in the active taxonomy, run the creditor with that emotion
held constant for the entire negotiation (i.e. policy = "always X"). Compare
success rate and savings against the vanilla (no-emotion) baseline.

If results are flat across emotions → the LLM is not actually responsive to
the emotion prompt, and DQN-new has no signal to learn from. Run THIS first.

Output per experiment (per emotion):
  - success rate
  - avg negotiation rounds (success)
  - avg final days (success)
  - per-scenario detail
Plus an aggregate comparison table.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baselines.base_model import BaseEmotionModel
from EmoDistill.emotions import (
    emotion_to_idx,
    get_emotions,
    prompt_for,
)
from EmoDistill.reward import compute_step_reward, compute_final_reward
from EmoDistill.offline_dataset import OfflineDataset, build_transitions_from_episode


class FixedEmotionModel(BaseEmotionModel):
    """Always returns the same emotion. No learning."""

    def __init__(self, emotion: str):
        if emotion not in get_emotions():
            raise ValueError(
                f"Emotion {emotion!r} not in active taxonomy {get_emotions()}"
            )
        self.emotion = emotion
        self.action_idx = emotion_to_idx()[emotion]
        self.negotiation_count = 0
        self.success_history: List[bool] = []

    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.negotiation_count += 1
        return {
            "emotion": self.emotion,
            "emotion_text": prompt_for(self.emotion),
            "confidence": 1.0,
            "exploration_rate": 0.0,
            "action_idx": self.action_idx,
            "temperature": 0.7,
            "use_emotion": True,
            "strategy": f"fixed_emotion::{self.emotion}",
        }

    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        self.success_history.append(negotiation_result.get("final_state") == "accept")

    def get_stats(self) -> Dict[str, Any]:
        if not self.success_history:
            return {"emotion": self.emotion, "n": 0, "success_rate": 0.0}
        return {
            "emotion": self.emotion,
            "n": len(self.success_history),
            "success_rate": float(np.mean(self.success_history)),
        }

    def reset(self) -> None:
        pass


def _compute_reward_metrics(
    neg_result: Dict[str, Any],
    creditor_target: int,
    debtor_initial: int,
    max_turn: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk a completed negotiation dialog and compute the SAME dense reward
    DQN-new / IQL would see during training. Returns per-step + episode aggregates.

    This is what lets us answer: "is there a learnable signal at all?"
    """
    dialog = neg_result.get("dialog", [])
    initial_gap = max(1, abs(debtor_initial - creditor_target))
    if max_turn is None:
        max_turn = neg_result.get("max_dialog_len") or max(1, neg_result.get("negotiation_rounds", 30))

    # Build debtor offer trajectory
    debtor_offers: List[int] = []
    for entry in dialog:
        if entry.get("speaker") == "buyer" and entry.get("requested_days") is not None:
            debtor_offers.append(int(entry["requested_days"]))

    # Per-step rewards: indexed by creditor turn (since action happens on creditor turn,
    # reward depends on debtor's *response* in the immediately following turn).
    step_rewards: List[float] = []
    step_components: List[Dict[str, float]] = []

    prev_debtor_offer: Optional[int] = debtor_initial  # seed with initial target
    rolling_debtor_history: List[int] = [debtor_initial]

    creditor_indices = [i for i, e in enumerate(dialog) if e.get("speaker") == "seller"]
    for k, ci in enumerate(creditor_indices):
        # Find the debtor response after this creditor turn
        debtor_response_idx = ci + 1
        if debtor_response_idx >= len(dialog):
            break
        debtor_entry = dialog[debtor_response_idx]
        if debtor_entry.get("speaker") != "buyer":
            continue
        new_debtor_offer = debtor_entry.get("requested_days")
        if new_debtor_offer is None:
            # No parseable offer — skip but record neutral reward
            step_rewards.append(0.0)
            step_components.append({"note": "no_offer_parsed"})
            continue
        new_debtor_offer = int(new_debtor_offer)
        rolling_debtor_history.append(new_debtor_offer)

        step_r, comp = compute_step_reward(
            prev_debtor_offer=prev_debtor_offer,
            new_debtor_offer=new_debtor_offer,
            debtor_offer_history=list(rolling_debtor_history),
            initial_gap=initial_gap,
            debtor_message=debtor_entry.get("message", ""),
            observer_breakdown_risk=0.0,  # No observer in fixed-emotion sweep
            turn=k + 1,
            max_turn=max_turn,
        )
        step_rewards.append(step_r)
        step_components.append(comp)
        prev_debtor_offer = new_debtor_offer

    success = neg_result.get("final_state") == "accept"
    final_days = neg_result.get("final_days")
    final_r, final_comp = compute_final_reward(success, final_days, creditor_target, debtor_initial)

    total_step_reward = float(sum(step_rewards))
    total_episode_reward = total_step_reward + float(final_r)

    # Concession metrics
    total_concession = 0.0
    first_step_concession_norm = 0.0
    if debtor_offers:
        total_concession = float(debtor_initial - debtor_offers[-1])
        first_step_concession_norm = float(debtor_initial - debtor_offers[0]) / initial_gap

    savings_ratio = None
    if final_days is not None:
        span = max(1, debtor_initial - creditor_target)
        savings_ratio = float(np.clip((debtor_initial - final_days) / span, 0.0, 1.0))

    return {
        "step_rewards": step_rewards,
        "step_components": step_components,
        "total_step_reward": total_step_reward,
        "final_outcome_reward": float(final_r),
        "total_episode_reward": total_episode_reward,
        "debtor_offer_trajectory": debtor_offers,
        "total_debtor_concession": total_concession,
        "total_debtor_concession_norm": total_concession / initial_gap,
        "first_step_concession_norm": first_step_concession_norm,
        "savings_ratio": savings_ratio,
        "initial_gap": initial_gap,
    }


def _run_single_negotiation(
    emotion: str,
    scenario: Dict[str, Any],
    iteration: int,
    model_creditor: str,
    model_debtor: str,
    debtor_emotion: str,
    max_dialog_len: int,
) -> Dict[str, Any]:
    """Run a single (emotion, scenario, iter) negotiation. Thread-safe — each
    call constructs its own FixedEmotionModel and DebtNegotiator."""
    from llm.negotiator import DebtNegotiator

    model = FixedEmotionModel(emotion)
    negotiator = DebtNegotiator(
        config=scenario,
        emotion_model=model,
        model_creditor=model_creditor,
        model_debtor=model_debtor,
        debtor_emotion=debtor_emotion,
        debtor_model_type="vanilla",  # debtor stays vanilla (no emotion prompts)
    )
    try:
        result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
    except Exception as e:
        print(f"      ⚠️  [{emotion}|{scenario.get('id')}|it{iteration}] negotiation failed: {e}")
        result = {
            "final_state": "breakdown",
            "final_days": None,
            "negotiation_rounds": 0,
            "dialog": [],
            "emotion_sequence": [],
            "scenario_id": scenario.get("id"),
            "error": str(e),
        }

    # Compute DQN-new-equivalent reward metrics so we can verify signal exists
    creditor_target = int(scenario.get("seller", {}).get("target_price", 30))
    debtor_initial = int(scenario.get("buyer", {}).get("target_price", creditor_target * 3))
    # Stamp max_dialog_len onto result so downstream reward + dataset builders
    # can apply the time-pressure term using the correct denominator.
    result["max_dialog_len"] = max_dialog_len
    reward_metrics = _compute_reward_metrics(
        result, creditor_target, debtor_initial, max_turn=max_dialog_len
    )

    # Also build IQL-ready (s, a, r, s', done) transitions from this episode
    transitions = build_transitions_from_episode(
        emotion_label=emotion,
        scenario=scenario,
        neg_result=result,
        iteration=iteration,
        observer_features_per_turn=None,  # Level 1 only for now
    )

    return {
        "emotion": emotion,
        "scenario": scenario.get("id"),
        "iteration": iteration,
        "success": result.get("final_state") == "accept",
        "final_state": result.get("final_state"),
        "final_days": result.get("final_days"),
        "creditor_target_days": result.get("creditor_target_days"),
        "debtor_initial_days": debtor_initial,
        "rounds": result.get("negotiation_rounds", 0),
        "emotion_sequence": result.get("emotion_sequence", []),
        # Full dialog transcript — every turn's speaker / message / offer.
        # Preserved so we can replay any episode, audit emotion adherence,
        # rebuild states with different observers, etc.
        "dialog": result.get("dialog", []),
        # DQN-new reward signals (for verifying signal exists)
        "total_episode_reward": reward_metrics["total_episode_reward"],
        "total_step_reward": reward_metrics["total_step_reward"],
        "final_outcome_reward": reward_metrics["final_outcome_reward"],
        "step_rewards": reward_metrics["step_rewards"],
        "total_debtor_concession_norm": reward_metrics["total_debtor_concession_norm"],
        "first_step_concession_norm": reward_metrics["first_step_concession_norm"],
        "savings_ratio": reward_metrics["savings_ratio"],
        "debtor_offer_trajectory": reward_metrics["debtor_offer_trajectory"],
        # IQL trajectory payload (carried back to sweep for dataset assembly)
        "_iql_transitions": transitions,
    }


def run_one_emotion(
    emotion: str,
    scenarios: List[Dict[str, Any]],
    iterations: int,
    model_creditor: str,
    model_debtor: str,
    debtor_emotion: str,
    max_dialog_len: int,
    concurrency: int = 1,
    dataset: Optional[OfflineDataset] = None,
) -> Dict[str, Any]:
    """Run `iterations` negotiations per scenario with a fixed emotion.

    With concurrency > 1, runs (scenario × iteration) tuples in parallel via
    a thread pool, exploiting DashScope round-robin for throughput.
    """
    tasks = [(scenario, it) for scenario in scenarios for it in range(iterations)]
    episode_results: List[Dict[str, Any]] = []

    if concurrency <= 1:
        for scenario, it in tasks:
            print(f"\n   [{emotion}] scenario={scenario.get('id')} iter={it+1}/{iterations}")
            r = _run_single_negotiation(
                emotion=emotion,
                scenario=scenario,
                iteration=it,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                max_dialog_len=max_dialog_len,
            )
            if dataset is not None:
                dataset.append_episode(r.pop("_iql_transitions", None))
            else:
                r.pop("_iql_transitions", None)
            episode_results.append(r)
    else:
        print(f"\n   [{emotion}] launching {len(tasks)} negotiations × concurrency={concurrency}")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(
                    _run_single_negotiation,
                    emotion=emotion,
                    scenario=scenario,
                    iteration=it,
                    model_creditor=model_creditor,
                    model_debtor=model_debtor,
                    debtor_emotion=debtor_emotion,
                    max_dialog_len=max_dialog_len,
                )
                for scenario, it in tasks
            ]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                # Aggregate transitions in main thread (OfflineDataset isn't thread-safe)
                if dataset is not None:
                    dataset.append_episode(r.pop("_iql_transitions", None))
                else:
                    r.pop("_iql_transitions", None)
                episode_results.append(r)
                print(
                    f"      [{emotion}] {i+1}/{len(tasks)} done | scenario={r['scenario']} "
                    f"iter={r['iteration']} success={r['success']} rounds={r['rounds']}"
                )

    successes = [r for r in episode_results if r["success"]]
    n_total = len(episode_results)
    n_success = len(successes)
    success_rate = n_success / max(1, n_total)
    avg_rounds = float(np.mean([r["rounds"] for r in successes])) if successes else 0.0
    avg_final_days = (
        float(np.mean([r["final_days"] for r in successes if r["final_days"] is not None]))
        if successes
        else None
    )

    # Reward signal aggregates (per-episode, ALL episodes including failures)
    all_rewards = [r["total_episode_reward"] for r in episode_results]
    all_step_rewards = [r["total_step_reward"] for r in episode_results]
    all_first_step_concession = [r["first_step_concession_norm"] for r in episode_results]
    all_total_concession = [r["total_debtor_concession_norm"] for r in episode_results]
    savings_vals = [r["savings_ratio"] for r in episode_results if r["savings_ratio"] is not None]

    mean_savings_on_success = float(np.mean(savings_vals)) if savings_vals else 0.0
    # Expected savings = P(success) · E[savings | success]
    # i.e. unconditional expected savings per episode (failures treated as 0)
    expected_savings = success_rate * mean_savings_on_success

    return {
        "emotion": emotion,
        "n_episodes": n_total,
        "n_success": n_success,
        "success_rate": success_rate,
        "avg_rounds_success": avg_rounds,
        "avg_final_days_success": avg_final_days,
        # Reward-signal aggregates
        "mean_episode_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_episode_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
        "mean_step_reward": float(np.mean(all_step_rewards)) if all_step_rewards else 0.0,
        "mean_first_step_concession": float(np.mean(all_first_step_concession)) if all_first_step_concession else 0.0,
        "mean_total_concession": float(np.mean(all_total_concession)) if all_total_concession else 0.0,
        "mean_savings_ratio": mean_savings_on_success,
        "mean_expected_savings": expected_savings,
        "episode_results": episode_results,
    }


def _signal_diagnostic(per_emotion: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Decide whether the fixed-emotion sweep produced a learnable signal.

    Three orthogonal tests; ANY one passing flags a usable signal.

    1. Outcome layer: spread of success_rate across emotions
    2. Reward layer:  between-emotion variance / within-emotion variance
                      (an F-ratio-style test, no significance levels — we just
                      want a clean ratio for interpretation)
    3. Behavior layer: spread of first-step concession across emotions
    """
    emotions = list(per_emotion.keys())
    if not emotions:
        return {"verdict": "NO_DATA"}

    sr_vals = np.array([per_emotion[e]["success_rate"] for e in emotions])
    reward_means = np.array([per_emotion[e]["mean_episode_reward"] for e in emotions])
    reward_stds = np.array([per_emotion[e]["std_episode_reward"] for e in emotions])
    first_step = np.array([per_emotion[e]["mean_first_step_concession"] for e in emotions])

    # Layer 1: success rate spread
    sr_spread = float(sr_vals.max() - sr_vals.min())
    sr_signal = sr_spread > 0.20  # > 20 percentage points

    # Layer 2: reward F-ratio (between / within)
    between_var = float(np.var(reward_means))
    within_var = float(np.mean(reward_stds ** 2)) if reward_stds.size else 1.0
    f_ratio = float(between_var / within_var) if within_var > 1e-6 else float("inf")
    reward_signal = f_ratio > 2.0

    # Layer 3: first-step concession spread
    first_spread = float(first_step.max() - first_step.min())
    behavior_signal = first_spread > 0.15  # > 15% of initial gap

    any_signal = sr_signal or reward_signal or behavior_signal
    all_signal = sr_signal and reward_signal and behavior_signal

    if all_signal:
        verdict = "STRONG_SIGNAL"
    elif any_signal:
        verdict = "PARTIAL_SIGNAL"
    else:
        verdict = "NO_SIGNAL"

    # Best/worst emotions per axis
    def _argextrema(arr, names):
        return {"best": names[int(np.argmax(arr))], "worst": names[int(np.argmin(arr))]}

    return {
        "verdict": verdict,
        "any_signal": any_signal,
        "all_signal": all_signal,
        "outcome_layer": {
            "success_rate_spread": sr_spread,
            "passes": sr_signal,
            "threshold": 0.20,
            "best_worst": _argextrema(sr_vals, emotions),
            "values": {e: float(v) for e, v in zip(emotions, sr_vals)},
        },
        "reward_layer": {
            "between_emotion_var": between_var,
            "within_emotion_var": within_var,
            "f_ratio": f_ratio,
            "passes": reward_signal,
            "threshold": 2.0,
            "best_worst": _argextrema(reward_means, emotions),
            "mean_reward": {e: float(v) for e, v in zip(emotions, reward_means)},
        },
        "behavior_layer": {
            "first_step_concession_spread": first_spread,
            "passes": behavior_signal,
            "threshold": 0.15,
            "best_worst": _argextrema(first_step, emotions),
            "values": {e: float(v) for e, v in zip(emotions, first_step)},
        },
        "interpretation": {
            "STRONG_SIGNAL": "All three layers show clear emotion-dependent variation. DQN-new has strong signal to learn from.",
            "PARTIAL_SIGNAL": "At least one layer shows variation. DQN-new may learn but signal is weaker than ideal.",
            "NO_SIGNAL": "All three layers are flat. LLM is NOT responsive to emotion conditioning. Recheck prompts, model, or reward design BEFORE training DQN-new.",
        }[verdict],
    }


def run_fixed_emotion_sweep(
    scenarios: List[Dict[str, Any]],
    emotions: Optional[List[str]] = None,
    iterations: int = 3,
    model_creditor: str = "qwen-plus",
    model_debtor: str = "qwen-plus",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results/fixed_emotion_sweep",
    concurrency: int = 6,
    save_offline_dataset: bool = True,
    dataset_filename: str = "offline_trajectories.npz",
) -> Dict[str, Any]:
    """Sweep across all emotions in the active taxonomy (or a custom subset).

    concurrency: number of parallel negotiations per emotion (uses DashScope
    key rotation under the hood). Default 6 matches the number of DashScope keys.
    """
    os.makedirs(out_dir, exist_ok=True)
    emotions = list(emotions) if emotions else list(get_emotions())

    sweep: Dict[str, Any] = {
        "experiment_type": "fixed_emotion_sweep",
        "active_taxonomy_size": len(get_emotions()),
        "emotions_tested": emotions,
        "scenarios": [s.get("id") for s in scenarios],
        "iterations_per_scenario": iterations,
        "config": {
            "model_creditor": model_creditor,
            "model_debtor": model_debtor,
            "debtor_emotion": debtor_emotion,
            "max_dialog_len": max_dialog_len,
            "concurrency": concurrency,
            "setup": "creditor uses emotion-prompted LLM; debtor is vanilla (no emotion)",
        },
        "per_emotion": {},
    }

    print(f"\n🧪 Fixed-Emotion Sweep | {len(emotions)} emotions × {len(scenarios)} scenarios × {iterations} iter")
    print(f"   = {len(emotions) * len(scenarios) * iterations} total negotiations")
    print(f"   Concurrency per emotion: {concurrency}")
    print(f"   Setup: creditor = emotion-prompted LLM | debtor = vanilla LLM (no emotion prompt)")
    print(f"   Save offline (IQL) dataset: {save_offline_dataset}")

    # Mixed-policy offline dataset accumulator (one shared dataset across all emotions)
    dataset: Optional[OfflineDataset] = OfflineDataset() if save_offline_dataset else None

    for emo in emotions:
        print("\n" + "=" * 70)
        print(f"▶︎  Testing fixed emotion: {emo}")
        print("=" * 70)
        result = run_one_emotion(
            emotion=emo,
            scenarios=scenarios,
            iterations=iterations,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            max_dialog_len=max_dialog_len,
            concurrency=concurrency,
            dataset=dataset,
        )
        sweep["per_emotion"][emo] = result
        print(
            f"   ✅ [{emo}] success={result['success_rate']:.1%} "
            f"({result['n_success']}/{result['n_episodes']}), "
            f"avg_rounds={result['avg_rounds_success']:.1f}"
        )

        # Persist intermediate snapshot after each emotion (in case of interruption)
        snapshot_path = os.path.join(out_dir, "partial_results.json")
        with open(snapshot_path, "w") as f:
            json.dump(sweep, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    # Cross-emotion comparison
    summary_rows = []
    for emo, r in sweep["per_emotion"].items():
        summary_rows.append(
            {
                "emotion": emo,
                "success_rate": r["success_rate"],
                "n_success": r["n_success"],
                "n_total": r["n_episodes"],
                "avg_rounds_success": r["avg_rounds_success"],
                "avg_final_days_success": r["avg_final_days_success"],
                "mean_episode_reward": r["mean_episode_reward"],
                "std_episode_reward": r["std_episode_reward"],
                "mean_first_step_concession": r["mean_first_step_concession"],
                "mean_total_concession": r["mean_total_concession"],
                "mean_savings_ratio": r["mean_savings_ratio"],
                "mean_expected_savings": r["mean_expected_savings"],
            }
        )
    summary_rows.sort(key=lambda x: -x["mean_episode_reward"])
    sweep["summary_ranked"] = summary_rows

    # Three-layer signal diagnostic — answers "does emotion matter?"
    sweep["signal_diagnostic"] = _signal_diagnostic(sweep["per_emotion"])

    # Persist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"fixed_sweep_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(sweep, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    print("\n" + "=" * 115)
    print("🏁  FIXED-EMOTION SWEEP RESULTS (ranked by mean episode reward)")
    print("=" * 115)
    print(
        f"{'emotion':<14} {'success':<9} {'n':<9} "
        f"{'mean_R':<10} {'std_R':<9} "
        f"{'1st-step Δ':<12} {'total Δ':<10} "
        f"{'savings':<9} {'E[savings]':<11} {'rounds':<7}"
    )
    print("-" * 115)
    for row in summary_rows:
        print(
            f"{row['emotion']:<14} "
            f"{row['success_rate']:<9.1%} "
            f"{row['n_success']}/{row['n_total']:<6} "
            f"{row['mean_episode_reward']:<10.3f} "
            f"{row['std_episode_reward']:<9.3f} "
            f"{row['mean_first_step_concession']:<12.3f} "
            f"{row['mean_total_concession']:<10.3f} "
            f"{row['mean_savings_ratio']:<9.3f} "
            f"{row['mean_expected_savings']:<11.3f} "
            f"{row['avg_rounds_success']:<7.1f}"
        )

    diag = sweep["signal_diagnostic"]
    print("\n" + "=" * 100)
    print(f"🔬  SIGNAL DIAGNOSTIC: {diag['verdict']}")
    print("=" * 100)
    print(f"   {diag['interpretation']}")
    print()
    print(f"   1) Outcome layer  — success-rate spread = {diag['outcome_layer']['success_rate_spread']:.1%} "
          f"(threshold {diag['outcome_layer']['threshold']:.0%}) → {'✅ PASS' if diag['outcome_layer']['passes'] else '❌ flat'}")
    print(f"      best={diag['outcome_layer']['best_worst']['best']}, "
          f"worst={diag['outcome_layer']['best_worst']['worst']}")
    print(f"   2) Reward layer   — F-ratio (between/within) = {diag['reward_layer']['f_ratio']:.2f} "
          f"(threshold {diag['reward_layer']['threshold']:.1f}) → {'✅ PASS' if diag['reward_layer']['passes'] else '❌ flat'}")
    print(f"      best={diag['reward_layer']['best_worst']['best']}, "
          f"worst={diag['reward_layer']['best_worst']['worst']}")
    print(f"   3) Behavior layer — 1st-step Δoffer spread = {diag['behavior_layer']['first_step_concession_spread']:.3f} "
          f"(threshold {diag['behavior_layer']['threshold']:.2f}) → {'✅ PASS' if diag['behavior_layer']['passes'] else '❌ flat'}")
    print(f"      best={diag['behavior_layer']['best_worst']['best']}, "
          f"worst={diag['behavior_layer']['best_worst']['worst']}")
    print("=" * 100)
    print(f"💾 Saved sweep JSON → {out_path}")

    # Save offline RL dataset (NPZ + meta) if requested
    if dataset is not None and dataset.n_episodes() > 0:
        dataset_path = os.path.join(out_dir, dataset_filename)
        scenario_ids = [s.get("id") for s in scenarios]
        dataset.save_npz(dataset_path, scenarios_used=scenario_ids)
        sweep["offline_dataset_path"] = dataset_path
        sweep["offline_dataset_meta"] = {
            "n_transitions": dataset.n_transitions(),
            "n_episodes": dataset.n_episodes(),
            "state_dim": dataset.state_dim,
            "n_emotions": dataset.n_emotions,
            "taxonomy": dataset.taxonomy,
        }
        # Re-save sweep JSON with dataset path included
        with open(out_path, "w") as f:
            json.dump(sweep, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)
        print(f"   IQL dataset → {dataset_path} ({dataset.n_transitions()} transitions)")

    return sweep
