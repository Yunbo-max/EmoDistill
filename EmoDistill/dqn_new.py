"""
DQN-new: emotion-as-action meta-policy over a frozen negotiation LLM.

Differences vs the original DQN baseline (baselines/dqn_baseline.py):
1. State is two-level (16-dim):
   - Level 1 (13 dim): behavioral features from offer trajectory + history
   - Level 2 (3 dim): observer LLM (deal_prob, breakdown_risk, opponent_emotion)
2. Reward is DENSE per-step grounded in Δopponent_offer (objective), not sparse
   terminal reward distributed equally.
3. Double DQN + Dueling + PER + n-step (n=3).
4. State is contextual (depends on dialog/offers/observer), not just emotion history.

This class plugs into the existing BaseEmotionModel interface so the matching
NegotiatorNew (in EmoDistill.negotiator_new) can drive it.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from baselines.base_model import BaseEmotionModel
from EmoDistill.networks import DuelingDQN
from EmoDistill.replay_buffer import PrioritizedReplayBuffer, Transition
from EmoDistill.emotions import (
    get_emotions,
    n_emotions,
    emotion_to_idx,
    prompt_for,
    get_active_taxonomy_name,
)
from EmoDistill.state_builder import state_dim as _state_dim
from EmoDistill.reward import compute_step_reward, compute_final_reward


class DQNNew(BaseEmotionModel):
    """Two-level-state DQN with dense per-step reward."""

    def __init__(
        self,
        state_dim: int = None,
        n_actions: int = None,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 50000,
        batch_size: int = 64,
        n_step: int = 3,
        tau: float = 0.005,
        train_every: int = 4,
        warmup: int = 500,
        use_observer: bool = True,
        device: Optional[str] = None,
    ):
        if state_dim is None:
            state_dim = _state_dim()
        if n_actions is None:
            n_actions = n_emotions()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.taxonomy = get_active_taxonomy_name()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.train_every = train_every
        self.warmup = warmup
        self.use_observer = use_observer
        self.learning_rate = learning_rate

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.policy_net = DuelingDQN(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.buffer = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            n_step=n_step,
            gamma=gamma,
        )

        # Per-episode rollout cache (populated by NegotiatorNew via record_step)
        self._episode_id = 0
        self._current_episode_steps: List[Dict[str, Any]] = []

        # Aggregate stats
        self.total_episodes = 0
        self.total_train_steps = 0
        self.episode_rewards: List[float] = []
        self.train_losses: List[float] = []
        self.best_reward = -float("inf")
        self.best_sequence: Optional[List[str]] = None

        # Cache used by select_emotion when called inline (no batched state available)
        self._last_state_vec: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

    # -------- BaseEmotionModel interface --------

    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Select emotion given a model_state dict.

        Expected keys in `model_state` (provided by NegotiatorNew):
          - 'state_vec' (np.ndarray, shape (state_dim,))  REQUIRED
          - 'round' (int)
        Falls back to neutral if 'state_vec' is missing (defensive against the
        vanilla negotiator calling us without enrichment).
        """
        state_vec = model_state.get("state_vec")
        if state_vec is None:
            # Fallback: zero state, neutral-like action
            print("        ⚠️  DQNNew got no state_vec; falling back to neutral")
            emos = get_emotions()
            fallback = "neutral" if "neutral" in emos else emos[0]
            return {
                "emotion": fallback,
                "emotion_text": prompt_for(fallback),
                "confidence": 0.0,
                "exploration_rate": self.epsilon,
            }

        state_t = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
            q_vals = None
        else:
            with torch.no_grad():
                q_vals = self.policy_net(state_t)
                action = int(q_vals.argmax(dim=1).item())

        self._last_state_vec = np.asarray(state_vec, dtype=np.float32)
        self._last_action = action

        emos = get_emotions()
        emotion = emos[action]
        confidence = 0.5
        if q_vals is not None:
            qv = q_vals.squeeze(0).cpu().numpy()
            qmax = float(qv.max())
            qsel = float(qv[action])
            confidence = float(qsel / (qmax + 1e-8)) if qmax > 0 else 0.5

        return {
            "emotion": emotion,
            "emotion_text": prompt_for(emotion),
            "confidence": confidence,
            "exploration_rate": self.epsilon,
            "action_idx": action,
        }

    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Push the just-recorded episode into the buffer and run training steps."""
        self.total_episodes += 1
        episode_steps = list(self._current_episode_steps)
        if not episode_steps:
            self._reset_episode_cache()
            return

        # Compose terminal bonus and attach to last step's reward
        success = negotiation_result.get("final_state") == "accept"
        final_days = negotiation_result.get("final_days")
        creditor_target = negotiation_result.get("creditor_target_days", 30)
        debtor_initial = episode_steps[0].get("debtor_initial", creditor_target * 3)
        final_bonus, _ = compute_final_reward(success, final_days, creditor_target, debtor_initial)
        episode_steps[-1]["reward"] += final_bonus
        episode_steps[-1]["done"] = True

        # Build Transition list
        eid = self._episode_id
        transitions: List[Transition] = []
        ep_reward = 0.0
        for step_id, step in enumerate(episode_steps):
            transitions.append(
                Transition(
                    state=np.asarray(step["state"], dtype=np.float32),
                    action=int(step["action"]),
                    reward=float(step["reward"]),
                    next_state=np.asarray(step["next_state"], dtype=np.float32),
                    done=bool(step["done"]),
                    episode_id=eid,
                    step_id=step_id,
                )
            )
            ep_reward += float(step["reward"])

        self.buffer.add_episode(transitions)
        self.episode_rewards.append(ep_reward)

        emos = get_emotions()
        emotion_seq = [emos[t.action] for t in transitions]
        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            self.best_sequence = emotion_seq

        # Training: run multiple gradient updates proportional to new data
        n_updates = max(1, len(transitions) // self.train_every)
        if len(self.buffer) >= max(self.batch_size, self.warmup):
            for _ in range(n_updates):
                loss = self._train_step()
                if loss is not None:
                    self.train_losses.append(loss)

        # Epsilon decay per episode
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_id += 1
        self._reset_episode_cache()

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total_episodes": self.total_episodes,
            "total_train_steps": self.total_train_steps,
            "buffer_size": len(self.buffer),
            "epsilon": float(self.epsilon),
            "best_reward": float(self.best_reward) if self.best_reward != -float("inf") else None,
        }
        if self.episode_rewards:
            w = min(20, len(self.episode_rewards))
            recent = self.episode_rewards[-w:]
            stats["avg_reward_last"] = float(np.mean(recent))
            stats["std_reward_last"] = float(np.std(recent))
        if self.train_losses:
            w = min(50, len(self.train_losses))
            stats["avg_loss_last"] = float(np.mean(self.train_losses[-w:]))
        return stats

    def reset(self) -> None:
        """Called by negotiator between episodes."""
        self._reset_episode_cache()

    # -------- NegotiatorNew-facing hooks --------

    def record_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        debtor_initial: int,
    ) -> None:
        """Stash a (s,a,r,s',done) tuple to be flushed to buffer when episode ends."""
        self._current_episode_steps.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32),
                "done": bool(done),
                "debtor_initial": int(debtor_initial),
            }
        )

    def _reset_episode_cache(self) -> None:
        self._current_episode_steps = []
        self._last_state_vec = None
        self._last_action = None

    # -------- Training core --------

    def _train_step(self) -> Optional[float]:
        if not self.buffer.can_sample(self.batch_size):
            return None

        (states, actions, rewards_n, next_states_n, dones, ns, indices, weights) = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards_n).to(self.device)
        next_states_t = torch.from_numpy(next_states_n).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        ns_t = torch.from_numpy(ns).float().to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)

        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: argmax from policy, value from target
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            gamma_n = self.gamma ** ns_t
            target = rewards_t + (1.0 - dones_t) * gamma_n * next_q

        td_errors = target - q_current
        loss = (weights_t * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        # Soft update target
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(pp.data, alpha=self.tau)

        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.total_train_steps += 1
        return float(loss.item())

    # -------- Persistence --------

    def save_model(self, filepath: str) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "n_actions": self.n_actions,
                    "gamma": self.gamma,
                    "learning_rate": self.learning_rate,
                    "tau": self.tau,
                    "n_step": self.buffer.n_step,
                    "use_observer": self.use_observer,
                },
                "training": {
                    "total_episodes": self.total_episodes,
                    "total_train_steps": self.total_train_steps,
                    "best_reward": float(self.best_reward) if self.best_reward != -float("inf") else None,
                    "best_sequence": self.best_sequence,
                    "epsilon": self.epsilon,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            print(f"⚠️  Model file {filepath} not found")
            return
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass


def run_dqn_new_experiment(
    scenarios: List[Dict[str, Any]],
    episodes: int = 600,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    observer_model: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results",
    use_observer: bool = True,
    learning_rate: float = 1e-4,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    n_step: int = 3,
    tau: float = 0.005,
    hidden_dim: int = 128,
) -> Dict[str, Any]:
    """Driver: trains DQN-new across `episodes`, cycling scenarios."""
    from EmoDistill.negotiator_new import NegotiatorNew  # local import avoids circular

    os.makedirs(out_dir, exist_ok=True)

    model = DQNNew(
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        n_step=n_step,
        tau=tau,
        hidden_dim=hidden_dim,
        use_observer=use_observer,
    )

    results: Dict[str, Any] = {
        "experiment_type": "dqn_new",
        "config": {
            "episodes": episodes,
            "use_observer": use_observer,
            "model_creditor": model_creditor,
            "model_debtor": model_debtor,
            "observer_model": observer_model,
            "max_dialog_len": max_dialog_len,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "batch_size": batch_size,
            "n_step": n_step,
            "tau": tau,
            "hidden_dim": hidden_dim,
        },
        "scenarios_used": [s.get("id") for s in scenarios],
        "episode_results": {},
        "learning_curve": [],
    }

    for ep in range(episodes):
        scenario = scenarios[ep % len(scenarios)]
        print(f"\n🧠 [DQN-new] Episode {ep+1}/{episodes} | scenario={scenario.get('id')} | ε={model.epsilon:.3f} | buf={len(model.buffer)}")

        negotiator = NegotiatorNew(
            config=scenario,
            emotion_model=model,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            observer_model=observer_model,
            use_observer=use_observer,
            max_dialog_len=max_dialog_len,
        )
        neg_result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
        model.update_model(neg_result)

        success = neg_result.get("final_state") == "accept"
        rounds = neg_result.get("negotiation_rounds", 0)
        episode_reward = model.episode_rewards[-1] if model.episode_rewards else 0.0

        results["episode_results"][f"episode_{ep+1}"] = {
            "scenario": scenario.get("id"),
            "success": success,
            "rounds": rounds,
            "final_days": neg_result.get("final_days"),
            "emotion_sequence": neg_result.get("emotion_sequence", []),
            "reward": episode_reward,
            "epsilon": model.epsilon,
        }
        results["learning_curve"].append(
            {
                "episode": ep + 1,
                "reward": episode_reward,
                "success": success,
                "rounds": rounds,
                "epsilon": model.epsilon,
                "buffer_size": len(model.buffer),
                "train_steps": model.total_train_steps,
            }
        )

        if (ep + 1) % 20 == 0:
            stats = model.get_stats()
            print(f"   📊 Eval@{ep+1}: avg_reward={stats.get('avg_reward_last', 0):.2f}, loss={stats.get('avg_loss_last', 0):.4f}")

    # Aggregate
    succ_eps = [r for r in results["episode_results"].values() if r["success"]]
    overall_success = len(succ_eps) / max(1, episodes)
    avg_rounds = float(np.mean([r["rounds"] for r in succ_eps])) if succ_eps else 0.0

    results["final_stats"] = model.get_stats()
    results["best_sequence"] = model.best_sequence
    results["best_reward"] = model.best_reward if model.best_reward != -float("inf") else None
    results["performance"] = {
        "success_rate": overall_success,
        "avg_negotiation_rounds": avg_rounds,
        "total_episodes": episodes,
        "successful_episodes": len(succ_eps),
    }

    # Persist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"dqn_new_{timestamp}.json")
    model_path = os.path.join(out_dir, f"dqn_new_model_{timestamp}.pt")

    with open(json_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )
    model.save_model(model_path)

    print(f"\n💾 Saved results → {json_path}")
    print(f"💾 Saved model   → {model_path}")
    print(f"✅ Success rate: {overall_success:.1%} | Avg rounds (success): {avg_rounds:.1f}")
    return results
