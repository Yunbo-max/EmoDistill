"""
Implicit Q-Learning (IQL) for offline RL on the emotion-as-action MDP.

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit
Q-Learning" (ICLR 2022). https://arxiv.org/abs/2110.06169

Three networks (all MLPs on the 16-37 dim state):
  - Q-network    Q(s,a)  — Dueling DQN (reused from EmoDistill/networks.py)
  - V-network    V(s)    — scalar value
  - Policy-net   π(a|s)  — categorical over N emotions

Three losses (decoupled, no on-policy bootstrap):
  - V loss:   expectile_loss(τ) of (Q_target(s,a) - V(s))
              τ ≈ 0.7-0.9 implements an upper-expectile of Q
  - Q loss:   MSE(Q(s,a), r + γ * V(s') * (1-done))
              no max over a' → no extrapolation onto unseen actions
  - π loss:   AWR: -log π(a|s) * exp(β * (Q_target(s,a) - V(s)))
              clipped at exp(MAX_ADV) for stability

The policy is the deployed actor; Q and V are auxiliary heads.
"""

import math
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baselines.base_model import BaseEmotionModel
from EmoDistill.emotions import get_emotions, n_emotions, prompt_for, get_active_taxonomy_name, set_active_taxonomy
from EmoDistill.networks import DuelingDQN
from EmoDistill.offline_dataset import LoadedOfflineDataset, OfflineDataset
from EmoDistill.state_builder import state_dim as _state_dim


# --------- Networks ---------

class ValueNet(nn.Module):
    """V(s) → scalar."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class PolicyNet(nn.Module):
    """π(a|s) categorical logits over N discrete emotion actions."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)  # logits

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        logits = self.forward(s)
        return F.log_softmax(logits, dim=-1).gather(1, a.unsqueeze(1)).squeeze(1)

    def sample(self, s: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(s) / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return a, dist.log_prob(a)


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """Asymmetric L2 loss biased toward upper tail of `diff = Q(s,a) - V(s)`.

    weight = expectile if diff > 0 else (1 - expectile)
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


# --------- Trainer ---------

class IQLTrainer:
    """Offline IQL trainer with discrete-action policy.

    Lifecycle:
      iql = IQLTrainer(dataset, ...)
      for step in range(n_steps):
          metrics = iql.train_step()
      iql.save("path.pt")

    For evaluation: wrap the trained policy with IQLPolicy(dataset_meta, ckpt)
    and plug into NegotiatorNew (or DebtNegotiator) as the emotion model.
    """

    def __init__(
        self,
        dataset: LoadedOfflineDataset,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,                # Polyak target update rate
        expectile: float = 0.7,            # IQL upper-expectile
        beta: float = 3.0,                 # AWR inverse temperature
        max_advantage_exp: float = 100.0,  # clip exp(β·A) for stability
        batch_size: int = 256,
        device: Optional[str] = None,
        normalize_reward: str = "scale",   # 'none' | 'scale' | 'zscore'
    ):
        self.dataset = dataset
        if normalize_reward != "none":
            dataset.normalize_rewards(normalize_reward)

        self.state_dim = dataset.state_dim
        self.n_actions = dataset.n_actions
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.beta = beta
        self.max_advantage_exp = max_advantage_exp
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Networks
        self.q_net = DuelingDQN(state_dim=self.state_dim, n_actions=self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target = DuelingDQN(state_dim=self.state_dim, n_actions=self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.v_net = ValueNet(state_dim=self.state_dim, hidden_dim=hidden_dim).to(self.device)
        self.policy = PolicyNet(state_dim=self.state_dim, n_actions=self.n_actions, hidden_dim=hidden_dim).to(self.device)

        self.q_opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_opt = optim.Adam(self.v_net.parameters(), lr=lr)
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=lr)

        self.train_steps = 0
        self.metric_history: List[Dict[str, float]] = []

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        b = self.dataset.sample(self.batch_size)
        return {
            "s": torch.from_numpy(b["states"]).float().to(self.device),
            "a": torch.from_numpy(b["actions"]).long().to(self.device),
            "r": torch.from_numpy(b["rewards"]).float().to(self.device),
            "sn": torch.from_numpy(b["next_states"]).float().to(self.device),
            "d": torch.from_numpy(b["dones"]).float().to(self.device),
        }

    def train_step(self) -> Dict[str, float]:
        batch = self._sample_batch()
        s, a, r, sn, d = batch["s"], batch["a"], batch["r"], batch["sn"], batch["d"]

        # --- V update: expectile regression of Q_target(s,a) onto V(s) ---
        with torch.no_grad():
            q_sa_target = self.q_target(s).gather(1, a.unsqueeze(1)).squeeze(1)
        v_s = self.v_net(s)
        v_loss = expectile_loss(q_sa_target - v_s, self.expectile)

        self.v_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(self.v_net.parameters(), 1.0)
        self.v_opt.step()

        # --- Q update: TD with V(s') as bootstrap target (no max-a') ---
        with torch.no_grad():
            v_sn = self.v_net(sn)
            q_target = r + (1.0 - d) * self.gamma * v_sn
        q_sa = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_sa, q_target)

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_opt.step()

        # Polyak target update
        with torch.no_grad():
            for tp, pp in zip(self.q_target.parameters(), self.q_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(pp.data, alpha=self.tau)

        # --- Policy update: AWR ---
        with torch.no_grad():
            adv = q_sa_target - v_s
            weight = torch.exp(self.beta * adv).clamp(max=self.max_advantage_exp)
        log_pi_a = self.policy.log_prob(s, a)
        policy_loss = -(weight * log_pi_a).mean()

        self.pi_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.pi_opt.step()

        self.train_steps += 1
        metrics = {
            "step": self.train_steps,
            "v_loss": float(v_loss.item()),
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "v_mean": float(v_s.mean().item()),
            "adv_mean": float(adv.mean().item()),
            "adv_max": float(adv.max().item()),
            "weight_mean": float(weight.mean().item()),
            "reward_mean": float(r.mean().item()),
            "reward_std": float(r.std().item()),
            "q_target_mean": float(q_target.mean().item()),
        }
        return metrics

    def train(self, n_steps: int, log_every: int = 500, eval_fn=None) -> List[Dict[str, float]]:
        for step in range(n_steps):
            m = self.train_step()
            self.metric_history.append(m)
            if (step + 1) % log_every == 0 or step == 0:
                print(
                    f"   [IQL] step {step+1}/{n_steps}  "
                    f"v_loss={m['v_loss']:.4f}  q_loss={m['q_loss']:.4f}  pi_loss={m['policy_loss']:.4f}  "
                    f"Q̄={m['q_mean']:.3f}  V̄={m['v_mean']:.3f}  Ā={m['adv_mean']:.3f}"
                )
            if eval_fn is not None and (step + 1) % (log_every * 4) == 0:
                eval_fn(self, step + 1)
        return self.metric_history

    # -------- Persistence --------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Infer hidden_dim from the first PolicyNet linear layer for round-trip fidelity
        hidden_dim = int(self.policy.net[0].out_features)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "v_net": self.v_net.state_dict(),
                "policy": self.policy.state_dict(),
                "config": {
                    "state_dim": self.state_dim,
                    "n_actions": self.n_actions,
                    "hidden_dim": hidden_dim,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "expectile": self.expectile,
                    "beta": self.beta,
                    "batch_size": self.batch_size,
                },
                "training": {
                    "train_steps": self.train_steps,
                    "taxonomy": get_active_taxonomy_name(),
                },
            },
            path,
        )

    @staticmethod
    def load(path: str, dataset: Optional[LoadedOfflineDataset] = None, **overrides) -> "IQLTrainer":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        if dataset is None:
            raise ValueError("IQLTrainer.load requires a dataset for shape/normalization")
        trainer = IQLTrainer(
            dataset=dataset,
            hidden_dim=overrides.get("hidden_dim", 256),
            lr=overrides.get("lr", 3e-4),
            gamma=cfg["gamma"],
            tau=cfg["tau"],
            expectile=cfg["expectile"],
            beta=cfg["beta"],
            batch_size=cfg["batch_size"],
            normalize_reward="none",  # already normalized when dataset was saved? caller handles
        )
        trainer.q_net.load_state_dict(ckpt["q_net"])
        trainer.q_target.load_state_dict(ckpt["q_target"])
        trainer.v_net.load_state_dict(ckpt["v_net"])
        trainer.policy.load_state_dict(ckpt["policy"])
        trainer.train_steps = ckpt.get("training", {}).get("train_steps", 0)
        return trainer


# --------- Deployment-side emotion model ---------

class IQLPolicy(BaseEmotionModel):
    """Wraps a trained IQL policy network for use in NegotiatorNew (or any
    negotiator that calls `select_emotion`). Inference-only.
    """

    def __init__(
        self,
        ckpt_path: str,
        taxonomy: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.5,
        greedy: bool = True,
    ):
        if taxonomy:
            set_active_taxonomy(taxonomy)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["config"]
        self.state_dim = cfg["state_dim"]
        self.n_actions = cfg["n_actions"]
        hidden_dim = cfg.get("hidden_dim", 256)
        self.policy = PolicyNet(self.state_dim, self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()
        self.q_net = DuelingDQN(self.state_dim, self.n_actions, hidden_dim=hidden_dim).to(self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_net.eval()

        self.temperature = temperature
        self.greedy = greedy
        self.history: List[str] = []

    def select_emotion(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        state_vec = model_state.get("state_vec")
        emos = get_emotions()
        if state_vec is None:
            fallback = "neutral" if "neutral" in emos else emos[0]
            self.history.append(fallback)
            return {
                "emotion": fallback,
                "emotion_text": prompt_for(fallback),
                "confidence": 0.0,
                "exploration_rate": 0.0,
                "action_idx": emos.index(fallback),
            }

        s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(s)
            probs = F.softmax(logits / max(self.temperature, 1e-6), dim=-1).squeeze(0).cpu().numpy()
            if self.greedy:
                a = int(np.argmax(probs))
            else:
                a = int(np.random.choice(len(probs), p=probs))
            q_vals = self.q_net(s).squeeze(0).cpu().numpy()

        emo = emos[a]
        self.history.append(emo)
        return {
            "emotion": emo,
            "emotion_text": prompt_for(emo),
            "confidence": float(probs[a]),
            "exploration_rate": 0.0,
            "action_idx": a,
            "q_value": float(q_vals[a]),
            "policy_probs": probs.tolist(),
        }

    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        # Inference-only model; no updates
        pass

    def record_step(self, *args, **kwargs) -> None:
        """No-op: NegotiatorNew calls this to push transitions into a replay
        buffer for online training (DQN-new). IQLPolicy is inference-only,
        so we just swallow the call."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model_type": "iql_policy",
            "n_actions": self.n_actions,
            "history_last_5": self.history[-5:],
            "temperature": self.temperature,
            "greedy": self.greedy,
        }

    def reset(self) -> None:
        self.history = []


def run_iql_experiment(
    dataset_path: str,
    out_dir: str = "results/iql",
    n_steps: int = 50000,
    log_every: int = 500,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    expectile: float = 0.7,
    beta: float = 3.0,
    batch_size: int = 256,
    normalize_reward: str = "scale",
    seed: int = 42,
) -> Dict[str, Any]:
    """Offline IQL training driver. Loads dataset NPZ, trains for n_steps, saves."""
    os.makedirs(out_dir, exist_ok=True)

    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n📦 Loading offline dataset: {dataset_path}")
    dataset = OfflineDataset.load(dataset_path)
    print(dataset.summary())

    # Make sure the active taxonomy matches what produced this dataset
    ds_taxonomy = dataset.meta.get("taxonomy")
    if ds_taxonomy and ds_taxonomy != get_active_taxonomy_name():
        print(f"   Switching active taxonomy to {ds_taxonomy} to match dataset")
        set_active_taxonomy(ds_taxonomy)

    trainer = IQLTrainer(
        dataset=dataset,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        expectile=expectile,
        beta=beta,
        batch_size=batch_size,
        normalize_reward=normalize_reward,
    )

    print(f"\n🏋️  Training IQL for {n_steps} steps on {len(dataset)} transitions")
    trainer.train(n_steps=n_steps, log_every=log_every)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(out_dir, f"iql_{timestamp}.pt")
    metrics_path = os.path.join(out_dir, f"iql_{timestamp}_metrics.json")

    trainer.save(ckpt_path)

    # Save full history downsampled (every 100 steps → ~1000 points for 100k training)
    full_hist = trainer.metric_history
    stride = max(1, len(full_hist) // 1000)
    sampled = full_hist[::stride]
    if full_hist and sampled[-1] is not full_hist[-1]:
        sampled.append(full_hist[-1])

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_steps": n_steps,
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "gamma": gamma,
                    "expectile": expectile,
                    "beta": beta,
                    "batch_size": batch_size,
                    "normalize_reward": normalize_reward,
                    "seed": seed,
                },
                "dataset_meta": dataset.meta,
                "metrics_history": sampled,
                "metrics_last_100": full_hist[-100:],
                "final_metrics": full_hist[-1] if full_hist else {},
                "history_stride": stride,
                "history_full_len": len(full_hist),
            },
            f,
            indent=2,
            default=str,
        )

    # Also save metric arrays as NPZ for plot-friendly access
    if full_hist:
        npz_path = metrics_path.replace(".json", ".npz")
        arrays = {k: np.array([m[k] for m in full_hist]) for k in full_hist[0].keys()}
        np.savez_compressed(npz_path, **arrays)
        print(f"\n💾 Saved metric arrays → {npz_path}")

    print(f"\n💾 Saved IQL checkpoint → {ckpt_path}")
    print(f"💾 Saved training metrics → {metrics_path}")
    return {
        "ckpt_path": ckpt_path,
        "metrics_path": metrics_path,
        "final_metrics": trainer.metric_history[-1] if trainer.metric_history else {},
    }
