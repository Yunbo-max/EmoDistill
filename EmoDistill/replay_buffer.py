"""
Prioritized Experience Replay (PER) with n-step return support.

Stores per-episode trajectories so n-step returns never cross episode boundaries.
"""

import numpy as np
import random
from collections import namedtuple
from typing import List, Tuple

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done", "episode_id", "step_id"],
)


class PrioritizedReplayBuffer:
    """PER with proportional priorities + n-step return."""

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        n_step: int = 3,
        gamma: float = 0.95,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.n_step = n_step
        self.gamma = gamma
        self.epsilon = epsilon

        self.storage: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

        self._episode_index: dict = {}  # episode_id -> list of buffer indices

    def add_episode(self, transitions: List[Transition]):
        """Push a complete episode trajectory. n-step lookup will be episode-safe."""
        if not transitions:
            return
        episode_id = transitions[0].episode_id
        self._episode_index[episode_id] = []

        for t in transitions:
            idx = self.position
            if self.size < self.capacity:
                self.storage.append(t)
                self.size += 1
            else:
                old_t = self.storage[idx]
                self._remove_from_episode_index(old_t.episode_id, idx)
                self.storage[idx] = t

            self.priorities[idx] = self.max_priority
            self._episode_index[episode_id].append(idx)
            self.position = (self.position + 1) % self.capacity

    def _remove_from_episode_index(self, episode_id: int, idx: int):
        if episode_id in self._episode_index:
            try:
                self._episode_index[episode_id].remove(idx)
                if not self._episode_index[episode_id]:
                    del self._episode_index[episode_id]
            except ValueError:
                pass

    def __len__(self):
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def _compute_n_step_target(self, idx: int) -> Tuple[float, np.ndarray, bool, int]:
        """Walk forward up to n_step within the same episode, accumulating discounted rewards.

        Returns (cumulative_reward, n_step_next_state, done, actual_n).
        """
        anchor = self.storage[idx]
        episode_id = anchor.episode_id
        ep_indices = self._episode_index.get(episode_id, [])
        if not ep_indices:
            return anchor.reward, anchor.next_state, anchor.done, 1

        anchor_step = anchor.step_id
        cum_r = 0.0
        last_next = anchor.next_state
        done = False
        actual_n = 0

        for k in range(self.n_step):
            target_step = anchor_step + k
            found = None
            for buf_i in ep_indices:
                tr = self.storage[buf_i]
                if tr.step_id == target_step:
                    found = tr
                    break
            if found is None:
                break
            cum_r += (self.gamma ** k) * found.reward
            last_next = found.next_state
            actual_n = k + 1
            if found.done:
                done = True
                break

        return cum_r, last_next, done, actual_n

    def sample(self, batch_size: int):
        """Returns (states, actions, n_step_rewards, n_step_next_states, dones, ns, indices, weights)."""
        probs = self.priorities[: self.size] ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs = probs / probs_sum

        indices = np.random.choice(self.size, batch_size, p=probs)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards_n, next_states_n, dones, ns = [], [], [], [], [], []
        for i in indices:
            anchor = self.storage[i]
            cum_r, next_s, done, actual_n = self._compute_n_step_target(i)
            states.append(anchor.state)
            actions.append(anchor.action)
            rewards_n.append(cum_r)
            next_states_n.append(next_s)
            dones.append(done)
            ns.append(actual_n)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards_n, dtype=np.float32),
            np.asarray(next_states_n, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(ns, dtype=np.int64),
            indices,
            np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td in zip(indices, td_errors):
            p = (abs(float(td)) + self.epsilon) ** 1.0
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p
