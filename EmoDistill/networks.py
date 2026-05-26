"""
Dueling DQN network for DQN-new.

Input: 16-dim state vector (Level 1: 13 dim + Level 2: 3 dim)
Output: Q-values for 7 emotion actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))"""

    def __init__(self, state_dim: int = 16, n_actions: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
