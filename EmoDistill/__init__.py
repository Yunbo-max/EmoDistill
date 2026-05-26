"""
DQN-new: Two-level state DQN with dense per-step reward and Observer LLM.

Architecture:
- Level 1 (13 dim): Behavioral features computed from offer trajectory + history
- Level 2 (3 dim): Observer LLM outputs (deal_prob, breakdown_risk, opponent_emotion)
- Total state: 16 dim, Action: 7 emotions
- Algorithm: Double DQN + Dueling + PER + n-step (n=3)
"""

from EmoDistill.dqn_new import DQNNew, run_dqn_new_experiment

__all__ = ["DQNNew", "run_dqn_new_experiment"]
