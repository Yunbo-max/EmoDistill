"""
Model 3: DQN (Deep Q-Network) Baseline
Uses neural networks to approximate Q-values for emotional transitions
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque, namedtuple
from baselines.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

# Emotion definitions
EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# Experience replay memory
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'context'])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling exponent
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, experience: Experience, td_error: Optional[float] = None):
        priority = (abs(td_error) + 1e-5) ** self.alpha if td_error is not None else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)
    
#     def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
#         if self.size < batch_size:
#             batch_size = self.size
            
#         # Calculate sampling probabilities
#         priorities = self.priorities[:self.size]
#         probabilities = priorities ** self.alpha
#         probabilities /= probabilities.sum()
        
#         # Sample indices
#         indices = np.random.choice(self.size, batch_size, p=probabilities)
        
#         # Calculate importance sampling weights
#         total = self.size
#         weights = (total * probabilities[indices]) ** (-self.beta)
#         weights /= weights.max()
        
#         # Update beta
#         self.beta = min(1.0, self.beta + self.beta_increment)
        
#         experiences = [self.buffer[idx] for idx in indices]
#         return experiences, indices, np.array(weights, dtype=np.float32)
    
#     def update_priorities(self, indices: List[int], td_errors: np.ndarray):
#         for idx, td_error in zip(indices, td_errors):
#             priority = (abs(td_error) + 1e-5) ** self.alpha
#             self.priorities[idx] = priority
    
#     def __len__(self):
#         return self.size

# class StateEncoder(nn.Module):
#     """Encode state with context information"""
#     def __init__(self, state_dim: int = 7, context_dim: int = 10):
#         super().__init__()
#         self.state_dim = state_dim
#         self.context_dim = context_dim
        
#         # State embedding (one-hot emotion + context)
#         self.state_embedding = nn.Linear(state_dim, 32)
#         self.context_embedding = nn.Linear(context_dim, 32)
        
#         # Combined features
#         self.combined = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.LayerNorm(128),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.LayerNorm(64)
#         )
        
#         # Output layer for Q-values
#         self.q_head = nn.Linear(64, state_dim)
        
#         # Value head for dueling architecture
#         self.value_head = nn.Linear(64, 1)
#         self.advantage_head = nn.Linear(64, state_dim)
        
#     def forward(self, state_onehot: torch.Tensor, context: torch.Tensor, 
#                 dueling: bool = True) -> torch.Tensor:
#         # Encode state and context
#         state_encoded = F.relu(self.state_embedding(state_onehot))
#         context_encoded = F.relu(self.context_embedding(context))
        
#         # Combine
#         combined = torch.cat([state_encoded, context_encoded], dim=-1)
#         features = self.combined(combined)
        
#         if dueling:
#             # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
#             value = self.value_head(features)
#             advantage = self.advantage_head(features)
#             q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
#         else:
#             # Standard DQN
#             q_values = self.q_head(features)
            
#         return q_values

# class DQNBaseline(BaseEmotionModel):
#     """
#     Deep Q-Network Baseline for emotional transition optimization
#     Uses neural networks with experience replay and target networks
#     """
    
#     def __init__(
#         self,
#         learning_rate: float = 1e-4,
#         discount_factor: float = 0.95,
#         exploration_rate: float = 1.0,
#         exploration_decay: float = 0.995,
#         min_exploration: float = 0.05,
#         replay_buffer_size: int = 10000,
#         batch_size: int = 32,
#         target_update_freq: int = 100,
#         use_double_dqn: bool = True,
#         use_dueling: bool = True,
#         use_per: bool = True,
#         tau: float = 0.01,  # For soft target updates
#         device: str = "cpu"
#     ):
#         # DQN parameters
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.min_exploration = min_exploration
#         self.batch_size = batch_size
#         self.target_update_freq = target_update_freq
#         self.use_double_dqn = use_double_dqn
#         self.use_dueling = use_dueling
#         self.use_per = use_per
#         self.tau = tau
        
#         # Device
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")
#         print(f"DQN using device: {self.device}")
        
#         # Context dimensions (negotiation context features)
#         self.context_dim = 10  # round_num, dialog_length, progress, etc.
        
#         # Networks
#         self.policy_net = StateEncoder(N_EMOTIONS, self.context_dim).to(self.device)
#         self.target_net = StateEncoder(N_EMOTIONS, self.context_dim).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
        
#         # Optimizer
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
#         # Replay buffer
#         if use_per:
#             self.replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)
#         else:
#             self.replay_buffer = deque(maxlen=replay_buffer_size)
        
#         # Current state
#         self.current_emotion_idx = EMOTION_TO_IDX['neutral']
#         self.emotion_history = []
        
#         # Training tracking
#         self.total_steps = 0
#         self.total_episodes = 0
#         self.episode_rewards = []
#         self.training_losses = []
        
#         # Best performance tracking
#         self.best_reward = -float('inf')
#         self.best_sequence = None
        
#         # Context features for current state
#         self.context_features = np.zeros(self.context_dim)
        
#     def _extract_context_features(self, state: Dict[str, Any]) -> np.ndarray:
#         """Extract context features for neural network input"""
#         round_num = state.get('round', 1)
#         dialog_length = state.get('dialog_length', 0)
#         max_dialog_len = state.get('max_dialog_len', 30)
        
#         # Normalized features
#         features = np.zeros(self.context_dim)
        
#         # Basic features
#         features[0] = round_num / 10.0  # Normalized round number
#         features[1] = dialog_length / max_dialog_len  # Progress
#         features[2] = state.get('debt_ratio', 0.5)  # Debt amount ratio
#         features[3] = state.get('concession_rate', 0.0)  # Concession rate
        
#         # Emotional history features (last 3 emotions)
#         if len(self.emotion_history) >= 1:
#             last_emotion_idx = EMOTION_TO_IDX.get(self.emotion_history[-1], 0)
#             features[4] = last_emotion_idx / (N_EMOTIONS - 1)
        
#         if len(self.emotion_history) >= 2:
#             second_last_idx = EMOTION_TO_IDX.get(self.emotion_history[-2], 0)
#             features[5] = second_last_idx / (N_EMOTIONS - 1)
#             features[6] = 1.0 if last_emotion_idx == second_last_idx else 0.0  # Emotion repeat
        
#         # Strategic features
#         features[7] = min(1.0, dialog_length / 5.0)  # Early vs late stage
#         features[8] = state.get('agreement_progress', 0.0)  # Progress towards agreement
#         features[9] = random.random()  # Small noise for exploration
        
#         return features
    
#     def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         """Select next emotion using DQN policy"""
#         # Update context features
#         self.context_features = self._extract_context_features(state)
        
#         # Prepare state tensor
#         state_onehot = np.zeros(N_EMOTIONS)
#         state_onehot[self.current_emotion_idx] = 1.0
        
#         state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
#         context_tensor = torch.FloatTensor(self.context_features).unsqueeze(0).to(self.device)
        
#         # Epsilon-greedy action selection
#         if random.random() < self.exploration_rate:
#             # Explore: random action
#             next_idx = random.randint(0, N_EMOTIONS - 1)
#             q_values = None
#         else:
#             # Exploit: best action according to policy network
#             with torch.no_grad():
#                 q_values = self.policy_net(state_tensor, context_tensor, self.use_dueling)
#                 next_idx = q_values.argmax(dim=1).item()
        
#         next_emotion = EMOTIONS[next_idx]
        
#         # Store action for experience replay
#         self.last_state = {
#             'state_onehot': state_onehot.copy(),
#             'context': self.context_features.copy(),
#             'state_idx': self.current_emotion_idx
#         }
#         self.last_action = next_idx

      
        
#         # Update state
#         prev_emotion_idx = self.current_emotion_idx
#         self.current_emotion_idx = next_idx
#         self.emotion_history.append(next_emotion)
        
#         # Calculate confidence if we have Q-values
#         confidence = 0.7
#         if q_values is not None:
#             max_q = q_values.max().item()
#             selected_q = q_values[0, next_idx].item()
#             confidence = min(1.0, selected_q / (max_q + 1e-8))
        
#         # Get emotion prompt
#         emotion_prompts = {
#             "happy": "Use an optimistic and positive tone, expressing confidence",
#             "surprising": "Use an engaging and unexpected approach",
#             "angry": "Use a firm and assertive tone, emphasizing urgency",
#             "sad": "Use an empathetic and understanding tone",
#             "disgust": "Use a disappointed tone while remaining professional",
#             "fear": "Use a cautious and concerned tone",
#             "neutral": "Use a balanced and professional tone"
#         }
        
#         return {
#             "emotion": next_emotion,
#             "emotion_text": emotion_prompts.get(next_emotion, "Use a professional tone"),
#             "confidence": float(confidence),
#             "exploration_rate": float(self.exploration_rate),
#             "using_dqn": True,
#             "network_confidence": float(confidence),
#             "context_features": self.context_features.tolist()
#         }
    
#     def calculate_reward(self, negotiation_result: Dict[str, Any], 
#                         emotion_sequence: List[str]) -> float:
#         """Calculate reward for emotional sequence"""
#         success = negotiation_result.get('final_state') == 'accept'
#         collection_days = negotiation_result.get('collection_days', 0)
#         target_days = negotiation_result.get('creditor_target_days', 30)
#         negotiation_rounds = negotiation_result.get('negotiation_rounds', 1)
#         sequence_length = len(emotion_sequence)
        
#         if not success:
#             return -5.0  # Penalty for failure
        
#         # Base reward for success
#         reward = 10.0
        
#         # Reward for efficiency
#         if target_days > 0:
#             time_efficiency = max(0, (target_days - collection_days) / target_days)
#             reward += 20.0 * time_efficiency
        
#         # Reward for shorter negotiations
#         round_efficiency = 1.0 / (1 + np.log(negotiation_rounds + 1))
#         reward += 15.0 * round_efficiency
        
#         # Strategic bonus
#         strategy_bonus = self._calculate_strategy_bonus(emotion_sequence)
#         reward += strategy_bonus
        
#         # Smoothness bonus (avoid erratic emotion changes)
#         smoothness = self._calculate_smoothness(emotion_sequence)
#         reward += 5.0 * smoothness
        
#         return reward
    
#     def _calculate_strategy_bonus(self, emotion_sequence: List[str]) -> float:
#         """Calculate bonus for strategic emotion patterns"""
#         if len(emotion_sequence) < 2:
#             return 0.0
        
#         bonus = 0.0
        
#         # Strategic patterns (learned from negotiation theory)
#         patterns = {
#             ('neutral', 'angry', 'neutral'): 3.0,  # Firm stance then compromise
#             ('neutral', 'sad', 'happy'): 2.5,      # Empathy then positivity
#             ('angry', 'sad', 'happy'): 3.0,        # Firm -> understanding -> resolution
#             ('fear', 'neutral', 'happy'): 2.0,     # Concern to resolution
#             ('neutral', 'surprising', 'happy'): 2.0, # Surprise then positivity
#         }
        
#         for i in range(len(emotion_sequence) - 2):
#             pattern = tuple(emotion_sequence[i:i+3])
#             if pattern in patterns:
#                 bonus += patterns[pattern]
        
#         # Diversity bonus (using varied emotions strategically)
#         unique_emotions = len(set(emotion_sequence))
#         if len(emotion_sequence) >= 4 and unique_emotions >= 3:
#             bonus += 2.0 * (unique_emotions / len(emotion_sequence))
        
#         # Ending bonus
#         final_emotion = emotion_sequence[-1]
#         if final_emotion in ['happy', 'neutral']:
#             bonus += 1.5
#         elif final_emotion == 'surprising':
#             bonus += 1.0
        
#         return bonus
    
#     def _calculate_smoothness(self, emotion_sequence: List[str]) -> float:
#         """Calculate smoothness of emotion transitions"""
#         if len(emotion_sequence) < 2:
#             return 1.0
        
#         # Emotion similarity matrix (hand-crafted)
#         similarity = {
#             ('happy', 'happy'): 1.0,
#             ('happy', 'surprising'): 0.7,
#             ('happy', 'neutral'): 0.6,
#             ('happy', 'sad'): 0.2,
#             ('happy', 'angry'): 0.1,
#             ('happy', 'disgust'): 0.1,
#             ('happy', 'fear'): 0.2,
#             ('neutral', 'neutral'): 1.0,
#             ('neutral', 'happy'): 0.6,
#             ('neutral', 'surprising'): 0.5,
#             ('neutral', 'sad'): 0.4,
#             ('neutral', 'angry'): 0.3,
#             ('neutral', 'disgust'): 0.3,
#             ('neutral', 'fear'): 0.4,
#             ('angry', 'angry'): 1.0,
#             ('angry', 'disgust'): 0.8,
#             ('angry', 'neutral'): 0.3,
#             ('angry', 'sad'): 0.4,
#             ('angry', 'fear'): 0.2,
#             ('sad', 'sad'): 1.0,
#             ('sad', 'fear'): 0.7,
#             ('sad', 'neutral'): 0.4,
#             ('sad', 'happy'): 0.2,
#         }
        
#         smoothness_sum = 0.0
#         for i in range(len(emotion_sequence) - 1):
#             from_emotion = emotion_sequence[i]
#             to_emotion = emotion_sequence[i + 1]
            
#             key = (from_emotion, to_emotion)
#             if key in similarity:
#                 smoothness_sum += similarity[key]
#             else:
#                 # Try reverse
#                 reverse_key = (to_emotion, from_emotion)
#                 if reverse_key in similarity:
#                     smoothness_sum += similarity[reverse_key]
#                 else:
#                     smoothness_sum += 0.1  # Default low similarity
        
#         return smoothness_sum / (len(emotion_sequence) - 1)
    

#     # Add a method to handle step updates
#     def step_update(self, next_context: np.ndarray, step_reward: float, done: bool = False):
#         """Update after each step in the negotiation"""
#         # Store transition
#         self.store_transition(next_context, step_reward, done)
        
#         # Train if we have enough samples
#         if len(self.replay_buffer) >= self.batch_size:
#             loss = self.train_step()
#             if loss is not None:
#                 self.training_losses.append(loss)
#                 self.total_steps += 1
        
#     def store_transition(self, next_context: np.ndarray, reward: float, done: bool):
#         """Store transition in replay buffer"""
#         if not hasattr(self, 'last_state') or not hasattr(self, 'last_action'):
#             print(f"⚠️ Warning: Cannot store transition - missing state or action")
#             return
        
#         # Get next state one-hot
#         next_state_onehot = np.zeros(N_EMOTIONS)
#         next_state_onehot[self.current_emotion_idx] = 1.0
        
#         experience = Experience(
#             state=self.last_state['state_onehot'],
#             action=self.last_action,
#             reward=reward,
#             next_state=next_state_onehot,
#             context=self.last_state['context'],
#             done=done
#         )
        
#         # Add to replay buffer
#         if self.use_per:
#             # For PER, we need TD error which we'll update after learning
#             self.replay_buffer.add(experience, td_error=1.0)  # Initial priority
#         else:
#             self.replay_buffer.append(experience)
        
#         # Clear for next step
#         if not done:
#             # Update last state for next step
#             self.last_state = {
#                 'state_onehot': next_state_onehot.copy(),
#                 'context': next_context.copy(),
#                 'state_idx': self.current_emotion_idx
#             }
    
#     def _get_next_state_onehot(self) -> np.ndarray:
#         """Get one-hot encoding of next state (current emotion)"""
#         state_onehot = np.zeros(N_EMOTIONS)
#         state_onehot[self.current_emotion_idx] = 1.0
#         return state_onehot
    
#     def train_step(self) -> Optional[float]:
#         """Perform one training step using experience replay"""
#         if len(self.replay_buffer) < self.batch_size:
#             return None
        
#         # Sample batch
#         if self.use_per:
#             batch, indices, weights = self.replay_buffer.sample(self.batch_size)
#             weights = torch.FloatTensor(weights).to(self.device)
#         else:
#             batch = random.sample(self.replay_buffer, self.batch_size)
#             indices = None
#             weights = torch.ones(self.batch_size).to(self.device)
        
#         # Prepare batch tensors
#         states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(self.device)
#         actions = torch.LongTensor(np.array([exp.action for exp in batch])).to(self.device)
#         rewards = torch.FloatTensor(np.array([exp.reward for exp in batch])).to(self.device)
#         next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(self.device)
#         contexts = torch.FloatTensor(np.array([exp.context for exp in batch])).to(self.device)
#         dones = torch.FloatTensor(np.array([exp.done for exp in batch])).to(self.device)
        
#         # Compute current Q values
#         current_q_values = self.policy_net(states, contexts, self.use_dueling)
#         current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
#         # Compute next Q values
#         with torch.no_grad():
#             if self.use_double_dqn:
#                 # Double DQN: use policy net for action, target net for value
#                 next_actions = self.policy_net(next_states, contexts, self.use_dueling).argmax(1)
#                 next_q_values = self.target_net(next_states, contexts, self.use_dueling)
#                 next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
#             else:
#                 # Standard DQN
#                 next_q_values = self.target_net(next_states, contexts, self.use_dueling)
#                 next_q = next_q_values.max(1)[0]
            
#             # Compute target Q values
#             target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
#         # Compute loss
#         td_errors = target_q - current_q
#         loss = (weights * td_errors.pow(2)).mean()
        
#         # Update priorities if using PER
#         if self.use_per and indices is not None:
#             self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
#         self.optimizer.step()
        
#         self.total_steps += 1
        
#         # Update target network
#         if self.total_steps % self.target_update_freq == 0:
#             if self.tau < 1.0:
#                 # Soft update
#                 for target_param, policy_param in zip(self.target_net.parameters(), 
#                                                      self.policy_net.parameters()):
#                     target_param.data.copy_(self.tau * policy_param.data + 
#                                            (1 - self.tau) * target_param.data)
#             else:
#                 # Hard update
#                 self.target_net.load_state_dict(self.policy_net.state_dict())
        
#         return loss.item()
    
#     def update_model(self, negotiation_result: Dict[str, Any]) -> None:
#         """Update DQN model based on negotiation result"""
#         self.total_episodes += 1
        
#         # Extract emotion sequence
#         emotion_sequence = negotiation_result.get('emotion_sequence', [])
#         if not emotion_sequence:
#             print("⚠️ No emotion sequence found")
#             return
        
#         # Calculate total reward for episode
#         episode_reward = self.calculate_reward(negotiation_result, emotion_sequence)
#         self.episode_rewards.append(episode_reward)
        
#         # Update best sequence
#         if episode_reward > self.best_reward:
#             self.best_reward = episode_reward
#             self.best_sequence = emotion_sequence.copy()
        
#         # For sparse rewards (only final reward), distribute it across steps
#         # This is crucial for DQN learning!
#         if len(emotion_sequence) >= 2:
#             # Distribute final reward across all steps
#             for i in range(len(emotion_sequence) - 1):
#                 # Calculate step reward (portion of total)
#                 step_reward = episode_reward / (len(emotion_sequence) - 1)
                
#                 # Simulate storing transition for each step
#                 # In real implementation, these should have been stored during negotiation
#                 if i < len(emotion_sequence) - 1:
#                     from_emotion = emotion_sequence[i]
#                     to_emotion = emotion_sequence[i + 1]
                    
#                     # Create fake context for this step
#                     step_context = self._extract_context_features({
#                         'round': i + 1,
#                         'dialog_length': i + 1,
#                         'max_dialog_len': len(emotion_sequence)
#                     })
                    
#                     # Store this transition
#                     # Note: This is a workaround - ideally transitions are stored in real-time
#                     self._store_synthetic_transition(
#                         from_emotion, to_emotion, step_reward, step_context,
#                         done=(i == len(emotion_sequence) - 2)
#                     )
        
#         # Train on accumulated experiences
#         training_losses = []
#         for _ in range(min(10, len(self.replay_buffer) // max(1, self.batch_size // 4))):
#             loss = self.train_step()
#             if loss is not None:
#                 training_losses.append(loss)
        
#         # Decay exploration rate
#         self.exploration_rate = max(
#             self.min_exploration,
#             self.exploration_rate * self.exploration_decay
#         )
        
#         print(f"✅ Episode {self.total_episodes}: reward={episode_reward:.2f}, "
#             f"buffer={len(self.replay_buffer)}, steps={self.total_steps}")
        
#         # Reset for next episode
#         self.current_emotion_idx = EMOTION_TO_IDX['neutral']
#         self.emotion_history = []
        
#         # Clear last state/action
#         if hasattr(self, 'last_state'):
#             delattr(self, 'last_state')
#         if hasattr(self, 'last_action'):
#             delattr(self, 'last_action')
    
#     def get_transition_matrix(self) -> np.ndarray:
#         """Extract transition matrix from learned Q-values"""
#         transition_matrix = np.zeros((N_EMOTIONS, N_EMOTIONS))
        
#         # Average context (neutral context)
#         avg_context = np.zeros(self.context_dim)
#         avg_context[0] = 0.5  # Mid-negotiation
#         avg_context[1] = 0.5  # Mid-progress
        
#         for i in range(N_EMOTIONS):
#             state_onehot = np.zeros(N_EMOTIONS)
#             state_onehot[i] = 1.0
            
#             state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
#             context_tensor = torch.FloatTensor(avg_context).unsqueeze(0).to(self.device)
            
#             with torch.no_grad():
#                 q_values = self.policy_net(state_tensor, context_tensor, self.use_dueling)
#                 probabilities = F.softmax(q_values / 0.1, dim=-1)  # Softmax with temperature
            
#             transition_matrix[i, :] = probabilities.cpu().numpy()[0]
        
#         return transition_matrix
    
#     def get_stats(self) -> Dict[str, Any]:
#         """Get model statistics"""
#         stats = {
#             'total_episodes': self.total_episodes,
#             'total_steps': self.total_steps,
#             'best_reward': float(self.best_reward),
#             'exploration_rate': float(self.exploration_rate),
#             'replay_buffer_size': len(self.replay_buffer),
#             'current_emotion': EMOTIONS[self.current_emotion_idx],
#             'emotion_history': self.emotion_history[-10:],
#             'using_double_dqn': self.use_double_dqn,
#             'using_dueling': self.use_dueling,
#             'using_per': self.use_per,
#         }
        
#         # Add recent performance
#         if self.episode_rewards:
#             window = min(10, len(self.episode_rewards))
#             recent_rewards = self.episode_rewards[-window:]
#             stats.update({
#                 'avg_reward_last_10': float(np.mean(recent_rewards)),
#                 'std_reward_last_10': float(np.std(recent_rewards)),
#                 'max_reward_last_10': float(np.max(recent_rewards)),
#             })
        
#         # Network statistics
#         if self.training_losses:
#             window = min(10, len(self.training_losses))
#             recent_losses = self.training_losses[-window:]
#             stats.update({
#                 'avg_loss_last_10': float(np.mean(recent_losses)),
#                 'std_loss_last_10': float(np.std(recent_losses)),
#             })
        
#         return stats
    
#     def reset(self) -> None:
#         """Reset model state for new negotiation (keep learned network)"""
#         self.current_emotion_idx = EMOTION_TO_IDX['neutral']
#         self.emotion_history = []
#         self.context_features = np.zeros(self.context_dim)
        
#         # Clear last state/action
#         if hasattr(self, 'last_state'):
#             delattr(self, 'last_state')
#         if hasattr(self, 'last_action'):
#             delattr(self, 'last_action')
    
#     def save_model(self, filepath: str) -> None:
#         """Save DQN model to file"""
#         model_data = {
#             'policy_net_state_dict': self.policy_net.state_dict(),
#             'target_net_state_dict': self.target_net.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'exploration_rate': self.exploration_rate,
#             'total_episodes': self.total_episodes,
#             'total_steps': self.total_steps,
#             'best_reward': self.best_reward,
#             'best_sequence': self.best_sequence,
#             'episode_rewards': self.episode_rewards,
#             'training_losses': self.training_losses,
#             'config': {
#                 'learning_rate': self.learning_rate,
#                 'discount_factor': self.discount_factor,
#                 'exploration_decay': self.exploration_decay,
#                 'min_exploration': self.min_exploration,
#                 'use_double_dqn': self.use_double_dqn,
#                 'use_dueling': self.use_dueling,
#                 'use_per': self.use_per,
#                 'tau': self.tau,
#             }
#         }
        
#         torch.save(model_data, filepath)
    
#     def load_model(self, filepath: str) -> None:
#         """Load DQN model from file"""
#         if not os.path.exists(filepath):
#             print(f"Warning: Model file {filepath} not found")
#             return
        
#         model_data = torch.load(filepath, map_location=self.device)
        
#         self.policy_net.load_state_dict(model_data['policy_net_state_dict'])
#         self.target_net.load_state_dict(model_data['target_net_state_dict'])
#         self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
#         self.exploration_rate = model_data['exploration_rate']
#         self.total_episodes = model_data['total_episodes']
#         self.total_steps = model_data['total_steps']
#         self.best_reward = model_data['best_reward']
#         self.best_sequence = model_data['best_sequence']
#         self.episode_rewards = model_data['episode_rewards']
#         self.training_losses = model_data['training_losses']

"""
Model 3: DQN (Deep Q-Network) Baseline - Simplified Version
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from collections import deque, namedtuple
from baselines.base_model import BaseEmotionModel

# Emotion definitions
EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# Experience replay memory
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'context'])

class StateEncoder(nn.Module):
    """Neural network for Q-value approximation"""
    def __init__(self, state_dim: int = 7, context_dim: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        
        # State embedding (one-hot emotion)
        self.state_embedding = nn.Linear(state_dim, 16)
        
        # Context embedding (5 features)
        self.context_embedding = nn.Linear(context_dim, 16)
        
        # Combined features
        self.combined = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Dueling architecture: V(s) + A(s,a)
        self.value_head = nn.Linear(32, 1)
        self.advantage_head = nn.Linear(32, state_dim)
        
    def forward(self, state_onehot: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Encode state and context
        state_encoded = F.relu(self.state_embedding(state_onehot))
        context_encoded = F.relu(self.context_embedding(context))
        
        # Combine
        combined = torch.cat([state_encoded, context_encoded], dim=-1)
        features = self.combined(combined)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            
        return q_values

class DQNBaseline(BaseEmotionModel):
    """
    Simplified DQN for emotional transition optimization
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.05,
        replay_buffer_size: int = 5000,
        batch_size: int = 32,
        use_double_dqn: bool = True,
        use_dueling: bool = True,
        use_per: bool = True,
        tau: float = 0.01,
        device: str = "cpu"
    ):
        # DQN parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.tau = tau
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Context dimensions: 5 features as specified
        self.context_dim = 5
        
        # Networks
        self.policy_net = StateEncoder(N_EMOTIONS, self.context_dim).to(self.device)
        self.target_net = StateEncoder(N_EMOTIONS, self.context_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Simple replay buffer (no PER for simplicity)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Current state
        self.current_emotion_idx = EMOTION_TO_IDX['neutral']
        self.emotion_history = []
        
        # Training tracking
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = []
        
        # Best performance
        self.best_reward = -float('inf')
        self.best_sequence = None
        
        # Context features
        self.context_features = np.zeros(self.context_dim)
        
    def _extract_context_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Extract 5 context features as specified"""
        dialog_length = state.get('dialog_length', 0)
        max_dialog_len = state.get('max_dialog_len', 30)
        
        features = np.zeros(self.context_dim)
        
        # Feature 1: Dialog progress
        features[0] = dialog_length / max_dialog_len
        
        # Feature 2: Debt ratio (0-1)
        features[1] = state.get('debt_ratio', 0.5)
        
        # Features 3-5: Last 3 emotions normalized
        if len(self.emotion_history) >= 1:
            last_idx = EMOTION_TO_IDX.get(self.emotion_history[-1], 0)
            features[2] = last_idx / (N_EMOTIONS - 1)
        
        if len(self.emotion_history) >= 2:
            second_idx = EMOTION_TO_IDX.get(self.emotion_history[-2], 0)
            features[3] = second_idx / (N_EMOTIONS - 1)
        
        if len(self.emotion_history) >= 3:
            third_idx = EMOTION_TO_IDX.get(self.emotion_history[-3], 0)
            features[4] = third_idx / (N_EMOTIONS - 1)
        
        return features
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion using DQN policy with epsilon-greedy"""
        # Update context features
        self.context_features = self._extract_context_features(state)
        
        # Prepare state tensor (one-hot emotion)
        state_onehot = np.zeros(N_EMOTIONS)
        state_onehot[self.current_emotion_idx] = 1.0
        
        state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
        context_tensor = torch.FloatTensor(self.context_features).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            # Explore: random action
            next_idx = random.randint(0, N_EMOTIONS - 1)
            q_values = None
        else:
            # Exploit: best action according to policy network
            with torch.no_grad():
                q_values = self.policy_net(state_tensor, context_tensor)
                next_idx = q_values.argmax(dim=1).item()
        
        next_emotion = EMOTIONS[next_idx]
        
        # Store for experience replay
        self.last_state = {
            'state_onehot': state_onehot.copy(),
            'context': self.context_features.copy()
        }
        self.last_action = next_idx
        
        # Update state
        self.current_emotion_idx = next_idx
        self.emotion_history.append(next_emotion)
        
        # Calculate confidence
        confidence = 0.7
        if q_values is not None:
            max_q = q_values.max().item()
            selected_q = q_values[0, next_idx].item()
            confidence = min(1.0, selected_q / (max_q + 1e-8))
        
        # Emotion prompts
        emotion_prompts = {
            "happy": "Use an optimistic and positive tone",
            "surprising": "Use an engaging and unexpected approach",
            "angry": "Use a firm and assertive tone",
            "sad": "Use an empathetic tone",
            "disgust": "Use a disappointed tone",
            "fear": "Use a cautious tone",
            "neutral": "Use a balanced tone"
        }
        
        return {
            "emotion": next_emotion,
            "emotion_text": emotion_prompts.get(next_emotion, "Use a professional tone"),
            "confidence": float(confidence),
            "exploration_rate": float(self.exploration_rate)
        }
    
    def calculate_reward(self, negotiation_result: Dict[str, Any], 
                        emotion_sequence: List[str]) -> float:
        """Calculate reward using simplified formula: R = R_base + R_time + R_rounds"""
        success = negotiation_result.get('final_state') == 'accept'
        collection_days = negotiation_result.get('collection_days', 0)
        target_days = negotiation_result.get('creditor_target_days', 30)
        negotiation_rounds = negotiation_result.get('negotiation_rounds', 1)
        
        if not success:
            return -5.0  # Penalty for failure
        
        # Base reward for success
        R_base = 10.0
        
        # Time efficiency reward
        if target_days > 0:
            time_efficiency = max(0, (target_days - collection_days) / target_days)
            R_time = 20.0 * time_efficiency
        else:
            R_time = 0.0
        
        # Round efficiency reward
        R_rounds = 15.0 / (1 + np.log(negotiation_rounds + 1))
        
        # Total reward (additive as requested)
        total_reward = R_base + R_time + R_rounds
        
        return total_reward
    
    def store_transition(self, next_context: np.ndarray, reward: float, done: bool):
        """Store transition in replay buffer"""
        if not hasattr(self, 'last_state') or not hasattr(self, 'last_action'):
            return
        
        # Get next state one-hot
        next_state_onehot = np.zeros(N_EMOTIONS)
        next_state_onehot[self.current_emotion_idx] = 1.0
        
        experience = Experience(
            state=self.last_state['state_onehot'],
            action=self.last_action,
            reward=reward,
            next_state=next_state_onehot,
            context=self.last_state['context'],
            done=done
        )
        
        # Add to replay buffer
        self.replay_buffer.append(experience)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare tensors
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(self.device)
        actions = torch.LongTensor(np.array([exp.action for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp.reward for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(self.device)
        contexts = torch.FloatTensor(np.array([exp.context for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp.done for exp in batch])).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states, contexts)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values with Double DQN
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_states, contexts).argmax(1)
                next_q_values = self.target_net(next_states, contexts)
                next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_net(next_states, contexts)
                next_q = next_q_values.max(1)[0]
            
            # Target Q values
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                             self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + 
                                   (1 - self.tau) * target_param.data)
        
        self.total_steps += 1
        return loss.item()
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Update DQN model based on negotiation result"""
        self.total_episodes += 1
        
        # Extract emotion sequence
        emotion_sequence = negotiation_result.get('emotion_sequence', [])
        if not emotion_sequence:
            return
        
        # Calculate total reward
        episode_reward = self.calculate_reward(negotiation_result, emotion_sequence)
        self.episode_rewards.append(episode_reward)
        
        # Update best sequence
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_sequence = emotion_sequence.copy()
        
        # Distribute reward across steps and store experiences
        if len(emotion_sequence) >= 2:
            # Equal distribution of total reward across transitions
            step_reward = episode_reward / (len(emotion_sequence) - 1)
            
            # Store each transition with appropriate reward
            for i in range(len(emotion_sequence) - 1):
                done = (i == len(emotion_sequence) - 2)  # Last transition
                
                # Context for this step
                step_context = self._extract_context_features({
                    'dialog_length': i + 1,
                    'max_dialog_len': len(emotion_sequence),
                    'debt_ratio': negotiation_result.get('debt_ratio', 0.5)
                })
                
                # Store transition
                self.store_transition(step_context, step_reward, done)
        
        # Train on multiple batches
        training_losses = []
        for _ in range(min(5, len(self.replay_buffer) // max(1, self.batch_size // 4))):
            loss = self.train_step()
            if loss is not None:
                training_losses.append(loss)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # Reset for next episode
        self.current_emotion_idx = EMOTION_TO_IDX['neutral']
        self.emotion_history = []
        
        # Clear last state/action
        if hasattr(self, 'last_state'):
            delattr(self, 'last_state')
        if hasattr(self, 'last_action'):
            delattr(self, 'last_action')
    
    def get_transition_matrix(self) -> np.ndarray:
        """Extract transition matrix from learned Q-values"""
        transition_matrix = np.zeros((N_EMOTIONS, N_EMOTIONS))
        
        # Neutral context (mid-negotiation)
        avg_context = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        for i in range(N_EMOTIONS):
            state_onehot = np.zeros(N_EMOTIONS)
            state_onehot[i] = 1.0
            
            state_tensor = torch.FloatTensor(state_onehot).unsqueeze(0).to(self.device)
            context_tensor = torch.FloatTensor(avg_context).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(state_tensor, context_tensor)
                # Softmax with temperature τ=0.1
                probabilities = F.softmax(q_values / 0.1, dim=-1)
            
            transition_matrix[i, :] = probabilities.cpu().numpy()[0]
        
        return transition_matrix
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'best_reward': float(self.best_reward),
            'exploration_rate': float(self.exploration_rate),
            'replay_buffer_size': len(self.replay_buffer),
            'current_emotion': EMOTIONS[self.current_emotion_idx]
        }
        
        # Recent performance
        if self.episode_rewards:
            window = min(10, len(self.episode_rewards))
            recent_rewards = self.episode_rewards[-window:]
            stats.update({
                'avg_reward_last_10': float(np.mean(recent_rewards)),
                'std_reward_last_10': float(np.std(recent_rewards))
            })
        
        return stats
    
    def reset(self) -> None:
        """Reset model state for new negotiation"""
        self.current_emotion_idx = EMOTION_TO_IDX['neutral']
        self.emotion_history = []
        self.context_features = np.zeros(self.context_dim)
        
        if hasattr(self, 'last_state'):
            delattr(self, 'last_state')
        if hasattr(self, 'last_action'):
            delattr(self, 'last_action')
    
    def save_model(self, filepath: str) -> None:
        """Save the trained DQN model"""
        model_state = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'min_exploration': self.min_exploration,
                'batch_size': self.batch_size,
                'use_double_dqn': self.use_double_dqn,
                'tau': self.tau,
                'context_dim': self.context_dim
            },
            'training_stats': {
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
                'best_reward': self.best_reward,
                'best_sequence': self.best_sequence
            }
        }
        torch.save(model_state, filepath)


def run_dqn_experiment(
    scenarios: List[Dict[str, Any]],
    episodes: int = 200,
    episodes_per_scenario: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results",
    use_double_dqn: bool = True,
    use_dueling: bool = True,
    use_per: bool = True,
    learning_rate: float = 1e-4,
    discount_factor: float = 0.95,
    exploration_rate: float = 1.0,
    exploration_decay: float = 0.995,
    batch_size: int = 32,
    replay_buffer_size: int = 10000,
    target_update_freq: int = 100,
    tau: float = 0.01,
    min_exploration: float = 0.05
) -> Dict[str, Any]:
    """Run DQN experiment for emotional transition optimization"""
    
    from llm.negotiator import DebtNegotiator
    
    # Create DQN model
    dqn_model = DQNBaseline(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        min_exploration=min_exploration,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        use_double_dqn=use_double_dqn,
        use_dueling=use_dueling,
        use_per=use_per,
        tau=tau
    )
    
    results = {
        'experiment_type': 'dqn_baseline',
        'total_episodes': episodes,
        'episodes_per_scenario': episodes_per_scenario,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'exploration_rate': exploration_rate,
        'exploration_decay': exploration_decay,
        'min_exploration': min_exploration,
        'batch_size': batch_size,
        'replay_buffer_size': replay_buffer_size,
        'target_update_freq': target_update_freq,
        'tau': tau,
        'use_double_dqn': use_double_dqn,
        'use_dueling': use_dueling,
        'use_per': use_per,
        'scenarios_used': [s['id'] for s in scenarios],
        'episode_results': {},
        'learning_curve': []
    }
    
    # Training loop
    for episode in range(episodes):
        # Cycle through scenarios
        scenario_idx = episode % len(scenarios)
        scenario = scenarios[scenario_idx]
        
        print(f"\n🧠 Episode {episode + 1}/{episodes}")
        print(f"   Scenario: {scenario['id']}")
        print(f"   Exploration: {dqn_model.exploration_rate:.3f}")
        print(f"   Buffer size: {len(dqn_model.replay_buffer)}")
        
        # Create negotiator
        negotiator = DebtNegotiator(
            config=scenario,
            emotion_model=dqn_model,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion
        )
        
        # Run negotiation
        result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
        
        # Update DQN model
        dqn_model.update_model(result)
        
        # Extract results
        final_state = result.get('final_state', 'breakdown')
        final_days = result.get('final_days', 'N/A')
        rounds = len(result.get('dialog', []))
        emotion_seq = result.get('emotion_sequence', [])
        success = final_state == 'accept'
        
        # Calculate episode reward
        episode_reward = dqn_model.calculate_reward(result, emotion_seq)
        
        # Store episode results with negotiation_result for statistical analysis
        episode_key = f'episode_{episode+1}'
        results['episode_results'][episode_key] = {
            'scenario': scenario['id'],
            'success': success,
            'final_days': final_days,
            'rounds': rounds,
            'emotion_sequence': emotion_seq,
            'reward': episode_reward,
            'exploration_rate': dqn_model.exploration_rate,
            'stats': dqn_model.get_stats(),
            'negotiation_result': result  # Add full negotiation result for statistical analysis
        }
        
        # Store learning curve data
        results['learning_curve'].append({
            'episode': episode + 1,
            'reward': episode_reward,
            'success': success,
            'exploration_rate': dqn_model.exploration_rate,
            'buffer_size': len(dqn_model.replay_buffer),
            'training_steps': dqn_model.total_steps,
        })
        
        # Print progress
        outcome_emoji = "✅" if success else "❌"
        print(f"   Result: {outcome_emoji} {final_state} | Days: {final_days} | Rounds: {rounds}")
        print(f"   Reward: {episode_reward:.2f} | Emotions: {emotion_seq}")
        
        # Periodic evaluation
        if (episode + 1) % 10 == 0:
            stats = dqn_model.get_stats()
            print(f"\n📊 Evaluation after {episode + 1} episodes:")
            print(f"   Avg reward (last 10): {stats.get('avg_reward_last_10', 0):.2f}")
            print(f"   Best reward: {stats.get('best_reward', 0):.2f}")
            print(f"   Exploration: {stats.get('exploration_rate', 0):.3f}")
            print(f"   Buffer: {stats.get('replay_buffer_size', 0)} experiences")
            print(f"   Training steps: {stats.get('total_steps', 0)}")
    
    # Collect all negotiation results for statistical analysis
    all_negotiation_results = [ep_result['negotiation_result'] for ep_result in results['episode_results'].values() if 'negotiation_result' in ep_result]
    
    # Final results
    results['final_stats'] = dqn_model.get_stats()
    results['final_transition_matrix'] = dqn_model.get_transition_matrix().tolist()
    results['best_sequence'] = dqn_model.best_sequence
    results['best_reward'] = dqn_model.best_reward
    
    # Calculate final performance metrics like vanilla model
    successful_episodes = [r for r in results['episode_results'].values() if r['success']]
    failed_episodes = [r for r in results['episode_results'].values() if not r['success']]
    
    if all_negotiation_results:
        overall_success_rate = len(successful_episodes) / episodes
        
        # Calculate collection rates for successful negotiations
        collection_rates = []
        for r in successful_episodes:
            if 'negotiation_result' in r:
                neg_result = r['negotiation_result']
                target_days = neg_result.get('creditor_target_days', 30)
                final_days = neg_result.get('final_days', target_days)
                if target_days > 0:
                    collection_rate = min(final_days, target_days) / target_days
                    collection_rates.append(collection_rate)
        
        avg_collection_rate = np.mean(collection_rates) if collection_rates else 0.0
        avg_rounds = np.mean([r['rounds'] for r in successful_episodes]) if successful_episodes else 0.0
    else:
        overall_success_rate = len(successful_episodes) / episodes if episodes > 0 else 0.0
        avg_collection_rate = 0.0
        avg_rounds = 0.0
    
    # Final results
    results['final_stats'] = dqn_model.get_stats()
    results['final_transition_matrix'] = dqn_model.get_transition_matrix().tolist()
    results['best_sequence'] = dqn_model.best_sequence
    results['best_reward'] = dqn_model.best_reward
    results['overall_success_rate'] = overall_success_rate
    
    # Add performance breakdown like vanilla model
    failure_reasons = {}
    for result in failed_episodes:
        if 'negotiation_result' in result:
            reason = result['negotiation_result'].get('final_state', 'unknown')
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    results['performance'] = {
        'success_rate': overall_success_rate,
        'avg_collection_rate': float(avg_collection_rate),
        'avg_negotiation_rounds': float(avg_rounds),
        'total_episodes': episodes,
        'successful_episodes': len(successful_episodes),
        'failed_episodes': len(failed_episodes)
    }
    
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([r['rounds'] for r in successful_episodes])) if successful_episodes else 0,
            'avg_rounds_failed': float(np.mean([r['rounds'] for r in failed_episodes])) if failed_episodes else 0
        }
    }
    
    # Calculate moving averages
    rewards = [ep['reward'] for ep in results['learning_curve']]
    window = min(20, len(rewards))
    if window > 0:
        results['final_avg_reward'] = np.mean(rewards[-window:])
        results['final_reward_std'] = np.std(rewards[-window:])
    
    # ===== ADD STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS =====
    if all_negotiation_results:  # Only if we have negotiation results
        results = enhance_results_with_statistics(
            results, 
            all_negotiation_results, 
            scenarios, 
            method="bootstrap"
        )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/dqn_baseline_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save learned model
    model_file = f"{out_dir}/dqn_model_{timestamp}.pt"
    dqn_model.save_model(model_file)
    
    print(f"\n💾 Results saved to: {result_file}")
    print(f"💾 Model saved to: {model_file}")
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS =====
    print("\n" + "="*80)
    print("📊 DQN RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    # Print statistical analysis with CIs
    if 'statistical_analysis' in results:
        format_ci_results(results['statistical_analysis'])
    
    # Save human-readable summary like vanilla model
    summary_file = f"{out_dir}/dqn_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("DQN BASELINE MODEL RESULTS SUMMARY\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Experiment Type: {results['experiment_type']}\\n")
        f.write(f"Total Episodes: {episodes}\\n")
        f.write(f"Episodes per Scenario: {episodes_per_scenario}\\n")
        f.write(f"Learning Rate: {learning_rate}\\n")
        f.write(f"Discount Factor: {discount_factor}\\n")
        f.write(f"Use Double DQN: {use_double_dqn}\\n\\n")
        
        f.write("PERFORMANCE METRICS:\\n")
        f.write("-" * 20 + "\\n")
        f.write(f"Overall Success Rate: {results['performance']['success_rate']:.1%}\\n")
        f.write(f"Average Collection Rate: {results['performance']['avg_collection_rate']:.3f}\\n")
        f.write(f"Average Rounds per Negotiation: {results['performance']['avg_negotiation_rounds']:.1f}\\n")
        f.write(f"Best Reward: {results['best_reward']:.2f}\\n")
        f.write(f"Final Average Reward: {results.get('final_avg_reward', 0):.2f}\\n\\n")
        
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            f.write("95% CONFIDENCE INTERVALS:\\n")
            f.write("-" * 25 + "\\n")
            sr_ci = stat_analysis['success_rate']['ci_95']
            f.write(f"Success Rate: {stat_analysis['success_rate']['mean']:.1%} [{sr_ci[0]:.1%}, {sr_ci[1]:.1%}]\\n")
            
            if 'collection_rate' in stat_analysis:
                cr_ci = stat_analysis['collection_rate']['ci_95']
                f.write(f"Collection Rate: {stat_analysis['collection_rate']['mean']:.3f} [{cr_ci[0]:.3f}, {cr_ci[1]:.3f}]\\n")
            
            if 'negotiation_rounds' in stat_analysis:
                nr_ci = stat_analysis['negotiation_rounds']['ci_95']
                f.write(f"Negotiation Rounds: {stat_analysis['negotiation_rounds']['mean']:.1f} [{nr_ci[0]:.1f}, {nr_ci[1]:.1f}]\\n")
        
        f.write(f"\\nBest Learned Sequence: {results['best_sequence']}\\n")
        
        if results['analysis']['failure_breakdown']:
            f.write(f"\\nFailure Reasons:\\n")
            for reason, count in results['analysis']['failure_breakdown'].items():
                f.write(f"  {reason}: {count} episodes\\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    
    print(f"\n📊 DQN BASELINE RESULTS")
    print("=" * 50)
    print(f"✅ Success Rate: {results['performance']['success_rate']:.1%}")
    print(f"📈 Avg Collection Rate: {results['performance']['avg_collection_rate']:.3f}")
    print(f"💬 Avg Negotiation Rounds: {results['performance']['avg_negotiation_rounds']:.1f}")
    print(f"🏆 Best Reward: {results['best_reward']:.2f}")
    print(f"📊 Final Avg Reward: {results.get('final_avg_reward', 0):.2f}")
    print(f"🧠 Total Training Steps: {results['final_stats'].get('total_steps', 0)}")
    print(f"🎯 Best Sequence: {results['best_sequence']}")
    
    return results



