"""
Model 2: Q-Learning Baseline for Emotional Transitions
Updated with simplified reward function
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from baselines.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

# Emotion definitions
EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

@dataclass
class QLearningPolicy:
    """Q-learning policy with Q-table and learning parameters"""
    q_table: np.ndarray  # 7x7 Q-values for state-action pairs (current_emotion -> next_emotion)
    learning_rate: float
    discount_factor: float
    exploration_rate: float
    exploration_decay: float
    min_exploration: float
    
    def __post_init__(self):
        self.exploration_rate = max(self.min_exploration, self.exploration_rate)

class QLearningBaseline(BaseEmotionModel):
    """
    Q-Learning Baseline for emotional transition optimization
    Uses reinforcement learning to learn optimal emotion sequences
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.1,
        window_size: int = 5,  # Size of window for sequence-based rewards
        use_softmax: bool = True,  # Use softmax or epsilon-greedy
        temperature: float = 1.0,  # For softmax exploration
    ):
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration = exploration_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.window_size = window_size
        self.use_softmax = use_softmax
        self.temperature = temperature
        
        # Initialize Q-table (7 emotions x 7 possible next emotions)
        # Each row: current emotion, each column: next emotion Q-value
        self.q_table = np.random.uniform(low=-0.1, high=0.1, size=(N_EMOTIONS, N_EMOTIONS))
        
        # Add baseline optimism for neutral start
        neutral_idx = EMOTION_TO_IDX['neutral']
        self.q_table[neutral_idx, :] += 0.1  # Slight optimism for neutral start
        
        # Normalize to ensure variety
        for i in range(N_EMOTIONS):
            row_sum = np.sum(np.abs(self.q_table[i, :]))
            if row_sum > 0:
                self.q_table[i, :] /= row_sum * 10
        
        # Current state
        self.current_emotion_idx = neutral_idx
        self.emotion_history = []
        self.action_history = []  # Store (state, action, reward) for sequence learning
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_sequences = []
        self.total_episodes = 0
        self.best_reward = -float('inf')
        self.best_sequence = None
        
        # Learning statistics
        self.q_table_history = []
        self.reward_history = []
        
        # Softmax normalization
        self.softmax_beta = 1.0 / temperature
        
        # Policy for selection
        self.current_policy = QLearningPolicy(
            q_table=self.q_table.copy(),
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration=min_exploration
        )
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion using Q-learning policy"""
        # Get context for adaptive exploration
        round_num = state.get('round', 1)
        dialog_length = state.get('dialog_length', 0)
        
        # Adaptive exploration based on progress
        if dialog_length > 0:
            adaptive_exploration = max(
                self.min_exploration,
                self.exploration_rate * (1.0 - min(1.0, dialog_length / 20))
            )
        else:
            adaptive_exploration = self.exploration_rate
        
        # Select action using epsilon-greedy or softmax
        if self.use_softmax:
            next_idx = self._softmax_action_selection(self.current_emotion_idx, adaptive_exploration)
        else:
            next_idx = self._epsilon_greedy_action_selection(self.current_emotion_idx, adaptive_exploration)
        
        next_emotion = EMOTIONS[next_idx]
        
        # Store action for learning
        action_record = {
            'state': self.current_emotion_idx,
            'action': next_idx,
            'round': round_num,
            'exploration_rate': adaptive_exploration,
            'q_values': self.q_table[self.current_emotion_idx, :].copy()
        }
        self.action_history.append(action_record)
        
        # Update state
        prev_emotion_idx = self.current_emotion_idx
        self.current_emotion_idx = next_idx
        self.emotion_history.append(next_emotion)
        
        # Calculate action confidence from Q-values
        q_values = self.q_table[prev_emotion_idx, :]
        max_q = np.max(q_values)
        selected_q = q_values[next_idx]
        confidence = selected_q / max_q if max_q > 0 else 0.5
        
        # Get emotion prompt
        emotion_prompts = {
            "happy": "Use an optimistic and positive tone, expressing confidence",
            "surprising": "Use an engaging and unexpected approach",
            "angry": "Use a firm and assertive tone, emphasizing urgency",
            "sad": "Use an empathetic and understanding tone",
            "disgust": "Use a disappointed tone while remaining professional",
            "fear": "Use a cautious and concerned tone",
            "neutral": "Use a balanced and professional tone"
        }
        
        return {
            "emotion": next_emotion,
            "emotion_text": emotion_prompts.get(next_emotion, "Use a professional tone"),
            "confidence": float(confidence),
            "exploration_rate": float(adaptive_exploration),
            "q_value": float(selected_q),
            "action_method": "softmax" if self.use_softmax else "epsilon_greedy",
            "temperature": self.temperature if self.use_softmax else None
        }
    
    def _epsilon_greedy_action_selection(self, state: int, epsilon: float) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, N_EMOTIONS - 1)
        else:
            # Exploit: best action according to Q-values
            q_values = self.q_table[state, :]
            # Add small random noise to break ties
            q_values = q_values + np.random.normal(0, 1e-6, size=q_values.shape)
            return np.argmax(q_values)
    
    def _softmax_action_selection(self, state: int, temperature: float) -> int:
        """Softmax action selection based on Q-values"""
        q_values = self.q_table[state, :]
        
        # Apply temperature scaling
        scaled_q = q_values / (temperature + 1e-8)
        
        # Subtract max for numerical stability
        scaled_q = scaled_q - np.max(scaled_q)
        
        # Compute softmax probabilities
        exp_q = np.exp(scaled_q)
        probabilities = exp_q / np.sum(exp_q)
        
        # Sample action
        return np.random.choice(N_EMOTIONS, p=probabilities)
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Update Q-value using standard Q-learning update rule"""
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state, :])
        
        # Q-learning update
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error
        
        # Record update for analysis
        update_info = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'td_error': td_error,
            'new_q_value': self.q_table[state, action]
        }
        return update_info
    
    def calculate_reward(self, negotiation_result: Dict[str, Any], 
                        emotion_sequence: List[str]) -> float:
        """
        Calculate reward using simplified formula:
        R_total = R_base + R_time + R_rounds
        where:
          R_base = 10·I[success] - 5·I[failure]
          R_time = 20·max(0, (D_target - D)/D_target)
          R_rounds = 15/(1 + log(R + 1))
        """
        success = negotiation_result.get('final_state') == 'accept'
        collection_days = negotiation_result.get('collection_days', 0)
        target_days = negotiation_result.get('creditor_target_days', 30)
        negotiation_rounds = negotiation_result.get('negotiation_rounds', 1)
        
        # Base reward
        if success:
            R_base = 10.0
        else:
            return -5.0  # Early return for failure
        
        # Time efficiency reward
        if target_days > 0:
            time_efficiency = max(0, (target_days - collection_days) / target_days)
            R_time = 20.0 * time_efficiency
        else:
            R_time = 0.0
        
        # Round efficiency reward
        R_rounds = 15.0 / (1 + np.log(negotiation_rounds + 1))
        
        # Total reward (no length penalty, no strategy bonus)
        total_reward = R_base + R_time + R_rounds
        
        return total_reward
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Update Q-learning model based on negotiation result"""
        self.total_episodes += 1
        
        # Extract emotion sequence
        emotion_sequence = negotiation_result.get('emotion_sequence', [])
        if not emotion_sequence:
            return
        
        # Calculate total reward for episode
        episode_reward = self.calculate_reward(negotiation_result, emotion_sequence)
        self.episode_rewards.append(episode_reward)
        
        # Update best sequence
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_sequence = emotion_sequence.copy()
        
        # Learn from sequence using equal credit assignment (same as DQN)
        if len(emotion_sequence) >= 2:
            # Equal credit assignment: distribute total reward evenly
            self._equal_credit_q_learning(emotion_sequence, episode_reward)
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # Store learning statistics
        self.reward_history.append({
            'episode': self.total_episodes,
            'reward': episode_reward,
            'exploration_rate': self.exploration_rate,
            'sequence_length': len(emotion_sequence),
            'success': negotiation_result.get('final_state') == 'accept'
        })
        
        # Periodically store Q-table snapshot
        if self.total_episodes % 10 == 0:
            self.q_table_history.append({
                'episode': self.total_episodes,
                'q_table': self.q_table.copy(),
                'exploration_rate': self.exploration_rate
            })
        
        # Reset for next episode
        self.current_emotion_idx = EMOTION_TO_IDX['neutral']
        self.emotion_history = []
        self.action_history = []
    
    def _equal_credit_q_learning(self, emotion_sequence: List[str], total_reward: float) -> None:
        """
        Update Q-values with equal credit assignment:
        Each transition gets reward = total_reward / (T-1)
        """
        sequence_length = len(emotion_sequence)
        
        # Calculate equal step reward
        step_reward = total_reward / (sequence_length - 1)
        
        # Update each transition
        for t in range(sequence_length - 1):
            # Convert emotions to indices
            state_idx = EMOTION_TO_IDX[emotion_sequence[t]]
            action_idx = EMOTION_TO_IDX[emotion_sequence[t + 1]]
            
            # Determine next state
            next_state_idx = action_idx if t < sequence_length - 2 else state_idx
            
            # Update Q-value
            self.update_q_value(state_idx, action_idx, step_reward, next_state_idx)
    
    def _discounted_credit_q_learning(self, emotion_sequence: List[str], total_reward: float) -> None:
        """
        Alternative: Update Q-values with discounted credit assignment:
        Each transition gets reward = total_reward * γ^(T-t-2) / T
        """
        sequence_length = len(emotion_sequence)
        
        # Update each transition with discounting
        for t in range(sequence_length - 1):
            # Convert emotions to indices
            state_idx = EMOTION_TO_IDX[emotion_sequence[t]]
            action_idx = EMOTION_TO_IDX[emotion_sequence[t + 1]]
            
            # Calculate step reward with discounting
            discount_factor = self.discount_factor ** (sequence_length - t - 2)
            step_reward = total_reward * discount_factor / sequence_length
            
            # Determine next state
            next_state_idx = action_idx if t < sequence_length - 2 else state_idx
            
            # Update Q-value
            self.update_q_value(state_idx, action_idx, step_reward, next_state_idx)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Convert Q-table to probability transition matrix using softmax"""
        transition_matrix = np.zeros((N_EMOTIONS, N_EMOTIONS))
        
        for i in range(N_EMOTIONS):
            q_values = self.q_table[i, :]
            
            # Apply softmax to convert Q-values to probabilities
            # Higher temperature = more exploration in probabilities
            exp_q = np.exp(q_values / self.temperature)
            probabilities = exp_q / np.sum(exp_q)
            
            # Store in transition matrix
            transition_matrix[i, :] = probabilities
        
        return transition_matrix
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
            std_reward = np.std(self.episode_rewards[-10:])
        else:
            avg_reward = 0.0
            std_reward = 0.0
        
        # Calculate Q-table statistics
        q_mean = np.mean(self.q_table)
        q_std = np.std(self.q_table)
        q_max = np.max(self.q_table)
        q_min = np.min(self.q_table)
        
        # Get learned transition matrix
        transition_matrix = self.get_transition_matrix()
        
        # Calculate matrix entropy (diversity of choices)
        entropy = 0.0
        for i in range(N_EMOTIONS):
            row = transition_matrix[i]
            row_entropy = -np.sum(row * np.log(row + 1e-10))
            entropy += row_entropy
        avg_entropy = entropy / N_EMOTIONS
        
        return {
            'total_episodes': self.total_episodes,
            'best_reward': float(self.best_reward),
            'avg_reward_last_10': float(avg_reward),
            'reward_std_last_10': float(std_reward),
            'exploration_rate': float(self.exploration_rate),
            'q_table_mean': float(q_mean),
            'q_table_std': float(q_std),
            'q_table_max': float(q_max),
            'q_table_min': float(q_min),
            'transition_entropy': float(avg_entropy),
            'current_emotion': EMOTIONS[self.current_emotion_idx],
            'emotion_history': self.emotion_history[-10:],
            'use_softmax': self.use_softmax,
            'temperature': self.temperature,
            'reward_function': 'simplified'
        }
    
    def reset(self) -> None:
        """Reset model state for new negotiation (keep learned Q-table)"""
        self.current_emotion_idx = EMOTION_TO_IDX['neutral']
        self.emotion_history = []
        self.action_history = []
    
    def save_model(self, filepath: str) -> None:
        """Save Q-learning model to file"""
        model_data = {
            'q_table': self.q_table.tolist(),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration': self.min_exploration,
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'best_sequence': self.best_sequence,
            'episode_rewards': self.episode_rewards,
            'reward_history': self.reward_history,
            'use_softmax': self.use_softmax,
            'temperature': self.temperature,
            'reward_function': 'simplified'
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str) -> None:
        """Load Q-learning model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.q_table = np.array(model_data['q_table'])
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.exploration_rate = model_data['exploration_rate']
        self.exploration_decay = model_data['exploration_decay']
        self.min_exploration = model_data['min_exploration']
        self.total_episodes = model_data['total_episodes']
        self.best_reward = model_data['best_reward']
        self.best_sequence = model_data['best_sequence']
        self.episode_rewards = model_data['episode_rewards']
        self.reward_history = model_data['reward_history']
        self.use_softmax = model_data.get('use_softmax', True)
        self.temperature = model_data.get('temperature', 1.0)

def run_qlearning_experiment(
    scenarios: List[Dict[str, Any]],
    episodes: int = 100,  # Total training episodes
    episodes_per_scenario: int = 5,  # Episodes per scenario before cycling
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results",
    use_softmax: bool = True,
    temperature: float = 1.0,
    learning_rate: float = 0.1,
    discount_factor: float = 0.9,
    exploration_rate: float = 1.0,
    exploration_decay: float = 0.995,
    credit_assignment: str = 'equal'  # 'equal' or 'discounted'
) -> Dict[str, Any]:
    """Run Q-learning experiment for emotional transition optimization"""
    
    from llm.negotiator import DebtNegotiator
    
    # Create Q-learning model
    q_learner = QLearningBaseline(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        exploration_decay=exploration_decay,
        use_softmax=use_softmax,
        temperature=temperature
    )
    
    results = {
        'experiment_type': 'q_learning_baseline',
        'reward_function': 'simplified',
        'credit_assignment': credit_assignment,
        'total_episodes': episodes,
        'episodes_per_scenario': episodes_per_scenario,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'use_softmax': use_softmax,
        'temperature': temperature,
        'scenarios_used': [s['id'] for s in scenarios],
        'episode_results': {},
        'learning_curve': []
    }
    
    # Training loop
    for episode in range(episodes):
        # Cycle through scenarios
        scenario_idx = episode % len(scenarios)
        scenario = scenarios[scenario_idx]
        
        print(f"\n🎯 Episode {episode + 1}/{episodes}")
        print(f"   Scenario: {scenario['id']}")
        print(f"   Exploration rate: {q_learner.exploration_rate:.3f}")
        
        # Create negotiator
        negotiator = DebtNegotiator(
            config=scenario,
            emotion_model=q_learner,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion
        )
        
        # Run negotiation
        result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
        
        # Update Q-learning model (with custom credit assignment if needed)
        if credit_assignment == 'discounted':
            # Manually handle discounted credit assignment
            q_learner.total_episodes += 1
            
            # Extract emotion sequence
            emotion_sequence = result.get('emotion_sequence', [])
            if emotion_sequence:
                # Calculate total reward
                episode_reward = q_learner.calculate_reward(result, emotion_sequence)
                q_learner.episode_rewards.append(episode_reward)
                
                # Update best sequence
                if episode_reward > q_learner.best_reward:
                    q_learner.best_reward = episode_reward
                    q_learner.best_sequence = emotion_sequence.copy()
                
                # Apply discounted credit assignment
                if len(emotion_sequence) >= 2:
                    q_learner._discounted_credit_q_learning(emotion_sequence, episode_reward)
                
                # Store statistics
                q_learner.reward_history.append({
                    'episode': q_learner.total_episodes,
                    'reward': episode_reward,
                    'exploration_rate': q_learner.exploration_rate,
                    'sequence_length': len(emotion_sequence),
                    'success': result.get('final_state') == 'accept'
                })
                
                # Decay exploration
                q_learner.exploration_rate = max(
                    q_learner.min_exploration,
                    q_learner.exploration_rate * q_learner.exploration_decay
                )
        else:
            # Use default equal credit assignment
            q_learner.update_model(result)
        
        # Extract results
        final_state = result.get('final_state', 'breakdown')
        collection_days = result.get('collection_days', 'N/A')
        target_days = result.get('creditor_target_days', 30)
        rounds = len(result.get('dialog', []))
        emotion_seq = result.get('emotion_sequence', [])
        success = final_state == 'accept'
        
        # Calculate episode reward
        episode_reward = q_learner.calculate_reward(result, emotion_seq)
        
        # Calculate reward components for analysis
        if success and target_days > 0:
            time_efficiency = max(0, (target_days - collection_days) / target_days)
            R_time = 20.0 * time_efficiency
            R_rounds = 15.0 / (1 + np.log(rounds + 1))
        else:
            R_time = 0.0
            R_rounds = 0.0
        
        # Store episode results with negotiation_result for statistical analysis
        episode_key = f'episode_{episode+1}'
        results['episode_results'][episode_key] = {
            'scenario': scenario['id'],
            'success': success,
            'final_days': collection_days,
            'target_days': target_days,
            'rounds': rounds,
            'emotion_sequence': emotion_seq,
            'total_reward': episode_reward,
            'reward_components': {
                'R_base': 10.0 if success else -5.0,
                'R_time': float(R_time),
                'R_rounds': float(R_rounds)
            },
            'exploration_rate': q_learner.exploration_rate,
            'stats': q_learner.get_stats(),
            'negotiation_result': result  # Store complete negotiation result
        }
        
        # Store learning curve data
        results['learning_curve'].append({
            'episode': episode + 1,
            'reward': episode_reward,
            'success': success,
            'exploration_rate': q_learner.exploration_rate,
            'q_table_mean': np.mean(q_learner.q_table),
            'q_table_std': np.std(q_learner.q_table)
        })
        
        # Print progress
        outcome_emoji = "✅" if success else "❌"
        print(f"   Result: {outcome_emoji} {final_state} | Days: {collection_days}/{target_days} | Rounds: {rounds}")
        print(f"   Reward: {episode_reward:.2f} (R_base: {10.0 if success else -5.0:.1f}, "
              f"R_time: {R_time:.2f}, R_rounds: {R_rounds:.2f})")
        print(f"   Emotion sequence: {emotion_seq}")
        
        # Periodic evaluation
        if (episode + 1) % 10 == 0:
            stats = q_learner.get_stats()
            print(f"\n📊 Evaluation after {episode + 1} episodes:")
            print(f"   Avg reward (last 10): {stats['avg_reward_last_10']:.2f}")
            print(f"   Best reward: {stats['best_reward']:.2f}")
            print(f"   Exploration rate: {stats['exploration_rate']:.3f}")
            print(f"   Transition entropy: {stats['transition_entropy']:.3f}")
    
    # Collect all negotiation results for statistical analysis
    all_negotiation_results = [ep_result['negotiation_result'] for ep_result in results['episode_results'].values() if 'negotiation_result' in ep_result]
    
    # Final results
    results['final_stats'] = q_learner.get_stats()
    results['final_q_table'] = q_learner.q_table.tolist()
    results['final_transition_matrix'] = q_learner.get_transition_matrix().tolist()
    results['best_sequence'] = q_learner.best_sequence
    results['best_reward'] = q_learner.best_reward
    
    # Calculate overall success rate
    successful_episodes = [r for r in results['episode_results'].values() if r['success']]
    results['overall_success_rate'] = len(successful_episodes) / episodes if episodes > 0 else 0
    
    # Calculate average final days for successful negotiations
    successful_days = []
    for ep in results['episode_results'].values():
        if ep['success'] and isinstance(ep['final_days'], (int, float)):
            successful_days.append(ep['final_days'])
    
    results['avg_successful_days'] = float(np.mean(successful_days)) if successful_days else 0
    results['avg_successful_rounds'] = float(np.mean([ep['rounds'] for ep in results['episode_results'].values() if ep['success']])) if successful_episodes else 0
    
    # Calculate moving averages
    rewards = [ep['total_reward'] for ep in results['episode_results'].values()]
    window = min(10, len(rewards))
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
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/qlearning_simplified_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save learned model
    model_file = f"{out_dir}/qlearning_simplified_model_{timestamp}.json"
    q_learner.save_model(model_file)
    
    # Save comprehensive summary file (matching vanilla model format)
    summary_file = f"{out_dir}/qlearning_simplified_summary_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Q-LEARNING BASELINE EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Episodes: {episodes}\n")
        f.write(f"Episodes per Scenario: {episodes_per_scenario}\n")
        f.write(f"Scenarios Used: {len(scenarios)}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Discount Factor: {discount_factor}\n")
        f.write(f"Use Softmax: {use_softmax}\n")
        f.write(f"Temperature: {temperature}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Success Rate: {results['overall_success_rate']:.1%}\n")
        f.write(f"Average Days (Successful): {results['avg_successful_days']:.1f}\n")
        f.write(f"Average Rounds (Successful): {results['avg_successful_rounds']:.1f}\n")
        f.write(f"Best Reward: {q_learner.best_reward:.2f}\n")
        f.write(f"Final Avg Reward: {results.get('final_avg_reward', 0):.2f}\n\n")
        
        f.write("BEST EMOTIONAL SEQUENCE:\n")
        f.write("-" * 30 + "\n")
        if q_learner.best_sequence:
            f.write(f"Sequence: {q_learner.best_sequence}\n")
            f.write(f"Length: {len(q_learner.best_sequence)}\n")
            f.write(f"Reward: {q_learner.best_reward:.2f}\n\n")
        else:
            f.write("No best sequence found\n\n")
        
        f.write("FINAL Q-LEARNING STATISTICS:\n")
        f.write("-" * 30 + "\n")
        stats = q_learner.get_stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, list) and len(value) <= 10:
                f.write(f"{key}: {value}\n")
        
        f.write("\nTRANSITION MATRIX SUMMARY:\n")
        f.write("-" * 30 + "\n")
        trans_matrix = q_learner.get_transition_matrix()
        for i, emotion in enumerate(['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']):
            best_next_idx = np.argmax(trans_matrix[i])
            best_next = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral'][best_next_idx]
            prob = trans_matrix[i, best_next_idx]
            f.write(f"{emotion:10s} -> {best_next:10s} (p={prob:.3f})\n")
        
        if 'statistical_analysis' in results:
            f.write("\n95% CONFIDENCE INTERVALS:\n")
            f.write("-" * 30 + "\n")
            for metric, data in results['statistical_analysis'].items():
                if isinstance(data, dict) and 'mean' in data and 'ci' in data:
                    f.write(f"{metric}: {data['mean']:.3f} [{data['ci'][0]:.3f}, {data['ci'][1]:.3f}]\n")
    
    print(f"\n💾 Results saved to: {result_file}")
    print(f"💾 Model saved to: {model_file}")
    print(f"💾 Summary saved to: {summary_file}")
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS =====
    if 'statistical_analysis' in results:
        print("\n" + "="*80)
        print("📊 Q-LEARNING RESULTS WITH 95% CONFIDENCE INTERVALS")
        print("="*80)
        format_ci_results(results['statistical_analysis'])
    
    print(f"\n🏆 Final Results:")
    print(f"   Total episodes: {episodes}")
    print(f"   Overall success rate: {results['overall_success_rate']:.1%}")
    print(f"   Best reward: {q_learner.best_reward:.2f}")
    print(f"   Final avg reward: {results.get('final_avg_reward', 0):.2f}")
    print(f"   Avg successful days: {results.get('avg_successful_days', 0):.1f}")
    print(f"   Avg successful rounds: {results.get('avg_successful_rounds', 0):.1f}")
    print(f"   Best sequence: {q_learner.best_sequence}")
    
    # Print transition matrix summary
    print(f"\n📈 Learned Transition Matrix (most likely transitions):")
    trans_matrix = q_learner.get_transition_matrix()
    for i, emotion in enumerate(EMOTIONS):
        best_next_idx = np.argmax(trans_matrix[i])
        best_next = EMOTIONS[best_next_idx]
        prob = trans_matrix[i, best_next_idx]
        print(f"   {emotion:10s} -> {best_next:10s} (p={prob:.3f})")
    
    return results


# Helper function for running the experiment
def run_simplified_qlearning_experiment(
    scenario_configs: List[Dict[str, Any]],
    num_episodes: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Convenience wrapper for running simplified Q-learning experiment"""
    
    print("=" * 60)
    print("🎯 SIMPLIFIED Q-LEARNING EXPERIMENT")
    print("=" * 60)
    print(f"Number of scenarios: {len(scenario_configs)}")
    print(f"Total episodes: {num_episodes}")
    print(f"Reward function: R = R_base + R_time + R_rounds")
    print(f"  R_base = 10·I[success] - 5·I[failure]")
    print(f"  R_time = 20·max(0, (D_target - D)/D_target)")
    print(f"  R_rounds = 15/(1 + log(R + 1))")
    print("=" * 60)
    
    # Default parameters
    params = {
        'scenarios': scenario_configs,
        'episodes': num_episodes,
        'out_dir': 'results',
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'credit_assignment': 'equal',  # Same as DQN
    }
    
    # Update with any provided kwargs
    params.update(kwargs)
    
    # Run experiment
    return run_qlearning_experiment(**params)