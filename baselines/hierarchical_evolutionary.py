"""
Model 1: Evolutionary Bayesian Optimization with Hierarchical Bayesian Updates
for Sparse Emotional Sequence Data
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from baselines.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

# ============================================================================
# EMOTION HIERARCHY DEFINITIONS
# ============================================================================

# Base emotions (7 individual)
BASE_EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_BASE = len(BASE_EMOTIONS)

# Emotion groups (3 groups for hierarchical modeling)
EMOTION_GROUPS = {
    'positive': ['happy', 'surprising'],
    'negative': ['angry', 'disgust', 'fear'],
    'neutral': ['sad', 'neutral']
}

GROUP_NAMES = list(EMOTION_GROUPS.keys())
N_GROUPS = len(GROUP_NAMES)

# Mapping functions
def emotion_to_group(emotion: str) -> str:
    """Map base emotion to its group"""
    for group, emotions in EMOTION_GROUPS.items():
        if emotion in emotions:
            return group
    return 'neutral'  # Default fallback

def group_to_emotions(group: str) -> List[str]:
    """Get all base emotions in a group"""
    return EMOTION_GROUPS.get(group, ['neutral'])

# Index mappings for fast lookup
EMOTION_TO_IDX = {emotion: i for i, emotion in enumerate(BASE_EMOTIONS)}
GROUP_TO_IDX = {group: i for i, group in enumerate(GROUP_NAMES)}

@dataclass
class EmotionSequence:
    """An emotional sequence with fitness score and hierarchical representation"""
    base_sequence: List[str]  # Original sequence ['happy', 'angry', ...]
    group_sequence: List[str]  # Group sequence ['positive', 'negative', ...]
    fitness: float = 0.0
    negotiation_result: Dict[str, Any] = None
    transition_stats: Dict = None  # Pre-computed transition statistics
    
    def __post_init__(self):
        if self.negotiation_result is None:
            self.negotiation_result = {}
        if self.transition_stats is None:
            self.transition_stats = self._compute_transition_stats()
    
    def _compute_transition_stats(self) -> Dict:
        """Pre-compute transition counts for fast Bayesian updates"""
        # Count base-level transitions
        base_counts = np.zeros((N_BASE, N_BASE), dtype=int)
        for i in range(len(self.base_sequence) - 1):
            from_idx = EMOTION_TO_IDX[self.base_sequence[i]]
            to_idx = EMOTION_TO_IDX[self.base_sequence[i + 1]]
            base_counts[from_idx, to_idx] += 1
        
        # Count group-level transitions
        group_counts = np.zeros((N_GROUPS, N_GROUPS), dtype=int)
        for i in range(len(self.group_sequence) - 1):
            from_idx = GROUP_TO_IDX[self.group_sequence[i]]
            to_idx = GROUP_TO_IDX[self.group_sequence[i + 1]]
            group_counts[from_idx, to_idx] += 1
        
        # Count within-group transitions (group → specific emotion)
        within_group_counts = {}
        for group in GROUP_NAMES:
            group_emotions = group_to_emotions(group)
            if len(group_emotions) > 1:  # Only for groups with multiple emotions
                counts = np.zeros((len(group_emotions), len(group_emotions)), dtype=int)
                within_group_counts[group] = counts
        
        # Fill within-group counts
        for i in range(len(self.base_sequence) - 1):
            from_emotion = self.base_sequence[i]
            to_emotion = self.base_sequence[i + 1]
            from_group = emotion_to_group(from_emotion)
            to_group = emotion_to_group(to_emotion)
            
            if from_group == to_group and from_group in within_group_counts:
                group_emotions = group_to_emotions(from_group)
                from_sub_idx = group_emotions.index(from_emotion)
                to_sub_idx = group_emotions.index(to_emotion)
                within_group_counts[from_group][from_sub_idx, to_sub_idx] += 1
        
        return {
            'base_counts': base_counts,
            'group_counts': group_counts,
            'within_group_counts': within_group_counts,
            'total_base_transitions': np.sum(base_counts),
            'total_group_transitions': np.sum(group_counts)
        }
    
    def __len__(self):
        return len(self.base_sequence)

class HierarchicalBayesianOptimizer(BaseEmotionModel):
    """
    Evolutionary Bayesian Optimization with HIERARCHICAL modeling
    for sparse emotional sequence data
    
    Key innovations:
    1. Hierarchical structure: Groups → Emotions (reduces sparsity)
    2. Multi-level Bayesian updates
    3. Smart crossover for variable-length sequences
    4. Regularization for limited data
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        bayesian_lambda: float = 0.6,  # Learning rate for Bayesian updates
        temperature_options: List[Tuple[float, float]] = None
    ):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bayesian_lambda = bayesian_lambda
        
        # Temperature options
        if temperature_options is None:
            self.temperature_options = [
                (0.25, 0.1), (0.5, 0.2), (0.75, 0.3)
            ]
        
        # ================= HIERARCHICAL TRANSITION MATRICES =================
        # Level 1: Group transitions (3x3, less sparse)
        self.group_matrix = self._initialize_group_matrix()
        
        # Level 2: Within-group emotion transitions
        self.within_group_matrices = {}
        for group in GROUP_NAMES:
            group_emotions = group_to_emotions(group)
            n_emotions = len(group_emotions)
            if n_emotions > 1:
                # Initialize with slight bias to stay in same emotion
                matrix = np.eye(n_emotions) * 0.6 + np.ones((n_emotions, n_emotions)) * 0.1
                self.within_group_matrices[group] = matrix / matrix.sum(axis=1, keepdims=True)
        
        # Level 3: Full base matrix (7x7) - will be computed from hierarchical structure
        self.base_matrix = self._reconstruct_base_matrix()
        
        # Temperature schedule
        self.temperature_schedule = (0.5, 0.2)  # Default
        
        # ================= EVOLUTIONARY COMPONENTS =================
        self.elite_sequences: List[EmotionSequence] = []  # Top sequences
        self.all_sequences: List[EmotionSequence] = []    # All sequences seen
        
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_sequence = None  # Track best emotional sequence
        
        # Current state
        self.current_emotion = 'neutral'
        self.current_group = 'neutral'
        
        # Priors for regularization
        self._initialize_priors()
    
    def _initialize_group_matrix(self) -> np.ndarray:
        """Initialize group-level transition matrix (3x3)"""
        # Game theory inspired: positive→positive, negative→neutral, etc.
        matrix = np.array([
            # To: positive, negative, neutral
            [0.4, 0.3, 0.3],  # From positive
            [0.2, 0.4, 0.4],  # From negative  
            [0.3, 0.3, 0.4],  # From neutral
        ])
        return matrix / matrix.sum(axis=1, keepdims=True)
    
    def _reconstruct_base_matrix(self) -> np.ndarray:
        """Reconstruct 7x7 base matrix from hierarchical structure"""
        base_matrix = np.zeros((N_BASE, N_BASE))
        
        for from_idx, from_emotion in enumerate(BASE_EMOTIONS):
            from_group = emotion_to_group(from_emotion)
            from_group_idx = GROUP_TO_IDX[from_group]
            
            # Find which emotion this is within its group
            group_emotions = group_to_emotions(from_group)
            within_group_idx = group_emotions.index(from_emotion) if from_emotion in group_emotions else 0
            
            for to_idx, to_emotion in enumerate(BASE_EMOTIONS):
                to_group = emotion_to_group(to_emotion)
                to_group_idx = GROUP_TO_IDX[to_group]
                
                # 1. Probability of switching to target group
                group_prob = self.group_matrix[from_group_idx, to_group_idx]
                
                # 2. Probability of specific emotion within target group
                if from_group == to_group and from_group in self.within_group_matrices:
                    # Stay in same group, use within-group matrix
                    group_emotions = group_to_emotions(from_group)
                    to_within_idx = group_emotions.index(to_emotion) if to_emotion in group_emotions else 0
                    within_prob = self.within_group_matrices[from_group][within_group_idx, to_within_idx]
                    base_matrix[from_idx, to_idx] = within_prob
                else:
                    # Switch groups, assume uniform distribution within target group
                    group_emotions = group_to_emotions(to_group)
                    within_prob = 1.0 / len(group_emotions)
                    base_matrix[from_idx, to_idx] = group_prob * within_prob
        
        # Normalize
        row_sums = base_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return base_matrix / row_sums
    
    def _initialize_priors(self):
        """Initialize Bayesian priors for regularization"""
        # Group-level prior (Dirichlet with concentration 2.0)
        self.group_prior = np.ones((N_GROUPS, N_GROUPS)) * 2.0
        
        # Within-group priors
        self.within_group_priors = {}
        for group in GROUP_NAMES:
            group_emotions = group_to_emotions(group)
            n = len(group_emotions)
            if n > 1:
                # Prior favors staying in same emotion (diagonal dominance)
                prior = np.eye(n) * 3.0 + np.ones((n, n)) * 1.0
                self.within_group_priors[group] = prior
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion using hierarchical sampling"""
        # Get temperature
        tau_0, delta = self.temperature_schedule
        round_num = state.get('round', 1)
        temperature = max(0.1, tau_0 * ((1 - delta) ** round_num))
        
        # Hierarchical sampling: Group → Emotion within group
        from_group = self.current_group
        from_group_idx = GROUP_TO_IDX[from_group]
        
        # 1. Sample next group
        group_probs = self.group_matrix[from_group_idx]
        next_group_idx = np.random.choice(N_GROUPS, p=group_probs)
        next_group = GROUP_NAMES[next_group_idx]
        
        # 2. Sample emotion within next group
        if from_group == next_group and next_group in self.within_group_matrices:
            # Stay in same group, use within-group matrix
            group_emotions = group_to_emotions(next_group)
            from_emotion_idx = group_emotions.index(self.current_emotion) if self.current_emotion in group_emotions else 0
            within_probs = self.within_group_matrices[next_group][from_emotion_idx]
            next_emotion = group_emotions[np.random.choice(len(group_emotions), p=within_probs)]
        else:
            # Switch groups, sample uniformly within target group
            group_emotions = group_to_emotions(next_group)
            next_emotion = random.choice(group_emotions)
        
        # Update state
        self.current_emotion = next_emotion
        self.current_group = next_group
        
        # Emotion prompts
        emotion_prompts = {
            "happy": "Use an optimistic and positive tone",
            "surprising": "Use an engaging and unexpected approach",
            "angry": "Use a firm and assertive tone",
            "sad": "Use an empathetic and understanding tone", 
            "disgust": "Use a disappointed tone",
            "fear": "Use a cautious and concerned tone",
            "neutral": "Use a balanced and professional tone"
        }
        
        return {
            "emotion": next_emotion,
            "emotion_text": emotion_prompts.get(next_emotion, "Professional tone"),
            "temperature": temperature,
            "group": next_group,
            "generation": self.generation,
            "matrix_entropy": self._calculate_hierarchical_entropy()
        }
    
    def evolve_generation(self, negotiation_results: List[Dict[str, Any]]):
        """Evolve using hierarchical Bayesian updates"""
        self.generation += 1
        print(f"\n{'='*60}")
        print(f"🧬 GENERATION {self.generation} - HIERARCHICAL BAYESIAN UPDATE")
        print(f"{'='*60}")
        
        # 1. Convert results to EmotionSequence objects with hierarchical stats
        new_sequences = []
        for result in negotiation_results:
            if 'emotion_sequence' in result and result['emotion_sequence']:
                base_seq = result['emotion_sequence']
                group_seq = [emotion_to_group(e) for e in base_seq]
                
                seq = EmotionSequence(
                    base_sequence=base_seq,
                    group_sequence=group_seq,
                    fitness=self.evaluate_sequence_fitness(result),
                    negotiation_result=result
                )
                new_sequences.append(seq)
        
        if not new_sequences:
            print("⚠️ No sequences to evolve from")
            return
        
        # 2. Add to total sequence pool
        self.all_sequences.extend(new_sequences)
        
        # 3. Select elite sequences (top 30%)
        new_sequences.sort(key=lambda s: s.fitness, reverse=True)
        elite_count = max(2, len(new_sequences) // 3)  # Top 33%
        elites = new_sequences[:elite_count]
        
        # 4. Update best fitness
        if elites[0].fitness > self.best_fitness:
            self.best_fitness = elites[0].fitness
            self.best_sequence = elites[0].base_sequence.copy()  # Store the best sequence
        
        print(f"📊 Sequence Statistics:")
        print(f"   New sequences: {len(new_sequences)}")
        print(f"   Elite sequences: {len(elites)}")
        print(f"   Best fitness: {self.best_fitness:.3f}")
        print(f"   Total sequence pool: {len(self.all_sequences)}")
        
        # ================= HIERARCHICAL BAYESIAN UPDATES =================
        print(f"\n🔁 HIERARCHICAL BAYESIAN UPDATES:")
        
        # 5. Update group-level matrix (Level 1)
        self._update_group_matrix(elites)
        print(f"   Level 1: Group matrix updated ({N_GROUPS}x{N_GROUPS})")
        
        # 6. Update within-group matrices (Level 2)
        self._update_within_group_matrices(elites)
        print(f"   Level 2: Within-group matrices updated")
        
        # 7. Reconstruct full base matrix (Level 3)
        self.base_matrix = self._reconstruct_base_matrix()
        print(f"   Level 3: Full {N_BASE}x{N_BASE} matrix reconstructed")
        
        # ================= EVOLUTIONARY OPERATIONS =================
        print(f"\n🧬 EVOLUTIONARY OPERATIONS:")
        
        # 8. Generate offspring through smart crossover
        offspring = self._generate_hierarchical_offspring(elites)
        print(f"   Generated {len(offspring)} offspring sequences")
        
        # 9. Add offspring to sequence pool
        self.all_sequences.extend(offspring)
        
        # 10. Update elite sequences
        self.elite_sequences = elites
        
        print(f"📈 Final Stats: Group matrix entropy = {self._calculate_group_entropy():.3f}")
        print(f"                Full matrix entropy = {self._calculate_full_entropy():.3f}")
    
    def _update_group_matrix(self, elite_sequences: List[EmotionSequence]):
        """Bayesian update of group-level transition matrix"""
        # Collect group transition counts from elites
        group_counts = np.zeros((N_GROUPS, N_GROUPS))
        
        for seq in elite_sequences:
            stats = seq.transition_stats
            group_counts += stats['group_counts']
        
        # Add prior for regularization
        group_counts_smoothed = group_counts + self.group_prior
        
        # Bayesian update: blend with current matrix
        row_sums = group_counts_smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        
        mle_group_matrix = group_counts_smoothed / row_sums
        
        # Blend with current matrix using learning rate
        self.group_matrix = (
            self.bayesian_lambda * self.group_matrix + 
            (1 - self.bayesian_lambda) * mle_group_matrix
        )
        
        # Ensure valid probabilities
        self.group_matrix = self.group_matrix / self.group_matrix.sum(axis=1, keepdims=True)
    
    def _update_within_group_matrices(self, elite_sequences: List[EmotionSequence]):
        """Bayesian update of within-group transition matrices - FIXED VERSION"""
        
        print(f"\n    🔄 Updating within-group matrices...")
        
        for group in GROUP_NAMES:
            group_emotions = group_to_emotions(group)
            n_emotions = len(group_emotions)
            
            if n_emotions <= 1:
                continue
            
            # DEBUG: Show what we're looking for
            print(f"      Group '{group}' has emotions: {group_emotions}")
            
            # Method 1: Count from sequences that STAY in this group
            within_counts = np.zeros((n_emotions, n_emotions))
            total_within_transitions = 0
            
            for seq in elite_sequences:
                for i in range(len(seq.base_sequence) - 1):
                    from_emotion = seq.base_sequence[i]
                    to_emotion = seq.base_sequence[i + 1]
                    from_group = emotion_to_group(from_emotion)
                    to_group = emotion_to_group(to_emotion)
                    
                    # Only count transitions that STAY within the same group
                    if from_group == group and to_group == group:
                        from_idx = group_emotions.index(from_emotion)
                        to_idx = group_emotions.index(to_emotion)
                        within_counts[from_idx, to_idx] += 1
                        total_within_transitions += 1
            
            print(f"        Found {total_within_transitions} within-group transitions")
            
            # If insufficient data, use a fallback strategy
            if total_within_transitions < 3:
                print(f"        ⚠️  Insufficient data, using reinforcement learning")
                # Method 2: Reinforce successful patterns
                self._reinforce_successful_within_group_patterns(group, elite_sequences)
                continue
            
            # Bayesian update with proper data
            if group in self.within_group_priors:
                prior = self.within_group_priors[group]
                within_counts_smoothed = within_counts + prior
            else:
                within_counts_smoothed = within_counts + 1.0
            
            # Calculate MLE
            row_sums = within_counts_smoothed.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            mle_within_matrix = within_counts_smoothed / row_sums
            
            # Blend with current
            current_matrix = self.within_group_matrices[group]
            updated_matrix = (
                self.bayesian_lambda * current_matrix + 
                (1 - self.bayesian_lambda) * mle_within_matrix
            )
            
            self.within_group_matrices[group] = updated_matrix / updated_matrix.sum(axis=1, keepdims=True)

    def _reinforce_successful_within_group_patterns(self, group: str, elite_sequences: List[EmotionSequence]):
        """Alternative learning when within-group data is sparse"""
        group_emotions = group_to_emotions(group)
        
        # Find most successful sequences involving this group
        successful_patterns = []
        
        for seq in elite_sequences:
            if seq.fitness > 0:  # Successful sequences
                for i in range(len(seq.base_sequence) - 1):
                    from_emotion = seq.base_sequence[i]
                    to_emotion = seq.base_sequence[i + 1]
                    
                    if (emotion_to_group(from_emotion) == group and 
                        emotion_to_group(to_emotion) == group):
                        successful_patterns.append((from_emotion, to_emotion))
        
        if successful_patterns:
            # Reinforce successful patterns
            matrix = self.within_group_matrices[group].copy()
            
            for from_emotion, to_emotion in successful_patterns:
                from_idx = group_emotions.index(from_emotion)
                to_idx = group_emotions.index(to_emotion)
                matrix[from_idx, to_idx] *= 1.2  # Boost successful transitions
            
            # Normalize
            self.within_group_matrices[group] = matrix / matrix.sum(axis=1, keepdims=True)
    def _generate_hierarchical_offspring(self, elites: List[EmotionSequence]) -> List[EmotionSequence]:
        """Generate offspring using smart hierarchical crossover"""
        if len(elites) < 2:
            return []
        
        offspring = []
        
        for _ in range(len(elites)):  # Generate as many offspring as elites
            # Select parents
            parent1, parent2 = random.sample(elites, 2)
            
            # Choose crossover method based on sequence characteristics
            if abs(len(parent1) - len(parent2)) <= 3:
                # Similar lengths: use standard crossover
                child_base_seq = self._crossover_fixed_length(
                    parent1.base_sequence, 
                    parent2.base_sequence
                )
            else:
                # Different lengths: use hierarchical crossover
                child_base_seq = self._crossover_hierarchical(
                    parent1, 
                    parent2
                )
            
            # Apply hierarchical mutation
            child_base_seq = self._mutate_hierarchical(child_base_seq)
            
            # Ensure minimum length
            if len(child_base_seq) < 2:
                child_base_seq.extend(random.sample(BASE_EMOTIONS, 2))
            
            # Convert to EmotionSequence
            group_seq = [emotion_to_group(e) for e in child_base_seq]
            offspring_seq = EmotionSequence(
                base_sequence=child_base_seq,
                group_sequence=group_seq,
                fitness=0.0  # Will be evaluated in next generation
            )
            
            offspring.append(offspring_seq)
        
        return offspring
    
    def print_complete_hierarchical_structure(self):
        """Print ALL hierarchical levels with details"""
        print("\n" + "="*80)
        print("🎭 COMPLETE HIERARCHICAL EMOTIONAL STRUCTURE")
        print("="*80)
        
        # ================= LEVEL 1: GROUP TRANSITIONS =================
        print("\n📊 LEVEL 1: GROUP TRANSITIONS (3×3)")
        print("-"*40)
        print(f"{'From':<10} {'To':<10} {'Probability':<12} {'Interpretation':<30}")
        print("-"*70)
        
        for i, from_group in enumerate(GROUP_NAMES):
            for j, to_group in enumerate(GROUP_NAMES):
                prob = self.group_matrix[i, j]
                
                # Interpretation
                if from_group == to_group:
                    interpretation = "Stay in same group"
                elif (from_group == "negative" and to_group == "positive"):
                    interpretation = "De-escalate from negative"
                elif (from_group == "positive" and to_group == "negative"):
                    interpretation = "Apply pressure"
                elif (from_group == "neutral" and to_group == "positive"):
                    interpretation = "Become optimistic"
                else:
                    interpretation = "Transition"
                
                if prob > 0.15:  # Show significant transitions
                    print(f"{from_group:<10} → {to_group:<10} {prob:.3f}{'':<9} {interpretation:<30}")
        
        # ================= LEVEL 2: WITHIN-GROUP TRANSITIONS =================
        print("\n📊 LEVEL 2: WITHIN-GROUP EMOTION TRANSITIONS")
        print("-"*40)
        
        for group in GROUP_NAMES:
            if group in self.within_group_matrices:
                print(f"\n  🎯 Group: {group} ({len(group_to_emotions(group))} emotions)")
                print(f"  {'Emotion':<12} {'→ Emotion':<12} {'Probability':<12} {'Pattern':<20}")
                print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
                
                group_emotions = group_to_emotions(group)
                matrix = self.within_group_matrices[group]
                
                for i, from_emotion in enumerate(group_emotions):
                    for j, to_emotion in enumerate(group_emotions):
                        prob = matrix[i, j]
                        
                        # Pattern description
                        if i == j:
                            pattern = "Self-persistence"
                        elif prob > 0.3:
                            pattern = "Strong preference"
                        elif prob > 0.2:
                            pattern = "Moderate shift"
                        else:
                            pattern = "Rare shift"
                        
                        if prob > 0.15 or i == j:  # Show self-transitions and significant ones
                            print(f"  {from_emotion:<12} → {to_emotion:<12} {prob:.3f}{'':<9} {pattern:<20}")
        
        # ================= LEVEL 3: COMPUTED FULL 7×7 MATRIX =================
        print("\n📊 LEVEL 3: RECONSTRUCTED FULL EMOTION MATRIX (7×7)")
        print("-"*40)
        
        # Header

        for emotion in BASE_EMOTIONS:
            print(f"{emotion:<10}", end="")
        print()
        print(f"{'-'*12}", end="")
        for _ in BASE_EMOTIONS:
            print(f"{'-'*10}", end="")
        print()
        
        # Matrix values
        for i, from_emotion in enumerate(BASE_EMOTIONS):
            print(f"{from_emotion:<12}", end="")
            for j, to_emotion in enumerate(BASE_EMOTIONS):
                prob = self.base_matrix[i, j]
                from_group = emotion_to_group(from_emotion)
                to_group = emotion_to_group(to_emotion)
                
                # Color coding for visualization
                if from_group == to_group:
                    # Same group
                    if prob > 0.3:
                        display = f"{prob:.3f}✓"   # Strong same-group
                    else:
                        display = f"{prob:.3f}·"   # Weak same-group
                else:
                    # Cross-group
                    if prob > 0.1:
                        display = f"{prob:.3f}→"   # Significant cross-group
                    else:
                        display = f"{prob:.3f} "   # Minor cross-group
                
                print(f"{display:<10}", end="")
            print()
            print(f"{' ':<12}", end="")
            for j, to_emotion in enumerate(BASE_EMOTIONS):
                prob = self.base_matrix[i, j]
                # Show group info below
                to_group = emotion_to_group(to_emotion)
                if prob > 0.15:
                    group_symbol = to_group[0].upper()
                    print(f"({group_symbol}){'':<8}", end="")
                else:
                    print(f"{'':<9}", end="")
            print()
        
        # ================= SPECIFIC CROSS-GROUP EXAMPLES =================
        print("\n📊 KEY CROSS-GROUP TRANSITION EXAMPLES")
        print("-"*40)
        
        examples = [
            ("happy", "angry", "Positive → Negative (apply pressure)"),
            ("angry", "happy", "Negative → Positive (de-escalate)"),
            ("neutral", "fear", "Neutral → Negative (show concern)"),
            ("sad", "happy", "Neutral → Positive (become optimistic)"),
            ("surprising", "disgust", "Positive → Negative (show disappointment)"),
            ("fear", "neutral", "Negative → Neutral (calm down)"),
        ]
        
        for from_e, to_e, description in examples:
            prob = self._compute_specific_probability(from_e, to_e)
            from_g = emotion_to_group(from_e)
            to_g = emotion_to_group(to_e)
            
            if prob > 0.05:  # Show meaningful transitions
                print(f"  {from_e:10} ({from_g:8}) → {to_e:10} ({to_g:8}): {prob:.3f}  ← {description}")

    def _compute_specific_probability(self, from_emotion: str, to_emotion: str) -> float:
        """Compute exact probability for specific emotion transition"""
        from_idx = EMOTION_TO_IDX[from_emotion]
        to_idx = EMOTION_TO_IDX[to_emotion]
        return float(self.base_matrix[from_idx, to_idx])

    def get_detailed_transition_probs(self) -> Dict[str, Any]:
        """Get detailed transition probabilities for analysis"""
        detailed = {
            'group_level': {},
            'within_group_level': {},
            'specific_transitions': {},
            'summary_metrics': {}
        }
        
        # Group level
        for i, from_group in enumerate(GROUP_NAMES):
            for j, to_group in enumerate(GROUP_NAMES):
                key = f"{from_group}→{to_group}"
                detailed['group_level'][key] = float(self.group_matrix[i, j])
        
        # Within group level
        for group in GROUP_NAMES:
            if group in self.within_group_matrices:
                group_emotions = group_to_emotions(group)
                matrix = self.within_group_matrices[group]
                
                detailed['within_group_level'][group] = {
                    'emotions': group_emotions,
                    'matrix': matrix.tolist(),
                    'self_persistence': float(np.mean(np.diag(matrix))),  # Avg diagonal
                    'entropy': float(self._calculate_matrix_entropy(matrix))
                }
        
        # Key specific transitions
        cross_examples = []
        for from_emotion in BASE_EMOTIONS:
            from_group = emotion_to_group(from_emotion)
            for to_emotion in BASE_EMOTIONS:
                to_group = emotion_to_group(to_emotion)
                
                if from_group != to_group:  # Only cross-group
                    prob = self._compute_specific_probability(from_emotion, to_emotion)
                    if prob > 0.05:  # Only significant ones
                        key = f"{from_emotion}→{to_emotion}"
                        detailed['specific_transitions'][key] = {
                            'probability': float(prob),
                            'from_group': from_group,
                            'to_group': to_group,
                            'interpretation': self._interpret_transition(from_emotion, to_emotion)
                        }
        
        # Summary metrics
        detailed['summary_metrics'] = {
            'group_matrix_entropy': float(self._calculate_group_entropy()),
            'full_matrix_entropy': float(self._calculate_full_entropy()),
            'avg_self_transition': float(np.mean(np.diag(self.base_matrix))),
            'avg_cross_group_transition': float(np.mean([
                self.base_matrix[i, j] 
                for i in range(N_BASE) 
                for j in range(N_BASE) 
                if emotion_to_group(BASE_EMOTIONS[i]) != emotion_to_group(BASE_EMOTIONS[j])
            ]))
        }
        
        return detailed

    def _interpret_transition(self, from_emotion: str, to_emotion: str) -> str:
        """Interpret the meaning of an emotion transition"""
        from_group = emotion_to_group(from_emotion)
        to_group = emotion_to_group(to_emotion)
        
        if from_group == to_group:
            return f"Stay in {from_group} group"
        
        interpretations = {
            ("positive", "negative"): "Apply pressure/be firm",
            ("negative", "positive"): "De-escalate/show goodwill",
            ("positive", "neutral"): "Become professional",
            ("neutral", "positive"): "Show optimism",
            ("negative", "neutral"): "Calm down/de-escalate",
            ("neutral", "negative"): "Express concern/worry"
        }
        
        return interpretations.get((from_group, to_group), "Transition")

    
    def _crossover_hierarchical(self, parent1: EmotionSequence, parent2: EmotionSequence) -> List[str]:
        """
        Smart hierarchical crossover that respects group structure
        
        Strategy: Crossover at group boundaries rather than random positions
        """
        seq1 = parent1.base_sequence
        seq2 = parent2.base_sequence
        
        # Find all group boundaries in both sequences
        def get_group_boundaries(sequence):
            boundaries = []
            current_group = emotion_to_group(sequence[0])
            
            for i, emotion in enumerate(sequence):
                group = emotion_to_group(emotion)
                if group != current_group:
                    boundaries.append(i)
                    current_group = group
            
            return boundaries
        
        boundaries1 = get_group_boundaries(seq1)
        boundaries2 = get_group_boundaries(seq2)
        
        if boundaries1 and boundaries2:
            # Crossover at a group boundary
            boundary1 = random.choice(boundaries1)
            boundary2 = random.choice(boundaries2)
            
            # Take first part from parent1, second from parent2
            child = seq1[:boundary1] + seq2[boundary2:]
        else:
            # No group boundaries, use segment crossover
            cut1 = random.randint(1, len(seq1) - 1) if len(seq1) > 1 else 1
            cut2 = random.randint(1, len(seq2) - 1) if len(seq2) > 1 else 1
            child = seq1[:cut1] + seq2[cut2:]
        
        return child
    
    def _crossover_fixed_length(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Standard crossover for similar-length sequences"""
        # Align lengths by truncating or extending
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        # Trim or pad sequences to similar length
        seq1_adj = seq1[:min_len] if len(seq1) > min_len else seq1 + random.choices(BASE_EMOTIONS, k=max_len - min_len)
        seq2_adj = seq2[:min_len] if len(seq2) > min_len else seq2 + random.choices(BASE_EMOTIONS, k=max_len - min_len)
        
        # Single-point crossover
        crossover_point = random.randint(1, min_len - 1) if min_len > 1 else 1
        
        if random.random() < 0.5:
            child = seq1_adj[:crossover_point] + seq2_adj[crossover_point:]
        else:
            child = seq2_adj[:crossover_point] + seq1_adj[crossover_point:]
        
        return child
    
    def _mutate_hierarchical(self, sequence: List[str]) -> List[str]:
        """Hierarchical mutation that respects group structure"""
        mutated = sequence.copy()
        
        # Mutation types with different probabilities
        mutation_type = random.random()
        
        if mutation_type < 0.6:
            # Point mutation: change individual emotions
            for i in range(len(mutated)):
                if random.random() < self.mutation_rate:
                    # When mutating, consider staying in same group 70% of time
                    current_emotion = mutated[i]
                    current_group = emotion_to_group(current_emotion)
                    
                    if random.random() < 0.7:
                        # Stay in same group, pick different emotion
                        group_emotions = group_to_emotions(current_group)
                        if len(group_emotions) > 1:
                            new_emotion = random.choice([e for e in group_emotions if e != current_emotion])
                            mutated[i] = new_emotion
                    else:
                        # Switch group completely
                        mutated[i] = random.choice(BASE_EMOTIONS)
        
        elif mutation_type < 0.8 and len(mutated) > 2:
            # Group boundary mutation: insert/remove at group boundaries
            boundaries = []
            for i in range(1, len(mutated)):
                if emotion_to_group(mutated[i]) != emotion_to_group(mutated[i-1]):
                    boundaries.append(i)
            
            if boundaries:
                boundary = random.choice(boundaries)
                if random.random() < 0.5:
                    # Insert emotion at boundary
                    group1 = emotion_to_group(mutated[boundary-1])
                    group2 = emotion_to_group(mutated[boundary])
                    
                    # Insert emotion from either group
                    if random.random() < 0.5:
                        new_emotion = random.choice(group_to_emotions(group1))
                    else:
                        new_emotion = random.choice(group_to_emotions(group2))
                    
                    mutated.insert(boundary, new_emotion)
                else:
                    # Remove emotion at boundary
                    if len(mutated) > 2:
                        mutated.pop(boundary)
        
        else:
            # Length mutation
            if random.random() < 0.3 and len(mutated) > 2:
                # Delete random emotion
                del_idx = random.randint(0, len(mutated) - 1)
                mutated.pop(del_idx)
            elif random.random() < 0.3:
                # Insert random emotion
                insert_idx = random.randint(0, len(mutated))
                mutated.insert(insert_idx, random.choice(BASE_EMOTIONS))
        
        return mutated
    
    def evaluate_sequence_fitness(self, result: Dict[str, Any]) -> float:
        """FIXED fitness calculation"""
        success = result.get('final_state') == 'accept'
        collection_days = result.get('collection_days', 0)
        target_days = result.get('creditor_target_days', 30)
        negotiation_rounds = result.get('negotiation_rounds', 1)
        
        if not success:
            return 0.0
        
        # FIX 1: Handle collection_days properly
        if collection_days is None:
            print(f"⚠️ WARNING: collection_days is None in result: {result.keys()}")
            return 0.0
        
        # FIX 2: Proper efficiency calculation
        if target_days > 0:
            # Ratio of actual to target (lower is better)
            ratio = collection_days / target_days
            
            # Efficiency curve:
            # ratio=0.5 → efficiency=1.0 (collected in half the time)
            # ratio=1.0 → efficiency=0.5 (collected exactly on time)
            # ratio=2.0 → efficiency=0.0 (took twice as long)
            if ratio <= 0.5:
                time_efficiency = 1.0
            elif ratio <= 1.0:
                time_efficiency = 1.0 - (ratio - 0.5)  # Linear from 1.0 to 0.5
            elif ratio <= 2.0:
                time_efficiency = 0.5 * (2.0 - ratio)  # Linear from 0.5 to 0.0
            else:
                time_efficiency = 0.0
        else:
            time_efficiency = 0.5  # Default if target_days is weird
        
        # FIX 3: Debug output
        debug = False
        if debug:
            print(f"DEBUG Fitness: days={collection_days}, target={target_days}, "
                f"ratio={ratio:.2f}, time_eff={time_efficiency:.3f}")
        
        # FIX 4: Negotiation efficiency (shorter negotiations = better)
        round_efficiency = 1.0 / (1 + np.log(negotiation_rounds))
        
        # FIX 5: Sequence length penalty (mild)
        seq_length = len(result.get('emotion_sequence', []))
        length_penalty = 1.0 / (1 + 0.02 * seq_length)  # Very mild penalty
        
        # FIX 6: Combined fitness
        fitness = 100.0 * time_efficiency * round_efficiency * length_penalty
        
        return max(0.0, min(100.0, fitness))  # Bound between 0-100
    
    def _calculate_hierarchical_entropy(self) -> Dict[str, float]:
        """Calculate entropy at all hierarchical levels"""
        return {
            'group_entropy': self._calculate_group_entropy(),
            'full_entropy': self._calculate_full_entropy(),
            'within_group_entropies': {
                group: self._calculate_matrix_entropy(matrix)
                for group, matrix in self.within_group_matrices.items()
            }
        }
    
    def _calculate_group_entropy(self) -> float:
        """Calculate entropy of group-level matrix"""
        return self._calculate_matrix_entropy(self.group_matrix)
    
    def _calculate_full_entropy(self) -> float:
        """Calculate entropy of full base matrix"""
        return self._calculate_matrix_entropy(self.base_matrix)
    
    def _calculate_matrix_entropy(self, matrix: np.ndarray) -> float:
        """Calculate entropy of any transition matrix"""
        entropy = 0.0
        for i in range(matrix.shape[0]):
            row = matrix[i]
            row_entropy = -np.sum(row * np.log(row + 1e-10))
            entropy += row_entropy
        return entropy / matrix.shape[0]
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Store negotiation result"""
        # This is now handled in evolve_generation
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive hierarchical statistics"""
        hierarchical_entropy = self._calculate_hierarchical_entropy()
        
        return {
            'generation': self.generation,
            'best_fitness': float(self.best_fitness),
            'best_sequence': self.best_sequence,
            'elite_count': len(self.elite_sequences),
            'total_sequences': len(self.all_sequences),
            'current_emotion': self.current_emotion,
            'current_group': self.current_group,
            'hierarchical_entropy': hierarchical_entropy,
            'group_matrix': self.group_matrix.tolist(),
            'within_group_matrices': {
                group: matrix.tolist()
                for group, matrix in self.within_group_matrices.items()
            }
        }
    
    def reset(self) -> None:
        """Reset for new negotiation"""
        self.current_emotion = 'neutral'
        self.current_group = 'neutral'
        # Keep learned hierarchical matrices intact
    
    def save_model(self, filepath: str) -> None:
        """Save trained hierarchical model"""
        model_data = {
            'group_matrix': self.group_matrix.tolist(),
            'within_group_matrices': {
                group: matrix.tolist() for group, matrix in self.within_group_matrices.items()
            },
            'base_matrix': self.base_matrix.tolist(),
            'temperature_schedule': self.temperature_schedule,
            'best_sequence': self.best_sequence,
            'best_fitness': self.best_fitness,
            'generation': self.generation,
            'model_type': 'hierarchical_evolutionary',
            'parameters': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'bayesian_lambda': self.bayesian_lambda
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Hierarchical model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained hierarchical model"""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.group_matrix = np.array(model_data['group_matrix'])
        
        # Load within-group matrices
        self.within_group_matrices = {}
        for group, matrix_data in model_data['within_group_matrices'].items():
            self.within_group_matrices[group] = np.array(matrix_data)
        
        self.base_matrix = np.array(model_data['base_matrix'])
        self.temperature_schedule = tuple(model_data['temperature_schedule'])
        self.best_sequence = model_data['best_sequence']
        self.best_fitness = model_data['best_fitness']
        self.generation = model_data['generation']
        
        print(f"🔄 Hierarchical model loaded from: {filepath}")
        print(f"   Best fitness: {self.best_fitness:.3f}")
        print(f"   Generation: {self.generation}")

# ============================================================================
# EXPERIMENT RUNNER WITH HIERARCHICAL OPTIMIZER
# ============================================================================

def run_hierarchical_experiment(
    scenarios: List[Dict[str, Any]],
    generations: int = 10,
    negotiations_per_gen: int = 10,  # Reduced from 20 - better for sparse data
    mutation_rate: float = 0.15,  # Mutation rate parameter
    crossover_rate: float = 0.7,  # Crossover rate parameter
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results"
) -> Dict[str, Any]:
    """Run hierarchical evolutionary Bayesian experiment"""
    
    from llm.negotiator import DebtNegotiator
    
    # Create hierarchical optimizer with passed parameters
    optimizer = HierarchicalBayesianOptimizer(
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        bayesian_lambda=0.6
    )
    
    results = {
        'experiment_type': 'hierarchical_evolutionary_bayesian',
        'generations': generations,
        'negotiations_per_generation': negotiations_per_gen,
        'hierarchical_structure': {
            'groups': GROUP_NAMES,
            'group_members': EMOTION_GROUPS,
            'base_emotions': BASE_EMOTIONS
        },
        'generation_results': {},
        'scenarios_used': [s['id'] for s in scenarios]
    }
    
    for generation in range(generations):
        print(f"\n{'='*80}")
        print(f"🎯 GENERATION {generation + 1}/{generations}")
        print(f"{'='*80}")
        
        generation_negotiations = []
        
        # Run negotiations to collect sequences
        for neg_idx in range(negotiations_per_gen):
            scenario = scenarios[neg_idx % len(scenarios)]
            
            print(f"  🧪 Negotiation {neg_idx + 1}/{negotiations_per_gen} - {scenario['id']}")
            
            # Create negotiator
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=optimizer,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            generation_negotiations.append(result)
            
            # Show quick result
            outcome = "✅" if result.get('final_state') == 'accept' else "❌"
            days = result.get('collection_days', 'N/A')
            seq_len = len(result.get('emotion_sequence', []))
            print(f"     {outcome} Days: {days} | Seq length: {seq_len}")
        
        # ================= HIERARCHICAL EVOLUTION =================
        print(f"\n  🧬 Starting Hierarchical Evolution...")
        optimizer.evolve_generation(generation_negotiations)
        
        # Calculate statistics
        successful = [r for r in generation_negotiations if r.get('final_state') == 'accept']
        success_rate = len(successful) / len(generation_negotiations)
        
        if successful:
            avg_days = np.mean([r.get('collection_days', 0) for r in successful])
            avg_rounds = np.mean([len(r.get('dialog', [])) for r in successful])
        else:
            avg_days = avg_rounds = 0
        
        # Get optimizer stats
        stats = optimizer.get_stats()
        
        print(f"  📊 Generation Summary:")
        print(f"     Success rate: {success_rate:.1%}")
        print(f"     Avg days: {avg_days:.1f}")
        print(f"     Best fitness: {optimizer.best_fitness:.3f}")
        print(f"     Group matrix entropy: {stats['hierarchical_entropy']['group_entropy']:.3f}")
        print(f"     Elite sequences: {stats['elite_count']}")
        
        # Store results
        results['generation_results'][f'generation_{generation+1}'] = {
            'stats': stats,
            'success_rate': success_rate,
            'avg_days': float(avg_days),
            'avg_rounds': float(avg_rounds),
            'elite_sequences': [
                {
                    'sequence': seq.base_sequence,
                    'fitness': seq.fitness,
                    'length': len(seq)
                } for seq in optimizer.elite_sequences[:5]  # Store top 5
            ]
        }
    
    # Collect all negotiation results for statistical analysis
    all_negotiation_results = []
    for gen_data in results['generation_results'].values():
        all_negotiation_results.extend(gen_data.get('negotiation_results', []))
    
    # Final results
    results['final_stats'] = optimizer.get_stats()
    results['final_hierarchical_matrices'] = {
        'group_matrix': optimizer.group_matrix.tolist(),
        'within_group_matrices': {
            group: matrix.tolist()
            for group, matrix in optimizer.within_group_matrices.items()
        },
        'full_base_matrix': optimizer.base_matrix.tolist()
    }
    
    # ===== ADD STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS =====
    results = enhance_results_with_statistics(
        results, 
        all_negotiation_results, 
        scenarios, 
        method="bootstrap"
    )
    
    # Calculate overall performance metrics
    if all_negotiation_results:
        successful_results = [r for r in all_negotiation_results if r.get('final_state') == 'accept']
        results['overall_success_rate'] = len(successful_results) / len(all_negotiation_results)
        
        if successful_results:
            results['avg_successful_days'] = float(np.mean([r.get('collection_days', 0) for r in successful_results if r.get('collection_days') is not None]))
            results['avg_successful_rounds'] = float(np.mean([len(r.get('dialog', [])) for r in successful_results]))
        else:
            results['avg_successful_days'] = 0
            results['avg_successful_rounds'] = 0
    else:
        results['overall_success_rate'] = 0
        results['avg_successful_days'] = 0
        results['avg_successful_rounds'] = 0
    
    # Add final model stats including matrices
    results['final_stats'] = {
        'best_fitness': optimizer.best_fitness,
        'best_sequence': optimizer.best_sequence,
        'generation': optimizer.generation,
        'group_matrix': optimizer.group_matrix.tolist(),
        'within_group_matrices': {
            group: matrix.tolist() for group, matrix in optimizer.within_group_matrices.items()
        },
        'base_matrix': optimizer.base_matrix.tolist(),
        'temperature_schedule': optimizer.temperature_schedule,
        'group_entropy': optimizer._calculate_group_entropy(),
        'full_entropy': optimizer._calculate_full_entropy()
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/hierarchical_evolution_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save trained model
    model_file = f"{out_dir}/trained_hierarchical_model_{timestamp}.json"
    optimizer.save_model(model_file)
    
    # Save comprehensive summary file (matching vanilla model format)
    summary_file = f"{out_dir}/hierarchical_evolution_summary_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("HIERARCHICAL EVOLUTIONARY BAYESIAN EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Generations: {generations}\n")
        f.write(f"Negotiations per Generation: {negotiations_per_gen}\n")
        f.write(f"Total Negotiations: {len(all_negotiation_results)}\n")
        f.write(f"Scenarios Used: {len(scenarios)}\n\n")
        
        f.write("HIERARCHICAL STRUCTURE:\n")
        f.write("-" * 30 + "\n")
        for group, emotions in EMOTION_GROUPS.items():
            f.write(f"{group}: {emotions}\n")
        f.write("\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Success Rate: {results['overall_success_rate']:.1%}\n")
        f.write(f"Average Days (Successful): {results['avg_successful_days']:.1f}\n")
        f.write(f"Average Rounds (Successful): {results['avg_successful_rounds']:.1f}\n")
        f.write(f"Best Fitness: {optimizer.best_fitness:.3f}\n\n")
        
        f.write("BEST EMOTIONAL SEQUENCE:\n")
        f.write("-" * 30 + "\n")
        if optimizer.best_sequence:
            f.write(f"Sequence: {optimizer.best_sequence}\n")
            f.write(f"Fitness: {optimizer.best_fitness:.3f}\n\n")
        else:
            f.write("No best sequence found\n\n")
        
        f.write("FINAL HIERARCHICAL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        stats = optimizer.get_stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, list) and len(value) <= 10:
                f.write(f"{key}: {value}\n")
        
        f.write("\nGROUP TRANSITION MATRIX:\n")
        f.write("-" * 30 + "\n")
        for i, from_group in enumerate(GROUP_NAMES):
            for j, to_group in enumerate(GROUP_NAMES):
                prob = optimizer.group_matrix[i, j]
                f.write(f"{from_group} -> {to_group}: {prob:.3f}\n")
        
        if 'statistical_analysis' in results:
            f.write("\n95% CONFIDENCE INTERVALS:\n")
            f.write("-" * 30 + "\n")
            for metric, data in results['statistical_analysis'].items():
                if isinstance(data, dict) and 'mean' in data and 'ci' in data:
                    f.write(f"{metric}: {data['mean']:.3f} [{data['ci'][0]:.3f}, {data['ci'][1]:.3f}]\n")
    
    print(f"\n💾 Results saved to: {result_file}")
    print(f"💾 Summary saved to: {summary_file}")
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS =====
    if 'statistical_analysis' in results:
        print("\n" + "="*80)
        print("📊 HIERARCHICAL EVOLUTIONARY RESULTS WITH 95% CONFIDENCE INTERVALS")
        print("="*80)
        format_ci_results(results['statistical_analysis'])
    
    # Print final learned structure
    print(f"\n{'='*80}")
    print(f"🏆 FINAL LEARNED HIERARCHICAL STRUCTURE")
    print(f"{'='*80}")
    print(f"   Best fitness: {optimizer.best_fitness:.3f}")
    print(f"   Generation: {optimizer.generation}")
    print(f"   Elite sequences: {len(optimizer.elite_sequences)}")
    print(f"   Total sequences analyzed: {len(optimizer.all_sequences)}")

    # Show complete hierarchical structure
    optimizer.print_complete_hierarchical_structure()

    # Show learning summary
    print(f"\n📈 HIERARCHICAL LEARNING SUMMARY:")
    print(f"   Group Matrix Entropy: {optimizer._calculate_group_entropy():.3f} "
        f"(lower = more certain)")
    print(f"   Full Matrix Entropy: {optimizer._calculate_full_entropy():.3f}")
    print(f"   Within-group matrices learned:")

    for group in GROUP_NAMES:
        if group in optimizer.within_group_matrices:
            matrix = optimizer.within_group_matrices[group]
            diag_avg = np.mean(np.diag(matrix))
            entropy = optimizer._calculate_matrix_entropy(matrix)
            print(f"     {group:8}: {matrix.shape[0]}x{matrix.shape[1]}, "
                f"self-persistence={diag_avg:.3f}, entropy={entropy:.3f}")
    
    print(f"\n🏆 Final Results:")
    print(f"   Total negotiations: {len(all_negotiation_results)}")
    print(f"   Overall success rate: {results['overall_success_rate']:.1%}")
    print(f"   Avg successful days: {results['avg_successful_days']:.1f}")
    print(f"   Avg successful rounds: {results['avg_successful_rounds']:.1f}")
    print(f"   Best fitness: {optimizer.best_fitness:.3f}")
    if optimizer.best_sequence:
        print(f"   Best sequence: {optimizer.best_sequence}")
    
    return results