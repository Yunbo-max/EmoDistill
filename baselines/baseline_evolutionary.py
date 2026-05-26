"""
Model 1: Evolutionary Bayesian Optimization
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from baselines.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

# Emotion definitions
EMOTIONS = ['happy', 'surprising', 'angry', 'sad', 'disgust', 'fear', 'neutral']
N_EMOTIONS = len(EMOTIONS)

@dataclass
class EmotionSequence:
    """An emotional sequence with fitness score"""
    sequence: List[str]
    fitness: float = 0.0
    negotiation_result: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.negotiation_result is None:
            self.negotiation_result = {}
    
    def __len__(self):
        return len(self.sequence)

@dataclass 
class EmotionPolicy:
    """An emotion policy with transition matrix and temperature schedule"""
    transition_matrix: np.ndarray  # 7x7 probability matrix  
    temperature_schedule: Tuple[float, float]  # (tau_0, delta)
    elite_sequences: List[EmotionSequence] = None  # Top performing sequences
    
    def __post_init__(self):
        if self.elite_sequences is None:
            self.elite_sequences = []

class BaselineEvolutionaryOptimizer(BaseEmotionModel):
    """
    Evolutionary Bayesian Optimization for emotional transition policies
    """
    
    def __init__(
        self,
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        temperature_options: List[Tuple[float, float]] = None
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Discrete temperature options
        if temperature_options is None:
            self.temperature_options = [
                (0.25, 0.1), (0.25, 0.3),
                (0.5, 0.1), (0.5, 0.3),
                (0.75, 0.1), (0.75, 0.3),
            ]
        
        # Initialize with single base transition matrix (game theory starting point)
        self.base_transition_matrix = self._initialize_base_matrix()
        self.current_policy = EmotionPolicy(self.base_transition_matrix.copy(), (0.5, 0.2))
        
        # Elite sequence population (from negotiations)
        self.elite_sequences = []  # Top 50% sequences from negotiations
        self.sequence_pool = []    # All sequences from current generation
        
        self.generation = 0
        self.best_sequence = None
        self.best_fitness = -float('inf')
        
        # Learning history
        self.fitness_history = []
        self.elite_history = []
        
        # Current state  
        self.current_emotion_idx = EMOTIONS.index('neutral')
        self.emotion_history = []
    
    def _initialize_base_matrix(self) -> np.ndarray:
        """Initialize single base transition matrix using game theory principles"""
        # Game theory inspired initial matrix - balanced but slightly strategic
        base_matrix = np.array([
            # From: happy, surprising, angry, sad, disgust, fear, neutral
            [0.3, 0.2, 0.05, 0.1, 0.05, 0.05, 0.25],  # happy -> tend to stay positive
            [0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1],   # surprising -> versatile
            [0.1, 0.1, 0.2, 0.15, 0.15, 0.1, 0.2],   # angry -> firm but de-escalate
            [0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2],     # sad -> empathetic balance
            [0.1, 0.1, 0.15, 0.15, 0.2, 0.1, 0.2],   # disgust -> professional reset
            [0.15, 0.1, 0.1, 0.2, 0.1, 0.15, 0.2],   # fear -> cautious approach
            [0.15, 0.15, 0.1, 0.15, 0.1, 0.1, 0.25]  # neutral -> balanced options
        ])
        
        # Normalize to ensure valid probabilities
        return base_matrix / base_matrix.sum(axis=1, keepdims=True)
    
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion based on current transition matrix (no debtor emotion needed)"""
        # Get temperature for current round
        tau_0, delta = self.current_policy.temperature_schedule
        round_num = state.get('round', 1)
        temperature = max(0.1, tau_0 * ((1 - delta) ** round_num))
        
        # Sample next emotion from current transition matrix
        probabilities = self.current_policy.transition_matrix[self.current_emotion_idx]
        next_idx = np.random.choice(N_EMOTIONS, p=probabilities)
        next_emotion = EMOTIONS[next_idx]
        
        # Update state
        self.current_emotion_idx = next_idx
        self.emotion_history.append(next_emotion)
        
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
            "temperature": temperature,
            "policy_generation": self.generation,
            "using_best_policy": len(self.elite_sequences) > 0,
            "current_matrix_entropy": self._calculate_matrix_entropy()
        }
    
    def evaluate_policy(self, policy: EmotionPolicy, result: Dict[str, Any]) -> float:
        """Calculate fitness for a policy"""
        success = result.get('final_state') == 'accept'
        collection_days = result.get('collection_days', 0)
        target_days = result.get('creditor_target_days', 30)
        negotiation_rounds = result.get('negotiation_rounds', 1)
        
        if not success:
            return 0.0
        
        # Calculate normalized savings (shorter collection = better)
        if target_days > 0:
            savings = 1.0 - (collection_days / target_days)
        else:
            savings = 0.5
        
        # Apply logarithmic penalty for longer negotiations
        efficiency = savings / (1 + np.log(negotiation_rounds + 1))
        
        # Scale for better numerical properties
        fitness = 100.0 * efficiency
        
        return fitness
    
    def evolve_generation(self, negotiation_results: List[Dict[str, Any]]):
        """Evolve using elite emotional sequences from negotiations"""
        self.generation += 1
        print(f"\n🧬 Generation {self.generation} Evolution:")
        
        # 1. Collect all emotional sequences from negotiations with fitness
        all_sequences = []
        for result in negotiation_results:
            if 'emotion_sequence' in result and result['emotion_sequence']:
                fitness = self.evaluate_sequence_fitness(result)
                sequence = EmotionSequence(
                    sequence=result['emotion_sequence'].copy(),
                    fitness=fitness,
                    negotiation_result=result
                )
                all_sequences.append(sequence)
        
        if not all_sequences:
            print("  ⚠️  No sequences collected, using random generation")
            return
        
        # 2. Sort by fitness and select top 50% ELITE sequences
        all_sequences.sort(key=lambda s: s.fitness, reverse=True)
        elite_count = max(1, len(all_sequences) // 2)  # Top 50%
        self.elite_sequences = all_sequences[:elite_count]
        
        print(f"  🏆 Selected {len(self.elite_sequences)} elite sequences from {len(all_sequences)} total")
        print(f"  📊 Elite fitness range: {self.elite_sequences[0].fitness:.3f} - {self.elite_sequences[-1].fitness:.3f}")
        
        # 3. Update best sequence
        if self.elite_sequences[0].fitness > self.best_fitness:
            self.best_sequence = self.elite_sequences[0]
            self.best_fitness = self.elite_sequences[0].fitness
        
        # 4. Generate new sequences through GENETIC ALGORITHM
        new_sequences = self._generate_offspring_sequences(self.elite_sequences)
        print(f"  🧬 Generated {len(new_sequences)} offspring sequences")
        
        # 5. BAYESIAN UPDATE: Use elite + offspring sequences to update transition matrix  
        all_evolution_sequences = [seq.sequence for seq in self.elite_sequences + new_sequences]
        self.current_policy.transition_matrix = self._bayesian_update_matrix(
            self.current_policy.transition_matrix,
            all_evolution_sequences,
            lambda_param=0.6
        )
        
        print(f"  🎯 Updated transition matrix using {len(all_evolution_sequences)} sequences")
        print(f"  📈 Matrix entropy: {self._calculate_matrix_entropy():.3f}")
        
        # 6. Store evolution history
        self.elite_history.append({
            'generation': self.generation,
            'elite_count': len(self.elite_sequences),
            'best_fitness': self.best_fitness,
            'avg_elite_fitness': np.mean([s.fitness for s in self.elite_sequences]),
            'sequence_lengths': [len(s) for s in self.elite_sequences]
        })
    
    def evaluate_sequence_fitness(self, negotiation_result: Dict[str, Any]) -> float:
        """Evaluate fitness of an emotional sequence based on negotiation outcome"""
        success = negotiation_result.get('final_state') == 'accept'
        collection_days = negotiation_result.get('collection_days', 0)
        target_days = negotiation_result.get('creditor_target_days', 30)
        negotiation_rounds = negotiation_result.get('negotiation_rounds', 1)
        
        if not success:
            return 0.0
        
        # Calculate normalized efficiency (shorter collection time + fewer rounds = better)
        if target_days > 0:
            collection_efficiency = max(0, (target_days - collection_days) / target_days)
        else:
            collection_efficiency = 0.5
        
        # Reward shorter negotiations
        negotiation_efficiency = 1.0 / (1 + np.log(negotiation_rounds + 1))
        
        # Combined fitness with sequence length consideration
        sequence_length = len(negotiation_result.get('emotion_sequence', []))
        length_penalty = 1.0 / (1 + 0.1 * sequence_length)  # Slight penalty for very long sequences
        
        fitness = 100.0 * collection_efficiency * negotiation_efficiency * length_penalty
        return max(0.0, fitness)
    
    def _bayesian_update_matrix(self, old_matrix: np.ndarray, sequences: List[List[str]], 
                               lambda_param: float = 0.6) -> np.ndarray:
        """Bayesian update using elite emotional sequences to refine transition matrix"""
        if not sequences:
            return old_matrix
        
        print(f"    🔄 Bayesian update using {len(sequences)} sequences (lengths: {[len(s) for s in sequences[:5]]}...)")
        
        # Count transitions in elite sequences
        counts = np.zeros((N_EMOTIONS, N_EMOTIONS))
        total_transitions = 0
        
        for seq in sequences:
            for t in range(len(seq) - 1):
                from_idx = EMOTIONS.index(seq[t])
                to_idx = EMOTIONS.index(seq[t+1])
                counts[from_idx, to_idx] += 1
                total_transitions += 1
        
        if total_transitions == 0:
            return old_matrix
        
        # Add Dirichlet smoothing (higher for less data)
        smoothing = max(0.1, 5.0 / total_transitions)
        counts = counts + smoothing
        
        # Calculate MLE from elite sequences
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mle_matrix = counts / row_sums
        
        # Bayesian blending: combine old knowledge with elite sequences
        new_matrix = lambda_param * old_matrix + (1 - lambda_param) * mle_matrix
        
        # Ensure valid probabilities
        new_matrix = new_matrix / new_matrix.sum(axis=1, keepdims=True)
        
        print(f"    📊 Matrix updated: {total_transitions} transitions, λ={lambda_param}")
        return new_matrix
    
    def _select_parents(self) -> Tuple[EmotionPolicy, EmotionPolicy]:
        """Select two parents using fitness-proportionate selection"""
        fitness_values = np.array([max(p.fitness, 0) for p in self.population])
        
        if np.sum(fitness_values) == 0:
            # Uniform selection if all zero fitness
            indices = np.random.choice(len(self.population), size=2, replace=True)
        else:
            # Fitness-proportionate selection
            probabilities = fitness_values / np.sum(fitness_values)
            indices = np.random.choice(len(self.population), size=2, p=probabilities, replace=True)
        
        return self.population[indices[0]], self.population[indices[1]]
    
    def _generate_offspring_sequences(self, elite_sequences: List[EmotionSequence]) -> List[EmotionSequence]:
        """Generate offspring sequences through crossover and mutation"""
        offspring = []
        target_offspring = len(elite_sequences)  # Generate same number as elites
        
        for _ in range(target_offspring):
            if random.random() < self.crossover_rate and len(elite_sequences) >= 2:
                # CROSSOVER: Handle variable-length sequences
                parent1, parent2 = random.sample(elite_sequences, 2)
                child_sequence = self._crossover_variable_length(
                    parent1.sequence, parent2.sequence
                )
            else:
                # REPRODUCTION: Copy with small variation
                parent = random.choice(elite_sequences)
                child_sequence = parent.sequence.copy()
            
            # MUTATION
            child_sequence = self._mutate_sequence(child_sequence)
            
            offspring.append(EmotionSequence(
                sequence=child_sequence,
                fitness=0.0  # Will be evaluated in next generation
            ))
        
        return offspring
    
    def _crossover_variable_length(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Proper crossover for variable-length emotional sequences"""
        if not seq1 or not seq2:
            return seq1 if seq1 else seq2
        
        # Method 1: Random segment crossover
        if random.random() < 0.5:
            # Take random segments from each parent
            cut1 = random.randint(0, len(seq1))
            cut2 = random.randint(0, len(seq2))
            
            child = seq1[:cut1] + seq2[cut2:]
        else:
            # Method 2: Alternating selection with random stops
            child = []
            max_length = max(len(seq1), len(seq2))
            
            for i in range(max_length):
                # Random stopping point (creates variable length)
                if random.random() < 0.15:  # 15% chance to stop
                    break
                
                # Alternate parents, handle out-of-bounds
                if i % 2 == 0 and i < len(seq1):
                    child.append(seq1[i])
                elif i < len(seq2):
                    child.append(seq2[i])
                elif i < len(seq1):
                    child.append(seq1[i])
        
        # Ensure minimum length of 2
        if len(child) < 2:
            child.extend(random.sample(EMOTIONS, 2 - len(child)))
        
        return child
    
    def _mutate(self, policy: EmotionPolicy) -> EmotionPolicy:
        """Apply mutations to policy"""
        mutated = self._clone_policy(policy)
        
        # Mutate transition matrix
        if random.random() < self.mutation_rate:
            for i in range(N_EMOTIONS):
                alpha = policy.transition_matrix[i] * 10 + 0.1
                mutated.transition_matrix[i] = np.random.dirichlet(alpha)
        
        # Mutate temperature
        if random.random() < self.mutation_rate:
            mutated.temperature_schedule = random.choice(self.temperature_options)
        
        # Mutate sequences
        if random.random() < self.mutation_rate and mutated.emotion_sequences:
            seq_idx = random.randint(0, len(mutated.emotion_sequences) - 1)
            if mutated.emotion_sequences[seq_idx]:
                emotion_idx = random.randint(0, len(mutated.emotion_sequences[seq_idx]) - 1)
                mutated.emotion_sequences[seq_idx][emotion_idx] = random.choice(EMOTIONS)
        
        return mutated
    
    def _clone_policy(self, policy: EmotionPolicy) -> EmotionPolicy:
        """Create a deep copy of policy"""
        return EmotionPolicy(
            transition_matrix=policy.transition_matrix.copy(),
            temperature_schedule=policy.temperature_schedule,
            fitness=policy.fitness,
            emotion_sequences=[seq.copy() for seq in policy.emotion_sequences]
        )
    
    def _mutate_sequence(self, sequence: List[str]) -> List[str]:
        """Apply mutations to emotional sequence"""
        mutated = sequence.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Point mutation: change emotion
                mutated[i] = random.choice(EMOTIONS)
        
        # Length mutations
        if random.random() < self.mutation_rate * 0.5:
            if random.random() < 0.5 and len(mutated) > 2:
                # Delete random emotion
                mutated.pop(random.randint(0, len(mutated) - 1))
            else:
                # Insert random emotion
                pos = random.randint(0, len(mutated))
                mutated.insert(pos, random.choice(EMOTIONS))
        
        return mutated
    
    def _calculate_matrix_entropy(self) -> float:
        """Calculate entropy of current transition matrix"""
        entropy = 0.0
        for i in range(N_EMOTIONS):
            row = self.current_policy.transition_matrix[i]
            row_entropy = -np.sum(row * np.log(row + 1e-10))
            entropy += row_entropy
        return entropy / N_EMOTIONS
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Store negotiation result for evolution (sequences collected automatically)"""
        self.sequence_pool.append(negotiation_result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'generation': self.generation,
            'best_fitness': float(self.best_fitness),
            'elite_count': len(self.elite_sequences),
            'avg_elite_fitness': float(np.mean([s.fitness for s in self.elite_sequences])) if self.elite_sequences else 0.0,
            'current_emotion': EMOTIONS[self.current_emotion_idx],
            'emotion_history': self.emotion_history[-10:],
            'matrix_entropy': self._calculate_matrix_entropy(),
            'sequence_pool_size': len(self.sequence_pool)
        }
    
    def reset(self) -> None:
        """Reset model state for new negotiation (keep learned matrix)"""
        self.current_emotion_idx = EMOTIONS.index('neutral')
        self.emotion_history = []
        # Keep elite sequences and updated matrix for continued learning
    
    def save_model(self, filepath: str) -> None:
        """Save trained model for later use"""
        model_data = {
            'base_transition_matrix': self.base_transition_matrix.tolist(),
            'current_policy': {
                'transition_matrix': self.current_policy.transition_matrix.tolist(),
                'temperature_schedule': self.current_policy.temperature_schedule
            },
            'best_sequence': self.best_sequence.sequence if self.best_sequence else None,
            'best_fitness': self.best_fitness,
            'generation': self.generation,
            'elite_sequences': [{
                'sequence': seq.sequence,
                'fitness': seq.fitness
            } for seq in self.elite_sequences[:5]],  # Save top 5
            'model_type': 'baseline_evolutionary',
            'parameters': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'population_size': self.population_size,
                'elite_size': self.elite_size
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Evolutionary model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.base_transition_matrix = np.array(model_data['base_transition_matrix'])
        self.current_policy.transition_matrix = np.array(model_data['current_policy']['transition_matrix'])
        self.current_policy.temperature_schedule = tuple(model_data['current_policy']['temperature_schedule'])
        
        if model_data['best_sequence']:
            # Reconstruct best sequence
            self.best_sequence = EmotionSequence(sequence=model_data['best_sequence'])
            self.best_sequence.fitness = model_data['best_fitness']
        
        self.best_fitness = model_data['best_fitness']
        self.generation = model_data['generation']
        
        # Reconstruct elite sequences
        self.elite_sequences = []
        for seq_data in model_data.get('elite_sequences', []):
            seq = EmotionSequence(sequence=seq_data['sequence'])
            seq.fitness = seq_data['fitness']
            self.elite_sequences.append(seq)
        
        print(f"🔄 Evolutionary model loaded from: {filepath}")
        print(f"   Best fitness: {self.best_fitness:.3f}")
        print(f"   Generation: {self.generation}")
        print(f"   Elite sequences: {len(self.elite_sequences)}")

def run_baseline_experiment(
    scenarios: List[Dict[str, Any]],
    generations: int = 10,
    population_size: int = 20,  # Now used as number of negotiations per generation
    mutation_rate: float = 0.1,  # Mutation rate parameter
    crossover_rate: float = 0.7,  # Crossover rate parameter
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini", 
    debtor_emotion: str = "neutral",
    max_dialog_len: int = 30,
    out_dir: str = "results"
) -> Dict[str, Any]:
    """Run evolutionary Bayesian experiment with sequence-focused approach"""
    
    from llm.negotiator import DebtNegotiator
    
    # Create single optimizer with passed parameters (no population of matrices)
    optimizer = BaselineEvolutionaryOptimizer(
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )
    
    results = {
        'experiment_type': 'evolutionary_bayesian_sequences',
        'generations': generations,
        'negotiations_per_generation': population_size,  # Now means negotiations per generation
        'generation_results': {},
        'scenarios_used': [s['id'] for s in scenarios]
    }
    
    for generation in range(generations):
        print(f"\n🧬 Generation {generation + 1}/{generations}")
        print(f"🎯 Goal: Run {population_size} negotiations to collect emotional sequences")
        
        generation_negotiations = []
        
        # Run multiple negotiations to collect diverse emotional sequences
        for negotiation_idx in range(population_size):
            # Cycle through scenarios
            scenario = scenarios[negotiation_idx % len(scenarios)]
            
            print(f"  🧪 Negotiation {negotiation_idx + 1}/{population_size} - Scenario: {scenario['id']}")
            print(f"        🎬 Starting negotiation (no debtor emotion detection needed)...")
            
            # Create negotiator - optimizer handles emotion selection internally
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=optimizer,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion  # Fixed emotion (not detected in evolutionary)
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            generation_negotiations.append(result)
            
            # Show results
            final_state = result.get('final_state', 'breakdown')
            final_days = result.get('final_days', 'N/A')
            rounds = len(result.get('dialog', []))
            emotion_seq = result.get('emotion_sequence', [])
            
            outcome_emoji = "✅" if final_state == "accept" else "❌"
            print(f"        📊 Result: {outcome_emoji} {final_state} | Days: {final_days} | Rounds: {rounds}")
            print(f"        🎭 Emotion sequence: {emotion_seq}")
            
            # Update model (stores results for evolution)
            optimizer.update_model(result)
        
        # EVOLUTION PHASE: Use collected sequences to evolve
        print(f"\n  🧬 Evolution Phase - Processing {len(generation_negotiations)} negotiations")
        optimizer.evolve_generation(generation_negotiations)
        
        # Calculate generation statistics
        successful_negotiations = [r for r in generation_negotiations if r.get('final_state') == 'accept']
        success_rate = len(successful_negotiations) / len(generation_negotiations)
        
        if successful_negotiations:
            avg_days = np.mean([r.get('final_days', 0) for r in successful_negotiations])
            avg_rounds = np.mean([len(r.get('dialog', [])) for r in successful_negotiations])
        else:
            avg_days = avg_rounds = 0
        
        print(f"  📊 Generation {generation + 1} Summary:")
        print(f"      Success rate: {success_rate:.1%} ({len(successful_negotiations)}/{len(generation_negotiations)})")
        print(f"      Avg days (successful): {avg_days:.1f}")
        print(f"      Avg rounds (successful): {avg_rounds:.1f}")
        print(f"      Best sequence fitness: {optimizer.best_fitness:.3f}")
        print(f"      Elite sequences: {len(optimizer.elite_sequences)}")
        
        # Store generation results
        results['generation_results'][f'generation_{generation+1}'] = {
            'stats': optimizer.get_stats(),
            'success_rate': success_rate,
            'avg_days': avg_days,
            'avg_rounds': avg_rounds,
            'elite_sequences': [
                {
                    'sequence': seq.sequence,
                    'fitness': seq.fitness,
                    'length': len(seq)
                } for seq in optimizer.elite_sequences
            ]
        }
    
    # Collect all negotiation results for statistical analysis
    all_negotiation_results = []
    for generation in range(generations):
        gen_key = f'generation_{generation+1}'
        if gen_key in results['generation_results']:
            # Get negotiations from this generation 
            gen_start = generation * population_size
            gen_end = (generation + 1) * population_size
            # Add the negotiation results from optimizer.sequence_pool
            if gen_end <= len(optimizer.sequence_pool):
                all_negotiation_results.extend(optimizer.sequence_pool[gen_start:gen_end])
    
    # If no results collected above, use all from sequence_pool
    if not all_negotiation_results and optimizer.sequence_pool:
        all_negotiation_results = optimizer.sequence_pool
    
    # Final results
    final_stats = optimizer.get_stats()
    final_stats['base_transition_matrix'] = optimizer.base_transition_matrix.tolist()
    final_stats['current_transition_matrix'] = optimizer.current_policy.transition_matrix.tolist()
    final_stats['temperature_schedule'] = optimizer.current_policy.temperature_schedule
    
    results['final_stats'] = final_stats
    results['final_best_sequence'] = {
        'sequence': optimizer.best_sequence.sequence if optimizer.best_sequence else None,
        'fitness': optimizer.best_fitness,
        'transition_matrix': optimizer.current_policy.transition_matrix.tolist()
    }

    # Create timestamp for all file operations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save trained model
    model_file = f"{out_dir}/trained_evolutionary_model_{timestamp}.json"
    optimizer.save_model(model_file)
    
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
    
    # Save results (using timestamp created earlier)
    result_file = f"{out_dir}/evolutionary_sequences_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save comprehensive summary file (matching vanilla model format)
    summary_file = f"{out_dir}/evolutionary_sequences_summary_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("BASELINE EVOLUTIONARY OPTIMIZATION EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Generations: {generations}\n")
        f.write(f"Negotiations per Generation: {population_size}\n")
        f.write(f"Total Negotiations: {len(all_negotiation_results)}\n")
        f.write(f"Scenarios Used: {len(scenarios)}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Success Rate: {results['overall_success_rate']:.1%}\n")
        f.write(f"Average Days (Successful): {results['avg_successful_days']:.1f}\n")
        f.write(f"Average Rounds (Successful): {results['avg_successful_rounds']:.1f}\n")
        f.write(f"Best Sequence Fitness: {optimizer.best_fitness:.3f}\n\n")
        
        f.write("BEST EMOTIONAL SEQUENCE:\n")
        f.write("-" * 30 + "\n")
        if optimizer.best_sequence:
            f.write(f"Sequence: {optimizer.best_sequence.sequence}\n")
            f.write(f"Length: {len(optimizer.best_sequence.sequence)}\n")
            f.write(f"Fitness: {optimizer.best_sequence.fitness:.3f}\n\n")
        else:
            f.write("No best sequence found\n\n")
        
        f.write("FINAL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        stats = optimizer.get_stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, list) and len(value) <= 10:
                f.write(f"{key}: {value}\n")
        
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
        print("📊 BASELINE EVOLUTIONARY RESULTS WITH 95% CONFIDENCE INTERVALS")
        print("="*80)
        format_ci_results(results['statistical_analysis'])
    
    print(f"\n🏆 Final Results:")
    print(f"   Total negotiations: {len(all_negotiation_results)}")
    print(f"   Overall success rate: {results['overall_success_rate']:.1%}")
    print(f"   Avg successful days: {results['avg_successful_days']:.1f}")
    print(f"   Avg successful rounds: {results['avg_successful_rounds']:.1f}")
    print(f"   Best sequence fitness: {optimizer.best_fitness:.3f}")
    if optimizer.best_sequence:
        print(f"   Best sequence: {optimizer.best_sequence.sequence}")
    
    return results