"""
Vanilla Baseline Model - Simple Two-Agent Negotiation
No emotion optimization, just direct LLM-to-LLM conversation
"""

import numpy as np
from typing import Dict, List, Any
from baselines.base_model import BaseEmotionModel
import json
from datetime import datetime
import os
import time
from utils.statistical_analysis import enhance_results_with_statistics, format_ci_results

# GPU memory monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return "GPU not available"
    
    allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
    cached = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    free = total - allocated
    
    return f"GPU Memory: {allocated:.2f}GB used, {free:.2f}GB free, {cached:.2f}GB cached (Total: {total:.2f}GB)"

class VanillaBaselineModel(BaseEmotionModel):
    """
    Vanilla baseline model - pure LLM-to-LLM negotiation.
    No emotion selection, no optimization, just direct context-based conversation.
    """
    
    def __init__(self):
        self.negotiation_count = 0
        self.success_history = []
        
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        For vanilla baseline, return minimal info - no emotion selection at all
        Just let LLMs negotiate based on pure context
        """
        # Track basic stats but don't actually select emotions
        self.negotiation_count += 1
        
        # Return minimal response - no emotion guidance
        return {
            "emotion": None,  # No emotion selection
            "emotion_text": None,  # No emotion prompt
            "temperature": 0.7,  # Standard temperature
            "strategy": "vanilla_llm_only",
            "use_emotion": False  # Flag to skip emotion processing
        }
    
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """
        Vanilla model doesn't learn - just tracks basic stats
        """
        # Store basic outcome info
        success = negotiation_result.get('final_state') == 'accept'
        self.success_history.append(success)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic model statistics"""
        success_rate = 0.0
        if self.success_history:
            success_rate = sum(self.success_history) / len(self.success_history)
            
        return {
            'model_type': 'vanilla_llm_only',
            'negotiation_count': self.negotiation_count,
            'success_rate': success_rate,
            'total_negotiations': len(self.success_history),
            'no_emotion_selection': True,
            'pure_llm_negotiation': True
        }
    
    def reset(self) -> None:
        """Reset for new scenario (keep learning history)"""
        # Nothing to reset - no emotional state
        pass

def run_vanilla_baseline_experiment(
    scenarios: List[Dict[str, Any]],
    iterations: int = 5,
    model_creditor: str = "gpt-4o-mini",
    model_debtor: str = "gpt-4o-mini",
    debtor_emotion: str = "neutral",
    debtor_model_type: str = "vanilla",
    max_dialog_len: int = 30,
    out_dir: str = "results/vanilla_baseline"
) -> Dict[str, Any]:
    """
    Run vanilla baseline experiment - simple two-agent negotiation
    """
    
    from llm.negotiator import DebtNegotiator
    
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Create vanilla model
    model = VanillaBaselineModel()
    
    results = {
        'experiment_type': 'vanilla_baseline',
        'model_type': 'vanilla_baseline',
        'iterations': iterations,
        'scenarios_used': [s['id'] if 'id' in s else f"scenario_{i}" for i, s in enumerate(scenarios)],
        'config': {
            'model_creditor': model_creditor,
            'model_debtor': model_debtor,
            'debtor_emotion': debtor_emotion,
            'max_dialog_len': max_dialog_len
        },
        'iteration_results': {}
    }
    
    all_negotiation_results = []
    
    print(f"🎯 Starting Vanilla Baseline Experiment")
    print(f"📊 Scenarios: {len(scenarios)}, Iterations: {iterations}")
    print(f"🤖 Models: {model_creditor} vs {model_debtor}")
    print(f"� Strategy: Pure LLM negotiation (no emotion selection)")
    print("-" * 50)
    
    for iteration in range(iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
        
        iteration_results = {
            'iteration': iteration + 1,
            'scenario_results': [],
            'model_stats': model.get_stats()
        }
        
        for i, scenario in enumerate(scenarios):
            print(f"  📋 Scenario {i+1}/{len(scenarios)}: ", end="")
            
            # Create negotiator with vanilla model
            negotiator = DebtNegotiator(
                config=scenario,
                emotion_model=model,
                model_creditor=model_creditor,
                model_debtor=model_debtor,
                debtor_emotion=debtor_emotion,
                debtor_model_type=debtor_model_type
            )
            
            # Run negotiation
            result = negotiator.run_negotiation(max_dialog_len=max_dialog_len)
            
            # Print quick result
            if result.get('final_state') == 'accept':
                days = result.get('collection_days', 0)
                amount = result.get('final_amount', 0)
                print(f"✅ Success (${amount:.0f} in {days} days)")
            else:
                print(f"❌ Failed ({result.get('final_state', 'unknown')})")
            
            iteration_results['scenario_results'].append(result)
            all_negotiation_results.append(result)
            
            # Update model (minimal for vanilla)
            model.update_model(result)
            
            # Clean up GPU memory after each negotiation (important for offline baselines)
            try:
                print(f"🧹 Starting cleanup for negotiation {i+1}...")
                negotiator.cleanup_models()
                print(f"✅ Cleanup completed successfully")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
                print(f"🔄 Attempting emergency GPU cleanup...")
                try:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"✅ Emergency cleanup completed")
                except Exception as e2:
                    print(f"❌ Emergency cleanup failed: {e2}")
           
            # Print GPU memory status after cleanup
            gpu_info = get_gpu_memory_info()
            print(f"🖥️ {gpu_info}")
            
            # Skip sleep for API-based baselines (no GPU needed)
            # print(f"⏳ Waiting 3 seconds for GPU memory to stabilize...")
            # time.sleep(3)
     
          
        
        # Store iteration results
        results['iteration_results'][f'iteration_{iteration+1}'] = iteration_results
    
    # Calculate final performance metrics
    successful = [r for r in all_negotiation_results if r.get('final_state') == 'accept']
    failed = [r for r in all_negotiation_results if r.get('final_state') != 'accept']
    
    if all_negotiation_results:
        success_rate = len(successful) / len(all_negotiation_results)
        
        # Calculate collection rates for successful negotiations
        collection_rates = []
        for r in successful:
            target_days = r.get('creditor_target_days', 30)
            actual_days = r.get('collection_days', target_days)
            if target_days > 0:
                collection_rate = min(1.0, target_days / actual_days)  # Higher is better
                collection_rates.append(collection_rate)
        
        avg_collection_rate = np.mean(collection_rates) if collection_rates else 0.0
        avg_rounds = np.mean([len(r.get('dialog', [])) for r in all_negotiation_results])
    else:
        success_rate = 0
        avg_collection_rate = 0.0
        avg_rounds = 0
    
    # Final statistics
    results['final_stats'] = model.get_stats()
    results['performance'] = {
        'success_rate': success_rate,
        'avg_collection_rate': float(avg_collection_rate),
        'avg_negotiation_rounds': float(avg_rounds),
        'total_negotiations': len(all_negotiation_results),
        'successful_negotiations': len(successful),
        'failed_negotiations': len(failed)
    }
    
    # Detailed breakdown
    failure_reasons = {}
    for result in failed:
        reason = result.get('final_state', 'unknown')
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    
    results['analysis'] = {
        'failure_breakdown': failure_reasons,
        'success_patterns': {
            'avg_rounds_successful': float(np.mean([len(r.get('dialog', [])) for r in successful])) if successful else 0,
            'avg_rounds_failed': float(np.mean([len(r.get('dialog', [])) for r in failed])) if failed else 0
        }
    }
    
    # ===== ADD STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS =====
    results = enhance_results_with_statistics(
        results, 
        all_negotiation_results, 
        scenarios, 
        method="bootstrap"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"{out_dir}/vanilla_baseline_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # ===== PRINT RESULTS WITH CONFIDENCE INTERVALS =====
    print("\n" + "="*80)
    print("📊 VANILLA BASELINE RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    # Print statistical analysis with CIs
    if 'statistical_analysis' in results:
        print(format_ci_results(results['statistical_analysis']))
        print("="*80)
    
    # Save human-readable summary
    summary_file = f"{out_dir}/summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("VANILLA BASELINE EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment Type: Vanilla Baseline (Pure LLM Negotiation)\n")
        f.write(f"Total Negotiations: {len(all_negotiation_results)}\n")
        
        # Add CI information to summary
        if 'statistical_analysis' in results:
            stat_analysis = results['statistical_analysis']
            sr_ci = stat_analysis['success_rate']['ci_95']
            cr_ci = stat_analysis['collection_rate']['ci_95']
            nr_ci = stat_analysis['negotiation_rounds']['ci_95']
            
            f.write(f"Success Rate: {stat_analysis['success_rate']['mean']:.1%} ")
            f.write(f"(95% CI: [{sr_ci[0]:.1%}, {sr_ci[1]:.1%}])\n")
            f.write(f"Collection Rate: {stat_analysis['collection_rate']['mean']:.3f} ")
            f.write(f"(95% CI: [{cr_ci[0]:.3f}, {cr_ci[1]:.3f}])\n")
            f.write(f"Negotiation Rounds: {stat_analysis['negotiation_rounds']['mean']:.1f} ")
            f.write(f"(95% CI: [{nr_ci[0]:.1f}, {nr_ci[1]:.1f}])\n\n")
        else:
            f.write(f"Success Rate: {success_rate:.1%}\n")
            f.write(f"Average Collection Rate: {avg_collection_rate:.3f}\n")
            f.write(f"Average Rounds: {avg_rounds:.1f}\n\n")
        
        f.write("FAILURE BREAKDOWN:\n")
        for reason, count in failure_reasons.items():
            f.write(f"  {reason}: {count} ({count/len(failed)*100:.1f}%)\n")
        
        f.write(f"\nModel Configuration:\n")
        f.write(f"  Creditor Model: {model_creditor}\n")
        f.write(f"  Debtor Model: {model_debtor}\n")
        f.write(f"  Emotion Strategy: None (Pure LLM)\n")
        f.write(f"  Max Dialog Length: {max_dialog_len}\n")
    
    print(f"\n📊 VANILLA BASELINE RESULTS")
    print("=" * 50)
    print(f"✅ Success Rate: {success_rate:.1%}")
    print(f"📈 Avg Collection Rate: {avg_collection_rate:.3f}")
    print(f"💬 Avg Negotiation Rounds: {avg_rounds:.1f}")
    print(f"📁 Results saved to: {result_file}")
    print(f"📄 Summary saved to: {summary_file}")
    
    return results