"""
Statistical Analysis Utilities for Negotiation Results
Provides 95% confidence intervals for success rates, collection rates, and negotiation rounds
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Any
import warnings

def bootstrap_ci(data: List[float], stat_func=np.mean, n_bootstrap: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic
    
    Args:
        data: List of data points
        stat_func: Function to calculate statistic (default: np.mean)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)
    
    if len(data) == 1:
        return (data[0], data[0])
    
    data_array = np.array(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data_array, size=len(data), replace=True)
        bootstrap_stats.append(stat_func(bootstrap_sample))
    
    # Calculate percentiles for CI
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return (float(ci_lower), float(ci_upper))


def wilson_score_ci(successes: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion
    More accurate than normal approximation, especially for small samples or extreme proportions
    
    Args:
        successes: Number of successes
        n_total: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n_total == 0:
        return (0.0, 0.0)
    
    if successes == 0:
        return (0.0, 1.96 / np.sqrt(n_total))  # Upper bound for 0 successes
    
    if successes == n_total:
        return (1.0 - 1.96 / np.sqrt(n_total), 1.0)  # Lower bound for all successes
    
    p_hat = successes / n_total
    z = stats.norm.ppf(1 - alpha/2)  # 1.96 for 95% CI
    
    center = (p_hat + z**2/(2*n_total)) / (1 + z**2/n_total)
    margin = z * np.sqrt(p_hat*(1-p_hat)/n_total + z**2/(4*n_total**2)) / (1 + z**2/n_total)
    
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    
    return (float(ci_lower), float(ci_upper))


def analyze_negotiation_results(
    success_list: List[int],          # 0/1 success indicators
    collection_rates: List[float],    # 0-1 continuous rates
    rounds_list: List[int],           # 1-30 integer rounds
    method: str = "bootstrap"         # 'normal', 'bootstrap', or 'auto'
) -> Dict[str, Any]:
    """
    Comprehensive analysis of negotiation results with 95% confidence intervals
    
    Args:
        success_list: List of 0/1 success indicators
        collection_rates: List of collection rates (0-1)
        rounds_list: List of negotiation rounds (integers)
        method: 'normal' for t-distribution, 'bootstrap' for bootstrap CI, 'auto' for best choice
    
    Returns:
        Dictionary with comprehensive statistics and 95% confidence intervals
    """
    n = len(success_list)
    
    if n == 0:
        return {
            "success_rate": {"mean": 0.0, "std": 0.0, "ci_95": (0.0, 0.0), "n_success": 0, "n_total": 0},
            "collection_rate": {"mean": 0.0, "std": 0.0, "ci_95": (0.0, 0.0), "min": 0.0, "max": 0.0},
            "negotiation_rounds": {"mean": 0.0, "std": 0.0, "ci_95": (0.0, 0.0), "min": 0, "max": 0, "median": 0.0}
        }
    
    # ===== 1. SUCCESS RATE ANALYSIS (Binomial) =====
    successes = sum(success_list)
    p_hat = successes / n
    
    # Use Wilson score interval (better than normal approximation)
    success_ci = wilson_score_ci(successes, n)
    success_std = np.sqrt(p_hat * (1 - p_hat))  # Binomial standard error
    
    # ===== 2. COLLECTION RATE ANALYSIS (Continuous) =====
    if len(collection_rates) == 0:
        collection_rates = [0.0]  # Fallback
    
    mean_cr = np.mean(collection_rates)
    std_cr = np.std(collection_rates, ddof=1) if len(collection_rates) > 1 else 0.0
    
    # Choose CI method
    if method == "bootstrap" or (method == "auto" and n >= 30):
        cr_ci = bootstrap_ci(collection_rates, np.mean)
    else:
        # t-distribution CI for small samples
        if n > 1 and std_cr > 0:
            cr_ci = stats.t.interval(0.95, n-1, mean_cr, std_cr/np.sqrt(n))
        else:
            cr_ci = (mean_cr, mean_cr)
    
    # Bound collection rates to [0, 1]
    cr_ci = (max(0.0, cr_ci[0]), min(1.0, cr_ci[1]))
    
    # ===== 3. NEGOTIATION ROUNDS ANALYSIS (Integer/Count) =====
    if len(rounds_list) == 0:
        rounds_list = [0]  # Fallback
    
    mean_rounds = np.mean(rounds_list)
    std_rounds = np.std(rounds_list, ddof=1) if len(rounds_list) > 1 else 0.0
    median_rounds = np.median(rounds_list)
    
    # Choose CI method for rounds
    if method == "bootstrap" or (method == "auto" and n >= 30):
        rounds_ci = bootstrap_ci(rounds_list, np.mean)
    else:
        # t-distribution CI
        if n > 1 and std_rounds > 0:
            rounds_ci = stats.t.interval(0.95, n-1, mean_rounds, std_rounds/np.sqrt(n))
        else:
            rounds_ci = (mean_rounds, mean_rounds)
    
    # Bound rounds to reasonable range [1, 50]
    rounds_ci = (max(1.0, rounds_ci[0]), min(50.0, rounds_ci[1]))
    
    return {
        "success_rate": {
            "mean": float(p_hat),
            "std": float(success_std),
            "ci_95": success_ci,
            "n_success": int(successes),
            "n_total": int(n),
            "method": "wilson_score"
        },
        "collection_rate": {
            "mean": float(mean_cr),
            "std": float(std_cr),
            "ci_95": cr_ci,
            "min": float(np.min(collection_rates)) if collection_rates else 0.0,
            "max": float(np.max(collection_rates)) if collection_rates else 0.0,
            "median": float(np.median(collection_rates)) if collection_rates else 0.0,
            "method": method if method != "auto" else ("bootstrap" if n >= 30 else "t_distribution")
        },
        "negotiation_rounds": {
            "mean": float(mean_rounds),
            "std": float(std_rounds),
            "ci_95": rounds_ci,
            "min": int(np.min(rounds_list)) if rounds_list else 0,
            "max": int(np.max(rounds_list)) if rounds_list else 0,
            "median": float(median_rounds),
            "method": method if method != "auto" else ("bootstrap" if n >= 30 else "t_distribution")
        },
        "sample_size": int(n),
        "analysis_method": method
    }


def format_ci_results(results: Dict[str, Any]) -> str:
    """
    Format confidence interval results for pretty printing
    
    Args:
        results: Results from analyze_negotiation_results()
    
    Returns:
        Formatted string for console output
    """
    sr = results["success_rate"]
    cr = results["collection_rate"] 
    nr = results["negotiation_rounds"]
    
    output = []
    output.append("📊 STATISTICAL ANALYSIS (95% Confidence Intervals)")
    output.append("=" * 60)
    
    # Success Rate
    output.append(f"🎯 Success Rate:")
    output.append(f"   Mean: {sr['mean']:.1%} ({sr['n_success']}/{sr['n_total']})")
    output.append(f"   95% CI: [{sr['ci_95'][0]:.1%}, {sr['ci_95'][1]:.1%}] ({sr['method']})")
    
    # Collection Rate
    output.append(f"💰 Collection Rate:")
    output.append(f"   Mean: {cr['mean']:.3f} ± {cr['std']:.3f}")
    output.append(f"   95% CI: [{cr['ci_95'][0]:.3f}, {cr['ci_95'][1]:.3f}] ({cr['method']})")
    output.append(f"   Range: [{cr['min']:.3f}, {cr['max']:.3f}]")
    
    # Negotiation Rounds
    output.append(f"🔄 Negotiation Rounds:")
    output.append(f"   Mean: {nr['mean']:.1f} ± {nr['std']:.1f}")
    output.append(f"   95% CI: [{nr['ci_95'][0]:.1f}, {nr['ci_95'][1]:.1f}] ({nr['method']})")
    output.append(f"   Range: [{nr['min']}, {nr['max']}], Median: {nr['median']:.1f}")
    
    output.append(f"📈 Sample Size: {results['sample_size']} negotiations")
    
    return "\n".join(output)


def extract_results_from_negotiations(all_negotiations: List[Dict[str, Any]], 
                                     scenarios: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[int]]:
    """
    Extract success list, collection rates, and rounds from negotiation results
    
    Args:
        all_negotiations: List of negotiation result dictionaries
        scenarios: List of scenario configurations (for target collection rates)
    
    Returns:
        Tuple of (success_list, collection_rates, rounds_list)
    """
    success_list = []
    collection_rates = []
    rounds_list = []
    
    for i, negotiation in enumerate(all_negotiations):
        # Success (0/1)
        is_success = 1 if negotiation.get('final_state') == 'accept' else 0
        success_list.append(is_success)
        
        # Rounds
        rounds = negotiation.get('negotiation_rounds', 0)
        rounds_list.append(int(rounds))
        
        # Collection rate (only for successful negotiations)
        if is_success:
            final_days = negotiation.get('collection_days')
            if final_days and final_days > 0:
                # Get target from corresponding scenario
                scenario_idx = i % len(scenarios) if scenarios else 0
                target_days = float(scenarios[scenario_idx]['seller']['target_price']) if scenarios else 30.0
                
                # Calculate collection rate as target/actual (higher is better)
                collection_rate = min(1.0, target_days / final_days)
                collection_rates.append(collection_rate)
            else:
                collection_rates.append(0.0)  # Failed to get payment details
        else:
            collection_rates.append(0.0)  # Failed negotiation = 0 collection rate
    
    return success_list, collection_rates, rounds_list


def enhance_results_with_statistics(results: Dict[str, Any], 
                                   all_negotiations: List[Dict[str, Any]],
                                   scenarios: List[Dict[str, Any]] = None,
                                   method: str = "bootstrap") -> Dict[str, Any]:
    """
    Enhance existing results dictionary with comprehensive statistical analysis
    
    Args:
        results: Existing results dictionary
        all_negotiations: List of negotiation results
        scenarios: List of scenarios (for target collection rates)
        method: Statistical method ('bootstrap', 'normal', 'auto')
    
    Returns:
        Enhanced results dictionary with statistical analysis
    """
    if not all_negotiations:
        return results
    
    # Extract data
    success_list, collection_rates, rounds_list = extract_results_from_negotiations(
        all_negotiations, scenarios or []
    )
    
    # Perform statistical analysis
    statistical_analysis = analyze_negotiation_results(
        success_list, collection_rates, rounds_list, method
    )
    
    # Add to existing results
    results['statistical_analysis'] = statistical_analysis
    results['confidence_intervals'] = {
        'success_rate_95_ci': statistical_analysis['success_rate']['ci_95'],
        'collection_rate_95_ci': statistical_analysis['collection_rate']['ci_95'],
        'negotiation_rounds_95_ci': statistical_analysis['negotiation_rounds']['ci_95']
    }
    
    # Update summary_statistics with CI information
    if 'summary_statistics' in results:
        results['summary_statistics'].update({
            'success_rate_ci': statistical_analysis['success_rate']['ci_95'],
            'collection_rate_ci': statistical_analysis['collection_rate']['ci_95'],
            'negotiation_rounds_ci': statistical_analysis['negotiation_rounds']['ci_95'],
            'statistical_method': statistical_analysis['analysis_method']
        })
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Sample negotiation results
    success_list = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]  # 66.7% success
    collection_rates = [0.8, 0.9, 0.0, 0.7, 0.0, 0.85, 0.75, 0.0, 0.6, 0.95, 0.0, 0.8, 0.9, 0.7, 0.0]
    rounds_list = [5, 8, 15, 6, 20, 7, 9, 25, 4, 6, 30, 8, 5, 10, 18]
    
    # Analyze results
    print("Testing with sample data:")
    results = analyze_negotiation_results(success_list, collection_rates, rounds_list, method="bootstrap")
    print(format_ci_results(results))
    
    # Compare methods
    print("\n" + "="*60)
    print("METHOD COMPARISON:")
    
    for method in ["normal", "bootstrap"]:
        print(f"\n{method.upper()} METHOD:")
        results = analyze_negotiation_results(success_list, collection_rates, rounds_list, method=method)
        print(f"  Success Rate CI: [{results['success_rate']['ci_95'][0]:.1%}, {results['success_rate']['ci_95'][1]:.1%}]")
        print(f"  Collection Rate CI: [{results['collection_rate']['ci_95'][0]:.3f}, {results['collection_rate']['ci_95'][1]:.3f}]")
        print(f"  Rounds CI: [{results['negotiation_rounds']['ci_95'][0]:.1f}, {results['negotiation_rounds']['ci_95'][1]:.1f}]")