"""
Utility functions
"""

import json
import os
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

# def load_scenarios(filepath: str) -> List[Dict[str, Any]]:
#     """Load scenarios from JSON file"""
#     try:
#         with open(filepath, 'r') as f:
#             return json.load(f)
#     except FileNotFoundError:
#         print(f"⚠️ Scenarios file not found: {filepath}")
#         return []


def load_scenarios(file_path: str = None, csv_path: str = None, 
                  scenario_type: str = None, n_scenarios: int = None) -> List[Dict[str, Any]]:
    """
    Load scenarios from either JSON file or CSV
    
    Args:
        file_path: Path to JSON file (if scenarios already processed)
        csv_path: Path to CSV file (if need to preprocess)
        scenario_type: Type of scenarios ("debt", "disaster", "student", "medical", or "auto")
        n_scenarios: Maximum number of scenarios to load
    
    Returns:
        List of scenario dictionaries
    """
    import os
    
    if file_path and os.path.exists(file_path):
        # Load from existing JSON
        with open(file_path, 'r') as f:
            scenarios = json.load(f)
        
        if n_scenarios:
            scenarios = scenarios[:n_scenarios]
        
        print(f"Loaded {len(scenarios)} scenarios from {file_path}")
        return scenarios
    
    elif csv_path and os.path.exists(csv_path):
        # Preprocess from CSV
        from .preprocessing import preprocess_all_scenarios  # Import the preprocessing function
        
        scenarios = preprocess_all_scenarios(
            csv_path=csv_path,
            scenario_type=scenario_type or "auto",
            output_path=None,  # Don't save, just return
            n_scenarios=n_scenarios
        )
        
        print(f"Preprocessed {len(scenarios)} scenarios from {csv_path}")
        return scenarios
    
    else:
        # Create default scenarios if none provided
        print("⚠️ No scenario file provided, creating default scenarios")
        return [
            {
                "id": "default_001",
                "product": {
                    "type": "debt_collection",
                    "amount": 15000
                },
                "seller": {
                    "target_price": 30,
                    "min_price": 20,
                    "max_price": 60
                },
                "buyer": {
                    "target_price": 90,
                    "min_price": 60,
                    "max_price": 120
                },
                "metadata": {
                    "outstanding_balance": 15000,
                    "creditor_name": "Default Creditor",
                    "debtor_name": "Default Debtor",
                    "recovery_stage": "Early"
                }
            }
        ]

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=json_serializer)

def json_serializer(obj):
    """Custom JSON serializer for numpy arrays and other types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def calculate_success_rate(results: List[Dict[str, Any]]) -> float:
    """Calculate success rate from negotiation results"""
    if not results:
        return 0.0
    
    successful = [r for r in results if r.get('final_state') == 'accept']
    return len(successful) / len(results)

def calculate_avg_days(results: List[Dict[str, Any]]) -> float:
    """Calculate average collection days (successful negotiations only)"""
    successful = [r for r in results if r.get('final_state') == 'accept' and r.get('collection_days')]
    if not successful:
        return 0.0
    
    return np.mean([r['collection_days'] for r in successful])

def create_results_dir(base_dir: str = "results") -> str:
    """Create results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{base_dir}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir