"""
Base model interface for all emotion optimization baselines
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class BaseEmotionModel(ABC):
    """Base class for all emotion optimization baselines"""
    
    @abstractmethod
    def select_emotion(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select next emotion based on current state"""
        pass
    
    @abstractmethod
    def update_model(self, negotiation_result: Dict[str, Any]) -> None:
        """Update model based on negotiation outcome"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset model state"""
        pass