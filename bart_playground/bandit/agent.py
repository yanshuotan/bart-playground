from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BanditAgent(ABC):
    """
    Abstract base class for bandit agents.
    """
    def __init__(self, n_arms: int, n_features: int) -> None:
        """
        Initialize the agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
        """
        self.n_arms = n_arms
        self.n_features = n_features
    
    @abstractmethod
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm based on input features x.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm.
            
        Returns:
            int: The index of the selected arm.
        """
        pass
    
    @abstractmethod
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "BanditAgent":
        """
        Update the agent's state after observing a reward.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        pass
