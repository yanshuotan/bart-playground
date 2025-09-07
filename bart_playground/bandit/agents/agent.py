from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)
    
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

AgentSpec = Tuple[str, type, Dict[str, Any]]

def instantiate_agents(agent_specs: List[Tuple[str, type, Dict]], 
                              n_arms: int, n_features: int, 
                              random_state: Optional[int] = None) -> List[BanditAgent]:
    """Create fresh agent instances using the same pattern as compare_agents.py"""
    agents = []
    for name, cls, base_kwargs in agent_specs:
        kwargs = base_kwargs.copy()
        kwargs['n_arms'] = n_arms
        kwargs['n_features'] = n_features
        
        # If needed, add random_state to the kwargs
        if 'need_random_state' in base_kwargs:
            if random_state is not None:
                kwargs['random_state'] = random_state
                logger.info(f"Individual random_state provided for {name}. Using random_state {random_state}.")
            else:
                raise ValueError(f"Individual random_state not provided for {name} but need_random_state is True.")
        else:
            logger.info(f"Individual random_state not provided for {name}. This agent will use its own random policy.")

        # Remove control-only flags that shouldn't be passed to constructors
        kwargs.pop('need_random_state', None)
        agents.append(cls(**kwargs))
    return agents
