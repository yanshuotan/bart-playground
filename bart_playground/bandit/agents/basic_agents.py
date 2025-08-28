import numpy as np
from typing import List, Optional, Dict, Any, Union
from bart_playground.bandit.agents.agent import BanditAgent

class SillyAgent(BanditAgent):
    """
    A simple agent that selects arms randomly.
    """
    def __init__(self, n_arms: int, n_features: int, random_state: Optional[int] = None) -> None:
        """
        Initialize the agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        super().__init__(n_arms, n_features)
    
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm randomly.
        
        Parameters:
            x (array-like): Feature vector (not used by this agent).
            
        Returns:
            int: The randomly selected arm index.
        """
        return self.rng.integers(0, self.n_arms)
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "SillyAgent":
        """
        Update agent state (no-op for this agent).
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Unchanged instance.
        """
        return self


class LinearTSAgent(BanditAgent):
    """
    Linear Thompson Sampling agent based on Agrawal and Goyal (2012), Appendix C.
    """
    def __init__(self, n_arms: int, n_features: int, v: Optional[float] = None, 
                 eps: float = 0.5, delta: float = 0.2, R: float = 1.0, random_state: Optional[int] = None) -> None:
        """
        Initialize the LinearTS agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            v (float, optional): Exploration parameter.
            eps (float): Epsilon parameter for exploration.
            delta (float): Delta parameter for confidence bound.
            R (float): Bound on the reward.
        """
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        super().__init__(n_arms, n_features + 1)  # +1 for the intercept term
        
        # Set exploration parameter v if not provided
        if v is None:
            v = R * np.sqrt(24 / eps * self.n_features * np.log(1 / delta))
        
        self.v = v
        
        # Initialize matrices for each arm
        self.B = [np.eye(self.n_features) for _ in range(n_arms)]
        self.m2_r = [np.zeros((self.n_features, 1)) for _ in range(n_arms)]
        self.B_inv = [np.eye(self.n_features) for _ in range(n_arms)]
        self.B_inv_sqrt = [np.eye(self.n_features) for _ in range(n_arms)]
        self.mean = [np.zeros((self.n_features, 1)) for _ in range(n_arms)]
    
    def _get_action_estimates(self, x: np.ndarray) -> List[float]:
        """
        Get action estimates for all arms based on input features x.
        """
        x = np.array(x)
        x = np.append(x, 1).reshape(-1, 1)
        
        # Sample parameters from the posterior for each arm
        w = [
            self.mean[i] + self.v * (self.B_inv_sqrt[i] @ self.rng.normal(0, 1, (self.n_features, 1)))
            for i in range(self.n_arms)
        ]
        
        # Compute the expected reward for each arm
        u = [(w[i].T @ x)[0, 0] for i in range(self.n_arms)]
        return u

    def choose_arm(self, x: Union[np.ndarray, List[float]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Thompson Sampling.
        
        Parameters:
            x (array-like): Feature vector.
            
        Returns:
            int: The index of the selected arm.
        """
        u = self._get_action_estimates(x)
        
        # Return the arm with the highest expected reward
        return int(np.argmax(u))
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], y: float) -> "LinearTSAgent":
        """
        Update the agent's state with new observation data.
        
        Parameters:
            arm (int): The index of the arm chosen (1-indexed).
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        arm_idx = arm
        x = np.array(x)
        x = np.append(x, 1).reshape(-1, 1)
        
        # Update the precision matrix
        self.B[arm_idx] = self.B[arm_idx] + x @ x.T
        
        # Update the weighted sum of rewards
        self.m2_r[arm_idx] = self.m2_r[arm_idx] + x * y
        
        # Update the inverse of the precision matrix
        self.B_inv[arm_idx] = np.linalg.inv(self.B[arm_idx])
        
        # Update the square root of the inverse precision matrix (Cholesky decomposition)
        self.B_inv_sqrt[arm_idx] = np.linalg.cholesky(self.B_inv[arm_idx])
        
        # Update the mean vector
        self.mean[arm_idx] = self.B_inv[arm_idx] @ self.m2_r[arm_idx]
        
        return self
