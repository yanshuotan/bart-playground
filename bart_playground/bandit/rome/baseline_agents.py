
from typing import Callable, Dict, Any, Optional
import numpy as np
from ..agent import BanditAgent
from .baseline_models import StandardTS, ActionCenteredTS, IntelligentPooling

class StandardTSAgent(BanditAgent):
    """
    Wrapper for Standard Thompson sampler compatible with BanditAgent interface.
    This agent considers each individual separately.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 featurize: Optional[Callable] = None,
                 nn: Optional[Callable] = None,
                 nn_dim: Optional[int] = None) -> None:
        """
        Initialize the Standard TS agent.
        
        Parameters:
            n_arms (int): Number of arms. Must be 2 for Standard TS (binary actions 0/1).
            n_features (int): Number of features.
            featurize (Callable, optional): Feature transformation function.
            nn (Callable, optional): Neural network transformation function.
            nn_dim (int, optional): Dimension of neural network output.
        """
        super().__init__(n_arms, n_features)
        
        if n_arms != 2:
            raise ValueError("StandardTSAgent only supports binary actions (n_arms must be 2)")
        
        # Default featurize function if none provided
        if featurize is None:
            featurize = lambda x: x if len(x.shape) > 1 else x.reshape(1, -1)
        
        # Single user for this implementation
        self.n_max = 1
        self.t_max = 1000 # useless for StandardTS
        self.time_idx = 0  # Current time index
        
        self.standard_ts = StandardTS(
            n_max=self.n_max,
            t_max=self.t_max,
            p=n_features+1,  # Adding 1 for intercept
            featurize=featurize,
            nn=nn,
            nn_dim=nn_dim
        )
    
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Standard Thompson sampling.
        
        Parameters:
            x (array-like): Feature vector.
            
        Returns:
            int: The index of the selected arm (0 or 1).
        """
        # Always use user_idx=0 for StandardTS
        user_idx = np.array([0])
        time_idx = np.array([self.time_idx])
        
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Sample action using StandardTS (returns 0.0 or 1.0)
        action = self.standard_ts.sample_action(x, user_idx, time_idx)
        
        # Convert to int for arm index
        return int(action[0])
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "StandardTSAgent":
        """
        Update the agent's state with new observation data.
        
        Parameters:
            arm (int): The index of the arm chosen (0 or 1).
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Empty context_extra for simplicity
        extra_context_dim = 0
        context_extra = np.zeros((1, extra_context_dim))
        
        # Convert arm to action array (0.0 or 1.0)
        action = np.array([float(arm)])
        
        # Default indices
        user_idx = np.array([0])
        time_idx = np.array([self.time_idx])
        
        # Convert reward to array
        reward = np.array([y])
        
        # Update StandardTS state
        self.standard_ts.update(x, context_extra, user_idx, time_idx, action, reward)
        
        # Increment time index for next update
        self.time_idx += 1
        
        return self
    
    
class ActionCenteredTSAgent(BanditAgent):
    """
    Wrapper for Action-Centered Thompson sampler compatible with BanditAgent interface.
    This agent uses the approach described in https://doi.org/10.48550/arXiv.1711.03596
    """
    def __init__(self, n_arms: int, n_features: int, 
                 featurize: Optional[Callable] = None,
                 pi_min: float = 0.0,
                 pi_max: float = 1.0) -> None:
        """
        Initialize the Action-Centered TS agent.
        
        Parameters:
            n_arms (int): Number of arms. Must be 2 for ActionCenteredTS (binary actions 0/1).
            n_features (int): Number of features.
            featurize (Callable, optional): Feature transformation function.
            pi_min (float): Minimum probability for action selection.
            pi_max (float): Maximum probability for action selection.
        """
        super().__init__(n_arms, n_features)
        
        if n_arms != 2:
            raise ValueError("ActionCenteredTSAgent only supports binary actions (n_arms must be 2)")
        
        # Default featurize function if none provided
        if featurize is None:
            featurize = lambda x: x if len(x.shape) > 1 else x.reshape(1, -1)
        
        # Single user for this implementation
        self.n_max = 1
        self.t_max = 1000 # useless for ActionCenteredTS
        self.time_idx = 0  # Current time index
        
        self.actc_ts = ActionCenteredTS(
            n_max=self.n_max,
            t_max=self.t_max,
            p=n_features+1,  # Adding 1 for intercept
            featurize=featurize,
            pi_min=pi_min,
            pi_max=pi_max
        )
    
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Action-Centered Thompson sampling.
        
        Parameters:
            x (array-like): Feature vector.
            
        Returns:
            int: The index of the selected arm (0 or 1).
        """
        # Always use user_idx=0 for ActionCenteredTS
        user_idx = np.array([0])
        time_idx = np.array([self.time_idx])
        
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Sample action using ActionCenteredTS (returns 0.0 or 1.0)
        action = self.actc_ts.sample_action(x, user_idx, time_idx)
        
        # Convert to int for arm index
        return int(action[0])
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "ActionCenteredTSAgent":
        """
        Update the agent's state with new observation data.
        
        Parameters:
            arm (int): The index of the arm chosen (0 or 1).
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Empty context_extra for simplicity
        extra_context_dim = 0
        context_extra = np.zeros((1, extra_context_dim))
        
        # Convert arm to action array (0.0 or 1.0)
        action = np.array([float(arm)])
        
        # Default indices
        user_idx = np.array([0])
        time_idx = np.array([self.time_idx])
        
        # Convert reward to array
        reward = np.array([y])
        
        # Update ActionCenteredTS state
        self.actc_ts.update(x, context_extra, user_idx, time_idx, action, reward)
        
        # Increment time index for next update
        self.time_idx += 1
        
        return self
    
    
class IntelligentPoolingAgent(BanditAgent):
    """
    Wrapper for Intelligent Pooling Thompson sampler compatible with BanditAgent interface.
    This agent uses user and time pooling for more efficient learning.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 featurize: Optional[Callable] = None,
                 sigma: float = 1.0,
                 n_max: int = 1,
                 t_max: int = 1000) -> None:
        """
        Initialize the Intelligent Pooling agent.
        
        Parameters:
            n_arms (int): Number of arms. Must be 2 for IntelligentPooling (binary actions 0/1).
            n_features (int): Number of features.
            featurize (Callable, optional): Feature transformation function.
            sigma (float): Noise standard deviation.
            n_max (int): Maximum number of users.
            t_max (int): Maximum number of time steps.
        """
        super().__init__(n_arms, n_features)
        
        if n_arms != 2:
            raise ValueError("IntelligentPoolingAgent only supports binary actions (n_arms must be 2)")
        
        # Default featurize function if none provided
        if featurize is None:
            featurize = lambda x: x if len(x.shape) > 1 else x.reshape(1, -1)
        
        self.n_max = n_max
        self.t_max = t_max
        self.time_idx = 0  # Current time index
        
        # Default covariance matrices if not provided
        self.P = n_features + 1  # Adding 1 for intercept        # Simple identity covariance matrices
        cov_epsilon = 1e-18
        user_cov = cov_epsilon * np.eye(self.P)
        time_cov = cov_epsilon * np.eye(self.P)
        
        self.intel_pool = IntelligentPooling(
            n_max=self.n_max,
            t_max=self.t_max,
            p=self.P,
            featurize=featurize,
            user_cov=user_cov,
            time_cov=time_cov,
            sigma=sigma
        )
    
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Intelligent Pooling Thompson sampling.
        
        Parameters:
            x (array-like): Feature vector.
            
        Returns:
            int: The index of the selected arm (0 or 1).
        """
        # Extract user_idx from kwargs if provided, otherwise use default
        user_idx = kwargs.get('user_idx', np.array([0]))
        time_idx = kwargs.get('time_idx', np.array([self.time_idx]))
        
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Sample action using IntelligentPooling (returns 0.0 or 1.0)
        action = self.intel_pool.sample_action(x, user_idx, time_idx)
        
        # Convert to int for arm index
        return int(action[0])
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "IntelligentPoolingAgent":
        """
        Update the agent's state with new observation data.
        
        Parameters:
            arm (int): The index of the arm chosen (0 or 1).
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Empty context_extra for simplicity
        extra_context_dim = 0
        context_extra = np.zeros((1, extra_context_dim))
        
        # Convert arm to action array (0.0 or 1.0)
        action = np.array([float(arm)])
        
        # Default indices
        user_idx = np.array([0])
        time_idx = np.array([self.time_idx])
        
        # Convert reward to array
        reward = np.array([y])
        
        # Update IntelligentPooling state
        self.intel_pool.update(x, context_extra, user_idx, time_idx, action, reward)
        
        # Increment time index for next update
        self.time_idx += 1
        
        return self
    