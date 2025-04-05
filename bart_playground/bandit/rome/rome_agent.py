from typing import Dict, Any, Callable, Optional
import numpy as np
import scipy.sparse
from ..agent import BanditAgent
from .standalone_rome import RoME, RiverBatchEstimator, BaggingRegressor
from river.tree import SGTRegressor
from river.tree.splitter import DynamicQuantizer

def _featurize(context: np.ndarray) -> np.ndarray:
        """Static method that creates feature vectors from the given context
        
        Parameters
        ----------
        context: np.ndarray
            1d or 2d array of context vector(s)

        Returns
        -------
        np.ndarray
            2d array of feature vectors of shape (n_obs, p)
        """
        if len(context.shape) == 1:
            context_dim = context.shape[0]
        elif len(context.shape) == 2:
            context_dim = context.shape[1]
        else:
            raise ValueError("context must be 1d or 2d")
        context = context.reshape((-1, context_dim))
        n_obs = context.shape[0]
        features = np.ones((n_obs, 1 + context_dim))
        features[:, 1:] = context
        return features

class RoMEAgent(BanditAgent):
    """
    Wrapper for RoME (Robust Mixed-Effects) Thompson sampler compatible with BanditAgent interface.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 featurize: Optional[Callable] = None, 
                 lambda_penalty: float = 1.0,
                 ml_interactions: bool = False, 
                 v: float = 1.0,
                 delta: float = 0.01,
                 zeta: float = 10.0,
                 n_neighbors: int = 1,
                 t_max: int = 1000,
                 pool_users: bool = True) -> None:
        """
        Initialize the RoME agent.
        
        Parameters:
            n_arms (int): Number of arms. Must be 2 for RoME (binary actions 0/1).
            n_features (int): Number of features.
            featurize (Callable, optional): Feature transformation function.
            lambda_penalty (float): Regularization parameter.
            ml_interactions (bool): Whether to use interactions in ML model.
            v (float): Confidence parameter.
            delta (float): Confidence parameter.
            zeta (float): Exploration constant.
            n_neighbors (int): Number of neighbors to use in graph Laplacian.
            pool_users (bool): Whether to pool user effects.
        """
        super().__init__(n_arms, n_features)
        
        if n_arms != 2:
            raise ValueError("RoMEAgent only supports binary actions (n_arms must be 2)")
        
        # Default featurize function if none provided
        if featurize is None:
            featurize = lambda x: x if len(x.shape) > 1 else x.reshape(1, -1)
        self.P = n_features + 1  # Adding 1 for the intercept term
        
        # Create dummy Laplacian matrices (no user/time relations by default)
        self.n_max = 1
        self.t_max = t_max
        self.time_idx = 0  # Current time index
        
        L_user = scipy.sparse.csr_matrix(np.zeros((self.n_max, self.n_max)))
        L_time = scipy.sparse.csr_matrix(np.zeros((self.t_max, self.t_max)))
        
        # Simple identity covariance matrices
        cov_epsilon = 1e-18
        user_cov = cov_epsilon * np.eye(self.P)
        time_cov = cov_epsilon * np.eye(self.P)
        
        # Default ML model
        ml_model = RiverBatchEstimator(
            BaggingRegressor(
                SGTRegressor(delta=0.05, grace_period=50, 
                             feature_quantizer=DynamicQuantizer(), 
                             lambda_value=0., gamma=0.), 
                n_models=100, subsample=0.8)
        )
        
        self.rome = RoME(
            n_max=self.n_max,
            t_max=self.t_max,
            p=self.P,
            featurize=featurize,
            L_user=L_user,
            L_time=L_time,
            user_cov=user_cov,
            time_cov=time_cov,
            # lambda_penalty=lambda_penalty,
            # ml_interactions=ml_interactions,
            # ml_model=ml_model,
            # v=v,
            # delta=delta,
            # zeta=zeta,
            # n_neighbors=n_neighbors,
            pool_users=pool_users
        )
    
    def choose_arm(self, x: np.ndarray, **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using RoME Thompson sampling.
        
        Parameters:
            x (array-like): Feature vector.
            
        Returns:
            int: The index of the selected arm (0 or 1).
        """
        # Extract user_idx and time_idx from kwargs if provided, otherwise use defaults
        user_idx = kwargs.get('user_idx', np.array([0]))
        time_idx = kwargs.get('time_idx', np.array([self.time_idx]))
        
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Sample action using RoME (returns 0.0 or 1.0)
        action = self.rome.sample_action(x, user_idx, time_idx)
        
        # Convert to int for arm index
        return int(action[0])
    
    def update_state(self, arm: int, x: np.ndarray, y: float) -> "RoMEAgent":
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
        context_extra = np.random.uniform(low=-1., high=1., size=(self.n_max, self.t_max, extra_context_dim))[np.array([0]), np.array([self.time_idx])]
        
        # Convert arm to action array (0.0 or 1.0)
        action = np.array([float(arm)])
        
        # Default indices
        user_idx = np.array([0])
        time_idx = np.array([self.time_idx])
        
        # Convert reward to array
        reward = np.array([y])
        
        # Update RoME state
        self.rome.update(x, context_extra, user_idx, time_idx, action, reward)
        
        # Increment time index for next update
        self.time_idx += 1
        
        return self
    