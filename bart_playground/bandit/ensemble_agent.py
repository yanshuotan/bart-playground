from tkinter.tix import Tree
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .agent import BanditAgent
from .basic_agents import LinearTSAgent
from .bcf_agent import BCFAgent

class EnsembleAgent(BanditAgent):
    """
    Ensemble agent that automatically selects between BCFAgent and LinearTSAgent
    using Thompson Sampling with MSE-based performance tracking.
    """
    def __init__(self, n_arms: int, n_features: int,
                window_size: int = 20,
                initial_phase_size: int = 20,
                bcf_kwargs: Optional[Dict[str, Any]] = None,
                linear_ts_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the ensemble agent.
        
        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features.
            exploration_prob (float): Probability of random exploration.
            window_size (int): Size of window for performance evaluation.
            initial_phase_size (int): Number of initial observations before performance comparison.
            bcf_kwargs (dict): Parameters for BCFAgent.
            linear_ts_kwargs (dict): Parameters for LinearTSAgent.
        """
        super().__init__(n_arms, n_features)
        
        # Handle default parameters
        bcf_kwargs = bcf_kwargs or {}
        linear_ts_kwargs = linear_ts_kwargs or {}
        
        # Initialize the component agents
        self.bcf_agent = BCFAgent(n_arms, n_features, **bcf_kwargs)
        self.linear_ts_agent = LinearTSAgent(n_arms, n_features, **linear_ts_kwargs)
        
        # Settings
        self.window_size = window_size
        self.initial_phase_size = initial_phase_size
        
        # Track performance history (predictions, chosen arms, observed rewards)
        self.features = []
        self.chosen_arms = []
        self.observed_rewards = []
        
        # Track predicted rewards for proper MSE calculation
        self.bcf_predicted_rewards = []
        self.linear_ts_predicted_rewards = []
        
        # Track BCF model availability for each decision point
        self.bcf_available = []
        
        # Store action estimates for each model to avoid redundant calculations
        self.bcf_arm_estimates = []
        self.linear_ts_arm_estimates = []
        
        # MSE scores for each agent
        self.bcf_mse = float('inf')
        self.linear_ts_mse = float('inf')
        
        # Thompson Sampling parameters for meta-agent
        self.bcf_alpha = 1.0
        self.bcf_beta = 1.0
        self.linear_ts_alpha = 1.0
        self.linear_ts_beta = 1.0
        
        # Track which agent was last used
        self.last_used_agent = None
        self.bcf_warm_up_iterations = 0
    
    def choose_arm(self, x: Union[np.ndarray, List[float]], **kwargs: Dict[str, Any]) -> int:
        """
        Choose an arm using Thompson Sampling to select between agents.
        If BCF model isn't fitted yet, defaults to LinearTS.
        
        Parameters:
            x (array-like): Feature vector.
            **kwargs: Additional parameters passed to the underlying agents.
            
        Returns:
            int: The selected arm index.
        """
        # Store features for tracking
        x_array = np.array(x).reshape(1, -1) if not isinstance(x, np.ndarray) else x
        self.features.append(x_array)
        
        # Check if BCF model is fitted
        bcf_is_fitted = self.bcf_agent.is_model_fitted
        self.bcf_available.append(bcf_is_fitted)
        
        # Get action estimates and recommended arms from both agents
        if bcf_is_fitted:
            bcf_estimates = self.bcf_agent._get_action_estimates(x, thompson_sampling=True)
            bcf_arm = int(np.argmax(bcf_estimates))
            self.bcf_predicted_rewards.append(bcf_estimates[bcf_arm])
            # Store full arm estimates for later use
            self.bcf_arm_estimates.append(bcf_estimates)
        else:
            # If BCF model isn't fitted, store placeholder values
            bcf_arm = -1  # Invalid arm to ensure it's not used
            self.bcf_predicted_rewards.append(float('nan'))  # NaN to indicate no prediction
            self.bcf_arm_estimates.append(None)
        
        # Always get LinearTS estimates
        linear_ts_estimates = self.linear_ts_agent._get_action_estimates(x)
        linear_ts_arm = int(np.argmax(linear_ts_estimates))
        self.linear_ts_predicted_rewards.append(linear_ts_estimates[linear_ts_arm])
        # Store full arm estimates for later use
        self.linear_ts_arm_estimates.append(linear_ts_estimates)
        
        # If BCF model isn't fitted or in initial phase, use LinearTS
        if not bcf_is_fitted: # or len(self.observed_rewards) < self.initial_phase_size:
            use_bcf = False
        elif self.bcf_warm_up_iterations < self.initial_phase_size:
            use_bcf = True
            self.bcf_warm_up_iterations += 1
        else: #if self.bcf_mse:
            # Normal distribution approach
            # Sample from normal distributions centered at 0
            # Lower MSE = tighter distribution = more consistent sampling
            bcf_std = np.sqrt(max(self.bcf_mse, 1e-10))
            linear_ts_std = np.sqrt(max(self.linear_ts_mse, 1e-10))

            use_bcf = bcf_std < linear_ts_std

            if np.random.rand() < 0.05:
                use_bcf = ~use_bcf # Randomly switch to the other agent 5% of the time

            # # We want lower variance to be better
            # bcf_sample = np.abs(np.random.normal(0, bcf_std))
            # linear_ts_sample = np.abs(np.random.normal(0, linear_ts_std))

            # use_bcf = bcf_sample < linear_ts_sample
        
        # Choose arm based on selected agent
        chosen_arm = bcf_arm if use_bcf else linear_ts_arm
        self.chosen_arms.append(chosen_arm)
        self.last_used_agent = 'BCF' if use_bcf else 'LinearTS'
        
        return chosen_arm
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], y: float) -> "EnsembleAgent":
        """
        Update the agent's state with new observation data.
        
        Parameters:
            arm (int): The index of the arm chosen.
            x (array-like): Feature vector.
            y (float): Observed reward.
            
        Returns:
            self: Updated instance.
        """
        # Update both underlying agents
        self.bcf_agent.update_state(arm, x, y)
        self.linear_ts_agent.update_state(arm, x, y)
        
        # Store observed reward
        self.observed_rewards.append(y)
        
        # Update MSE and Thompson Sampling parameters
        self._update_performance_metrics()
        
        return self
    
    def _update_performance_metrics(self) -> None:
        """
        Update performance metrics using direct MSE calculation between
        predicted and actual rewards for both agents.
        """
        # Need some history to calculate metrics
        if len(self.observed_rewards) < self.initial_phase_size:
            return
        
        # Get the recent window of data
        window_start = max(0, len(self.observed_rewards) - self.window_size)
        recent_features = self.features[window_start:]
        recent_bcf_preds = self.bcf_predicted_rewards[window_start:]
        recent_linear_ts_preds = self.linear_ts_predicted_rewards[window_start:]
        recent_chosen_arms = self.chosen_arms[window_start:]
        recent_rewards = self.observed_rewards[window_start:]
        recent_bcf_available = self.bcf_available[window_start:]
        
        # Use stored arm estimates to avoid redundant calculations
        recent_bcf_arm_estimates = self.bcf_arm_estimates[window_start:]
        recent_linear_ts_arm_estimates = self.linear_ts_arm_estimates[window_start:]
        
        # Calculate squared error for BCF when its recommendation was followed and model was fitted
        bcf_squared_errors = []
        for i in range(len(recent_rewards)):
            # Skip if BCF model wasn't available for this decision
            if not recent_bcf_available[i] or recent_bcf_arm_estimates[i] is None:
                continue
                
            # Get the arm that BCF would have chosen using stored estimates
            bcf_arm = np.argmax(recent_bcf_arm_estimates[i])
            
            # If this was the arm chosen, include in BCF's performance evaluation
            if bcf_arm == recent_chosen_arms[i]:
                # Calculate squared error between BCF's predicted reward and actual reward
                # Skip NaN predictions (placeholders from when BCF wasn't available)
                if not np.isnan(recent_bcf_preds[i]):
                    bcf_squared_errors.append((recent_bcf_preds[i] - recent_rewards[i])**2)
                
        # Calculate squared error for LinearTS when its recommendation was followed
        linear_ts_squared_errors = []
        for i in range(len(recent_rewards)):
            # Get the arm that LinearTS would have chosen using stored estimates
            linear_ts_arm = np.argmax(recent_linear_ts_arm_estimates[i])
            
            # If this was the arm chosen, include in LinearTS's performance evaluation
            if linear_ts_arm == recent_chosen_arms[i]:
                # Calculate squared error between LinearTS's predicted reward and actual reward
                linear_ts_squared_errors.append((recent_linear_ts_preds[i] - recent_rewards[i])**2)
        
        # Update MSE values
        if bcf_squared_errors:
            self.bcf_mse = np.mean(bcf_squared_errors)
        else:
            self.bcf_mse = float(1e-10) # Choose bcf if no data available for bcf
        if linear_ts_squared_errors:
            self.linear_ts_mse = np.mean(linear_ts_squared_errors)
        else:
            self.linear_ts_mse = float(1e-10)
            
