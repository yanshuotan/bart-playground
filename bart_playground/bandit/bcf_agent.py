import numpy as np
from typing import List, Optional, Union, Dict, Any
from ..bcf.bcf import BCF
from .agent import BanditAgent

class BCFAgent(BanditAgent):
    """
    A bandit agent that uses a Bayesian Causal Forest (BCF) model to choose an arm.
    BCF models provide causal treatment effect estimates which are ideal for contextual bandits.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 ndpost: int = 1000, nskip: int = 100, nadd = 3,
                 n_mu_trees: int = 200, n_tau_trees: Optional[List[int]] = None,
                 random_state: int = 42, nbatch: int = 1) -> None:
        """
        Initialize the BCF-based bandit agent.
        
        Parameters:
            n_arms (int): Number of total arms (including control)
            n_features (int): Number of features
            ndpost (int): Number of posterior samples
            nskip (int): Number of burn-in iterations
            n_mu_trees (int): Number of prognostic trees
            n_tau_trees (list[int]): Number of trees per treatment effect model
            random_state (int): Random seed
            nbatch (int): Number of observations to collect before updating the model
        """
        super().__init__(n_arms, n_features)
        self.n_features = n_features
        
        # Initialize storage for all data
        self.features = np.empty((0, n_features))  # X features
        self.outcomes = np.empty((0, 1))           # y outcomes
        self.treatments = np.empty((0, self.n_treat_arms))    # internal treatment indicators (excluding control)
        
        # BCF model parameters
        self.ndpost = ndpost
        self.nskip = nskip
        self.random_state = random_state
        
        # Initialize the BCF model
        if n_tau_trees is None:
            n_tau_trees = [50] * self.n_treat_arms
            
        self.model = BCF(
            n_treat_arms=self.n_treat_arms,
            n_mu_trees=n_mu_trees,
            n_tau_trees=n_tau_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state
        )
        
        # Track if model is fitted
        self.is_model_fitted = False
        # The number of additional posterior iterations to add when updating the model
        self.nadd = nadd
        
        # Batch processing parameters
        self.nbatch = max(1, nbatch)  # Ensure at least batch size 1
        self.batch_start_idx = 0  # Track the starting index of current batch

    @property
    def n_treat_arms(self) -> int:
        '''
        Number of treatment arms (excluding control arm).
        '''
        return self.n_arms - 1

    def choose_arm(self, x: Union[np.ndarray, List[float]], 
                   thompson_sampling: bool = True) -> int:
        """
        Choose an arm based on input features x using the BCF model.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm
            binary (bool): Whether the outcome is binary (not used in this implementation)
            thompson_sampling (bool): Whether to use Thompson sampling
                                     (if False, uses expected value)
            
        Returns:
            int: The index of the selected arm
        """
        # Ensure x is a 2D array with one row
        x = np.array(x).reshape(1, -1)
        
        # If the model is not fitted yet, choose a random arm
        if not self.is_model_fitted:
            return np.random.randint(self.n_arms)

        # Create treatment scenarios - one-hot encoded treatment options
        #   with shape (n_arms, n_treat_arms) 
        treatment_options = np.zeros((self.n_arms, self.n_treat_arms))
        for i in range(self.n_arms):
            if i > 0:  # For treatment arms 
                treatment_options[i, i-1] = 1
        
        # Replicate x for each treatment scenario
        x_repeated = np.tile(x, (self.n_arms, 1))
        
        if thompson_sampling:
            # Get posterior samples
            _, _, post_y = self.model.predict_all(x_repeated, treatment_options)
            # TODO could be improved by avoiding redundant prediction calculations
            
            # Sample one random draw from the posterior for each arm
            sample_idx = np.random.randint(post_y.shape[1])
            utility_estimates = post_y[:, sample_idx]
        else:
            # Use expected value (posterior mean)
            utility_estimates = self.model.predict(x_repeated, treatment_options)
        
        # Choose the arm with the highest predicted outcome
        return int(np.argmax(utility_estimates))
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], 
                     y: Union[float, np.ndarray]) -> "BCFAgent":
        """
        Update the agent's state with new observation data, optionally batching updates.
        
        Parameters:
            arm (int): The index of the arm chosen, 0 for the control
            x (array-like): Feature vector
            y (float): Observed reward
            
        Returns:
            self: Updated instance
        """
        # Convert inputs to the right shapes
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1)
        
        # Create treatment indicator (one-hot encoded)
        z = np.zeros((1, self.n_treat_arms))
        if arm > 0:
            z[0, arm - 1] = 1
        
        # Append to our overall dataset
        self.features = np.vstack([self.features, x])
        self.outcomes = np.vstack([self.outcomes, y.reshape(-1, 1)])
        self.treatments = np.vstack([self.treatments, z])
        
        # Calculate current batch size
        current_batch_size = self.features.shape[0] - self.batch_start_idx
        
        # Check if we have enough data to update
        should_update = False
        
        # Check if we have at least one sample for each treatment arm and one control sample
        has_all_treatments = not np.any(np.sum(self.treatments, axis=0) == 0)  # All arms have samples
        has_control = np.any(np.sum(self.treatments, axis=1) == 0)  # At least one control sample
        
        # We need enough samples overall and a complete batch
        if has_all_treatments and has_control and self.features.shape[0] >= 20:
            if current_batch_size >= self.nbatch:
                should_update = True
        
        if not should_update:
            return self
        
        # Update the model
        if not self.is_model_fitted:
            # Initial fit using all collected data
            self.model.fit(
                X=self.features, 
                y=self.outcomes.flatten(), 
                Z=self.treatments,
                quietly=True
            )
            self.is_model_fitted = True
        else:
            # Get the current batch data using slicing
            batch_features = self.features[self.batch_start_idx:]
            batch_outcomes = self.outcomes[self.batch_start_idx:]
            batch_treatments = self.treatments[self.batch_start_idx:]
            
            # Efficiently update the model with new batch data
            self.model.update_fit(
                X=batch_features, 
                y=batch_outcomes.flatten(), 
                Z=batch_treatments,
                add_ndpost=self.nadd,
                add_nskip=0,
                quietly=True
            )
        
        # Update batch start index for next batch
        self.batch_start_idx = self.features.shape[0]
        
        return self
