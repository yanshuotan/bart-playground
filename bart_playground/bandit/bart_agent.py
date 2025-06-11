import numpy as np
from typing import List, Optional, Union, Dict, Any

from ..bart import DefaultBART
from .agent import BanditAgent

class BARTAgent(BanditAgent):
    """
    A bandit agent that uses a Bayesian Additive Regression Trees (BART) model to choose an arm.
    BART models provide flexible nonparametric modeling of the reward function.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 ndpost: int = 1000, nskip: int = 100, nadd: int = 3,
                 n_trees: int = 200, 
                 random_state: int = 42, nbatch: int = 1,
                 encoding: str = 'multi') -> None:
        """
        Initialize the BART-based bandit agent.
        
        Parameters:
            n_arms (int): Number of arms
            n_features (int): Number of features
            ndpost (int): Number of posterior samples
            nskip (int): Number of burn-in iterations
            nadd (int): Number of additional posterior samples per update
            n_trees (int): Number of trees
            random_state (int): Random seed
            nbatch (int): Number of observations to collect before updating the model
        """
        super().__init__(n_arms, n_features)
        self.n_features = n_features
        
        # BART model parameters
        self.ndpost = ndpost
        self.nskip = nskip
        self.random_state = random_state
        
        self.encoding = encoding
        if encoding == 'one-hot':
            self.combined_dim = n_features + n_arms - 1
        elif encoding == 'multi':
            self.combined_dim = n_features * n_arms
        elif encoding == 'forest':
            raise NotImplementedError("Forest encoding is not yet implemented.")
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        # Initialize storage for all data
        self.features = np.empty((0, n_features))  # X features
        self.outcomes = np.empty((0, 1))           # y outcomes
        # self.actions = np.empty((0, self.n_arms)) # only for forest encoding
        self.encoded_features = np.empty((0, self.combined_dim))  # Encoded features
            
        # Initialize the BART model
        self.model = DefaultBART(
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        
        # Track if model is fitted
        self.is_model_fitted = False
        # The number of additional posterior iterations to add when updating the model
        self.nadd = nadd
        
        # Batch processing parameters
        self.nbatch = max(1, nbatch)  # Ensure at least batch size 1
        self.batch_start_idx = 0  # Track the starting index of current batch
        
        self.cnt = 0  # Counter for number of data
        
    @property
    def n_onehot_arms(self) -> int:
        """
        Number of one-hot encoded arms.
        """
        return self.n_arms - 1
    
    def _get_action_estimates(self, x: Union[np.ndarray, List[float]],
                              thompson_sampling: bool = True) -> np.ndarray:
        """
        Get action estimates for all arms based on input features x.
        """
        # Ensure x is a 2D array with one row
        x = np.array(x).reshape(1, -1)
        
        if self.encoding == 'one-hot':
            # Original one-hot encoding approach
            # Create treatment scenarios - one-hot encoded treatment options
            raise NotImplementedError("One-hot encoding is not yet implemented.")
            action_options = np.zeros((self.n_arms, self.n_onehot_arms))
            for i in range(self.n_arms):
                if i > 0:  # For treatment arms 
                    action_options[i, i-1] = 1
            
            # Replicate x for each treatment scenario
            x_repeated = np.tile(x, (self.n_arms, 1))
            x_combined = np.hstack([x_repeated, action_options])
            
        elif self.encoding == 'multi':
            # Block structure approach (data_multi style)
            x_combined = np.zeros((self.n_arms, self.combined_dim))
            
            for arm in range(self.n_arms):
                # Place features at arm-specific positions
                start_idx = arm * self.n_features
                end_idx = start_idx + self.n_features
                x_combined[arm, start_idx:end_idx] = x
        
        if thompson_sampling:
            # Get posterior sample from BART model
            total_k = len(self.model.trace)
            raw_prob = np.zeros(total_k)
            for i in range(total_k):
                if i + self.nadd < total_k:
                    raw_prob[i] = np.exp(i - (total_k - self.nadd))
                else:
                    raw_prob[i] = 1
            prob = raw_prob / np.sum(raw_prob)
            prob_schedule = lambda k: prob[k]
            post_y = self.model.posterior_sample(x_combined, schedule=prob_schedule)
            action_estimates = post_y
        else:
            # Use expected value (posterior mean)
            action_estimates = self.model.predict(x_combined)

        return action_estimates

    def choose_arm(self, x: Union[np.ndarray, List[float]], 
                   thompson_sampling: bool = True) -> int:
        """
        Choose an arm based on input features x using the BART model.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm
            thompson_sampling (bool): Whether to use Thompson sampling
                                     (if False, uses expected value)
            
        Returns:
            int: The index of the selected arm
        """
        # If the model is not fitted yet, choose a random arm
        if not self.is_model_fitted:
            return np.random.randint(self.n_arms)
        
        action_estimates = self._get_action_estimates(x, thompson_sampling)
        
        # Choose the arm with the highest predicted outcome
        return int(np.argmax(action_estimates))
    
    def _clear_internal_data(self) -> None:
        """
        Clear internal data arrays after model update.
        """
        self.features = np.empty((0, self.n_features))
        self.outcomes = np.empty((0, 1))    
        self.encoded_features = np.empty((0, self.combined_dim))
        self.batch_start_idx = 0  # Reset batch index since we cleared arrays
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], 
                     y: Union[float, np.ndarray]) -> "BARTAgent":
        """
        Update the agent's state with new observation data, optionally batching updates.
        
        Parameters:
            arm (int): The index of the arm chosen
            x (array-like): Feature vector
            y (float): Observed reward
            
        Returns:
            self: Updated instance
        """
        # Convert inputs to the right shapes
        x = np.array(x).reshape(1, -1)
        y = np.array(y).reshape(1)
        
        self.features = np.vstack([self.features, x])

        if self.encoding == 'one-hot':
            raise NotImplementedError("One-hot encoding is not yet implemented.")
            # Create treatment indicator (one-hot encoded)
            z = np.zeros((1, self.n_treat_arms))
            if arm > 0:
                z[0, arm - 1] = 1
            
            # Append to our overall dataset
            self.features = np.vstack([self.features, x])
            # self.treatments = np.vstack([self.treatments, z])
        
        elif self.encoding == 'multi':
            # For block encoding, we create a full context vector and append it directly
            block_x = np.zeros((1, self.combined_dim))
            start_idx = arm * self.n_features
            end_idx = start_idx + self.n_features
            block_x[0, start_idx:end_idx] = x

            self.encoded_features = np.vstack([self.encoded_features, block_x])

        self.outcomes = np.vstack([self.outcomes, y.reshape(-1, 1)])
        
        self.cnt += 1
        
        # Calculate current batch size
        current_batch_size = self.features.shape[0] - self.batch_start_idx
        
        # Check if we have at least one sample for each treatment arm and one control sample
        # has_all_treatments = not np.any(np.sum(self.treatments, axis=0) == 0)  # All arms have samples
        # has_control = np.any(np.sum(self.treatments, axis=1) == 0)  # At least one control sample
        
        # We need enough samples overall and a complete batch
        should_update = False
        if self.cnt >= 20: # has_all_treatments and has_control and self.cnt >= 20:
            if current_batch_size >= self.nbatch:
                should_update = True
                
        if not should_update:
            return self
        
        # Update the model
        if not self.is_model_fitted:
            # Initial fit using all collected data
            self.model.fit(
                X=self.encoded_features, 
                y=self.outcomes.flatten(),
                quietly=True
            )
            self.is_model_fitted = True
            
            # Clear all data after initial fit since we don't need it anymore
            self._clear_internal_data()
        else:
            # For updates, all current data is batch data since we reset after each update
            # Note that these data are to be added to the existing model
            self.model.update_fit(
                X=self.encoded_features, 
                y=self.outcomes.flatten(),
                add_ndpost=self.nadd,
                quietly=True
            )
            self._clear_internal_data()
        
        return self
