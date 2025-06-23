import numpy as np
from typing import Callable, List, Optional, Union, Dict, Any
import math
from ..bart import DefaultBART, LogisticBART
from ..mcbart import MultiChainBART
from .agent import BanditAgent

class BanditEncoder:
    """
    A utility class for encoding features for multi-armed bandit problems.
    This class handles different encoding strategies for the arms and features.
    """
    def __init__(self, n_arms: int, n_features: int, encoding: str) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.encoding = encoding
        
        if encoding == 'one-hot':
            self.combined_dim = n_features + n_arms
        elif encoding == 'multi':
            self.combined_dim = n_features * n_arms
        elif encoding == 'forest':
            raise NotImplementedError("Forest encoding is not yet implemented.")
        elif encoding == 'native':
            self.combined_dim = n_features
            # Native encoding is just the feature vector itself
            # This is useful for models that can handle categorical features directly
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
    def encode(self, x: Union[np.ndarray, List[float]], arm: int) -> np.ndarray:
        """
        Encode the feature vector x for a specific arm using the specified encoding strategy.
        
        Parameters:
            x (array-like): Feature vector
            arm (int): Index of the arm to encode for; if arm == -1, then encode all arms
            
        Returns:
            np.ndarray: Encoded feature vector
        """
        x = np.array(x).reshape(1, -1)

        if arm == -1:
            # Encode all arms
            range_arms = range(self.n_arms)
        else:
            range_arms = [arm]
            
        total_arms = len(range_arms)
        x_combined = np.zeros((total_arms, self.combined_dim))

        if self.encoding == 'one-hot':
            # One-hot encoded treatment options
            for row_idx, arm in enumerate(range_arms):
                x_combined[row_idx, :self.n_features] = x
                x_combined[row_idx, self.n_features + arm] = 1
        elif self.encoding == 'multi':
            # Block structure approach (data_multi style)
            for row_idx, arm in enumerate(range_arms):
                start_idx = arm * self.n_features
                end_idx = start_idx + self.n_features
                x_combined[row_idx, start_idx:end_idx] = x
        elif self.encoding == 'forest':
            raise NotImplementedError("Forest encoding is not yet implemented.")
        elif self.encoding == 'native':
            # Native encoding just returns the feature vector as is
            x_combined = x
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        return x_combined

class BARTAgent(BanditAgent):
    """
    A bandit agent that uses a Bayesian Additive Regression Trees (BART) model to choose an arm.
    BART models provide flexible nonparametric modeling of the reward function.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 ndpost: int = 1000, nskip: int = 100, nadd: int = 3,
                 n_trees: int = 200, 
                 random_state: int = 42,
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
        """
        super().__init__(n_arms, n_features)
        self.n_features = n_features
        
        # BART model parameters
        self.ndpost = ndpost
        self.nskip = nskip
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        self.encoding = encoding
        self.encoder = BanditEncoder(self.n_arms, self.n_features, self.encoding)
        self.combined_dim = self.encoder.combined_dim
        
        # Initialize storage for all data
        self.features = np.empty((0, n_features))  # X features
        self.outcomes = np.empty((0, 1))           # y outcomes
        # self.actions = np.empty((0, self.n_arms)) # only for forest encoding
        self.encoded_features = np.empty((0, self.combined_dim))  # Encoded features
        
        # Track if model is fitted
        self.is_model_fitted = False
        # The number of additional posterior iterations to add when updating the model
        self.nadd = nadd
        
        self.cnt = 0  # Counter for number of data
        
    @property
    def n_onehot_arms(self) -> int:
        """
        Number of one-hot encoded arms.
        """
        return self.n_arms - 1
    
    def _data_cnt_for_it(self, iteration):
        if iteration <= self.ndpost:
            return 0
        return (iteration - self.ndpost - 1) // self.nadd + 1
    
    def _mixing_bonus(self, iteration):
       return 1.0 / (1.0 + np.exp(iteration))
    
    def _default_schedule(self) -> Callable[[int], float]:
        total_k = self.model._trace_length
        raw_prob = self._mixing_bonus(np.arange(total_k))
        raw_prob = raw_prob / np.sum(raw_prob)
        return lambda k: raw_prob[k]
    
    def _get_action_estimates(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Get action estimates for all arms based on input features x.
        """
        # Ensure x is a 2D array with one row
        x = np.array(x).reshape(1, -1)
        
        x_combined = self.encoder.encode(x, arm=-1)  # Encode for all arms
        
        # Thompson Sampling:
        # Get posterior sample from BART model
        prob_schedule = self._default_schedule()
        post_mean_samples = self.model.posterior_sample(x_combined, schedule=prob_schedule)
        action_estimates = post_mean_samples

        return action_estimates

    def choose_arm(self, x: Union[np.ndarray, List[float]], **kwargs) -> int:
        """
        Choose an arm based on input features x using the BART model.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm
            
        Returns:
            int: The index of the selected arm
        """
        # If the model is not fitted yet, choose a random arm
        if not self.is_model_fitted:
            return self.rng.integers(0, self.n_arms)
        
        action_estimates = self._get_action_estimates(x)
        if action_estimates.ndim > 1 and action_estimates.shape[0] > 1:
            # For LogisticBART 'multi', the shape is (n_arms, 2), we only need the second column (the probability of success)
            action_estimates = action_estimates[:, 1] 
        # Note that for LogisticBART 'native', the shape is (1, n_arms) with n_arms=2, and we can use it directly
        
        # Choose the arm with the highest predicted outcome
        return int(np.argmax(action_estimates))
    
    def _clear_internal_data(self) -> None:
        """
        Clear internal data arrays after initial model fit.
        """
        self.features = np.empty((0, self.n_features))
        self.outcomes = np.empty((0, 1))    
        self.encoded_features = np.empty((0, self.combined_dim))
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], 
                     y: Union[float, np.ndarray]) -> "BARTAgent":
        """
        Update the agent's state with new observation data.
        
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
        
        # Common encoding logic for both phases
        new_features = self.encoder.encode(x, arm=arm)
        
        if self.encoding == 'native':
            # Native encoding just uses the feature vector as is
            # However, we need to encode y (standing for chosing the correct arm)
            # this given y stands for "real y == arm"
            y = np.logical_not(np.logical_xor(y, arm)).astype(float)
        
        if not self.is_model_fitted:
            # Accumulation phase: collect data for initial fit
            self.features = np.vstack([self.features, x])
            self.encoded_features = np.vstack([self.encoded_features, new_features])
            self.outcomes = np.vstack([self.outcomes, y.reshape(-1, 1)])
            
            self.cnt += 1
            
            # Check if we have enough data for initial fit
            # For LogisticBART, we need less observations and more than one unique outcome
            # For DefaultBART, we need more observations
            is_logistic = (isinstance(self.model, LogisticBART)) or (isinstance(self.model, MultiChainBART) and self.model.bart_class == LogisticBART)
            criteria = (self.cnt >= 10 and np.unique(self.outcomes).size > 1) if is_logistic else (self.cnt >= 20)
            if criteria:
                # Initial fit using all collected data
                print(f"Fitting initial BART model with first {self.cnt} observations...", end="")
                self.model.fit(
                    X=self.encoded_features, 
                    y=self.outcomes.flatten(),
                    quietly=True
                )
                print(" Done.")
                self.is_model_fitted = True
                
                # Clear all data after initial fit since we don't need it anymore
                self._clear_internal_data()
        else:
            # Online update phase: process single observation without accumulation
            # Update the model with the new data point only
            self.model.update_fit(
                X=new_features, 
                y=y.flatten(),
                add_ndpost=self.nadd,
                quietly=True
            )
        
        return self

class DefaultBARTAgent(BARTAgent):
    """
    Default BART agent with standard parameters.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 ndpost: int = 1000, nskip: int = 100, nadd: int = 3,
                 n_trees: int = 200, 
                 random_state: int = 42,
                 encoding: str = 'multi') -> None:
        super().__init__(n_arms, n_features, ndpost, nskip, nadd, n_trees, random_state, encoding)
        self.model = DefaultBART(
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        
class LogisticBARTAgent(BARTAgent):
    """
    BART agent for binary outcomes using logistic regression.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 ndpost: int = 1000, nskip: int = 100, nadd: int = 3,
                 n_trees: int = 200, 
                 random_state: int = 42,
                 encoding: str = 'native') -> None:
        if encoding != 'native' and encoding != 'multi':
            raise NotImplementedError("LogisticBARTAgent currently only supports 'native' encoding.")
        if n_arms > 2 and encoding == 'native':
            raise NotImplementedError("LogisticBARTAgent: native encoding currently only supports n_arms = 2.")
        super().__init__(n_arms, n_features, ndpost, nskip, nadd, n_trees, random_state, encoding)
        self.model = LogisticBART(
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        
class MultiChainBARTAgent(BARTAgent):
    """
    This agent can handle multiple chains of BART ensembles.
    """
    def __init__(self, n_arms: int, n_features: int, n_ensembles: int = 4,
                 bart_class: Callable = DefaultBART, 
                 ndpost: int = 1000, nskip: int = 100, nadd: int = 3,
                 n_trees: int = 200, 
                 random_state: int = 42,
                 encoding: str = '') -> None:
        if encoding == '':
            encoding = 'multi' # Should be very carefull if you want to use 'native' encoding here
        super().__init__(n_arms, n_features, ndpost, nskip, nadd, n_trees, random_state, encoding)
        self.model = MultiChainBART(
            n_ensembles=n_ensembles,  # Number of BART ensembles
            bart_class=bart_class,
            random_state=42,
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
    
    def clean_up(self) -> None:
        """
        Release resources held by the MultiChainBART agent.
        """
        self.model.clean_up()
        