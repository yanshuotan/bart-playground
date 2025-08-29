import numpy as np
from typing import Callable, List, Union
import logging
from bart_playground.bart import DefaultBART, LogisticBART
from bart_playground.bandit.agents.agent import BanditAgent
from bart_playground.bandit.experiment_utils.encoder import BanditEncoder

logger = logging.getLogger(__name__)

class BARTTSAgent(BanditAgent):
    """
    A BART agent that periodically re-fits the entire model from scratch,
    similar to the refresh strategy used in XGBoostTS and RandomForestTS agents.
    
    The model is re-trained when ceil(8*log(t)) > ceil(8*log(t-1)), where t is the time step.
    This ensures the refresh frequency increases logarithmically with time.
    """
    
    def __init__(self, n_arms: int, n_features: int, model_factory: Callable,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'multi') -> None:
        """
        Initialize the RefreshBART agent.
        
        Parameters:
            n_arms (int): Number of arms
            n_features (int): Number of features
            model_factory (Callable): Factory function to create BART models
            initial_random_selections (int): Number of initial random selections before using model
            random_state (int): Random seed
            encoding (str): Encoding strategy ('multi', 'one-hot', 'separate', 'native')
        """
        super().__init__(n_arms, n_features)
        
        self.t = 1  # Time step counter
        self.initial_random_selections = initial_random_selections
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        
        # Encoding setup
        self.encoding = encoding
        self.encoder = BanditEncoder(self.n_arms, self.n_features, self.encoding)
        self.combined_dim = self.encoder.combined_dim
        
        # Model setup
        self.model_factory = model_factory
        if self.encoding == 'separate':
            self.models = [model_factory() for _ in range(n_arms)]
        else:
            self.models = [model_factory()]
        
        # Check if model is logistic
        self.is_logistic = isinstance(self.model, LogisticBART)
        
        # Storage for all training data
        self.all_arms = []
        self.all_features = []
        self.all_rewards = []
        self.all_encoded_features = []
        
        # Model state
        self.is_model_fitted = False
        
    @property
    def model(self):
        return self.models[0]
    @model.setter
    def model(self, value):
        self.models[0] = value
    
    @property
    def separate_models(self) -> bool:
        return self.encoding == 'separate'
    
    def _should_refresh(self) -> bool:
        """Check if model should be refreshed based on time step."""
        if self.t < self.initial_random_selections:
            return False
        return np.ceil(8 * np.log(self.t)) > np.ceil(8 * np.log(self.t - 1))
    
    @staticmethod
    def _enough_data(outcomes, min_obs=4):
        """Check if we have enough data for the initial fit."""
        return outcomes.size >= min_obs # and np.unique(outcomes).size > 1
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data to fit the model."""
        if len(self.all_rewards) == 0:
            return False
            
        rewards = np.array(self.all_rewards)
        
        if self.separate_models:
            # Check that each arm has sufficient data
            arms = np.array(self.all_arms)
            return all(
                self._enough_data(rewards[arms == arm])
                for arm in range(self.n_arms)
            )
        else:
            # Check overall data sufficiency
            return self._enough_data(rewards)
    
    def _refresh_model(self) -> None:
        """Re-fit the model from scratch using all historical data."""
        if not self._has_sufficient_data():
            return

        logger.info(f't = {self.t} - re-training BART model from scratch')
        # sim_logger.info(f'Current model: {self.model.n_trees} trees, {self.model.ndpost} posterior samples, {self.model.nskip} burn-in samples, encoding = {self.encoding}')

        X = np.array(self.all_encoded_features)
        y = np.array(self.all_rewards)

        try:
            if self.separate_models:
                # Train separate models for each arm
                new_models = {}
                for arm in range(self.n_arms):
                    arm_mask = np.array(self.all_arms) == arm
                    X_arm = X[arm_mask]
                    y_arm = y[arm_mask]
                    model = self.model_factory()
                    model.fit(X_arm, y_arm, quietly=True)
                    new_models[arm] = model

                # Only replace if all fits succeeded
                self.models = new_models

            else:
                new_model = self.model_factory()
                new_model.fit(X, y, quietly=True)

            # Only replace on success
                self.model = new_model

            self.is_model_fitted = True

        except Exception:
            logger.exception('Failed to refresh model; keeping previous model(s) in place')
            # self.models or self.model and is_model_fitted remain as they were

    @staticmethod
    def _default_schedule(total_k) -> Callable[[int], float]:
    # weight 0 for the very first draw (burn-in),
    # then uniform 1/(total_k-1) for all subsequent samples
        return lambda k: 0.0 if k == 0 else 1.0 / (total_k - 1)

    def _get_action_estimates(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Get action estimates for all arms based on input features x."""
        x = np.array(x).reshape(1, -1)
        
        def _reformat(post_sample):
            if self.is_logistic:
                if self.encoding in ['multi', 'one-hot']:
                    post_sample = post_sample[:, 1] 
                elif self.encoding == 'native':
                    post_sample = post_sample.flatten()
                elif self.encoding == 'separate':
                    post_sample = post_sample[:, 1]
            return post_sample
        
        prob_schedule = self._default_schedule(self.model._trace_length)
        if not self.separate_models:
            x_combined = self.encoder.encode(x, arm=-1)  # Encode for all arms
            post_mean_samples = self.model.posterior_sample(x_combined, prob_schedule)
            action_estimates = _reformat(post_mean_samples)
        else:
            action_estimates = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                if hasattr(self.models[arm], 'posterior_sample'):  # Check if model is fitted
                    post_mean_samples = self.models[arm].posterior_sample(x, prob_schedule)
                    action_estimates[arm] = _reformat(post_mean_samples)
                else:
                    action_estimates[arm] = 0.0  # Default value for unfitted models
        
        return action_estimates
    
    def choose_arm(self, x: Union[np.ndarray, List[float]], **kwargs) -> int:
        """
        Choose an arm based on input features x.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm
            
        Returns:
            int: The index of the selected arm
        """
        # Random selection during initial phase
        if self.t < self.initial_random_selections or not self.is_model_fitted:
            return self.rng.integers(0, self.n_arms)
        
        # Use BART model for arm selection
        action_estimates = self._get_action_estimates(x)
        return int(np.argmax(action_estimates))
    
    def update_state(self, arm: int, x: Union[np.ndarray, List[float]], 
                     y: Union[float, np.ndarray]) -> "BARTTSAgent":
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
        x = np.array(x).reshape(-1)
        y = float(y)
        
        # Encode features for the chosen arm
        encoded_x = self.encoder.encode(x, arm=arm)
        if encoded_x.ndim > 1:
            encoded_x = encoded_x.flatten()
            
        # Handle native encoding for logistic models
        if self.encoding == 'native' and self.is_logistic:
            y = float(np.logical_not(np.logical_xor(y, arm)))
        
        # Store all data for potential refresh
        self.all_arms.append(arm)
        self.all_features.append(x.copy())
        self.all_rewards.append(y)
        self.all_encoded_features.append(encoded_x.copy())
        
        # Check if we should refresh the model
        if self._should_refresh():
            self._refresh_model()
        
        # Increment time counter
        self.t += 1
        
        return self


class DefaultBARTTSAgent(BARTTSAgent):
    """
    Refresh BART agent with default BART parameters.
    """
    def __init__(self, n_arms: int, n_features: int,
                 ndpost: int = 1000, nskip: int = 100,
                 n_trees: int = 200,
                 dirichlet_prior: bool = False,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'multi') -> None:
        model_factory = lambda: DefaultBART(
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1},
            dirichlet_prior=dirichlet_prior
        )
        super().__init__(n_arms, n_features, model_factory, 
                         initial_random_selections, random_state, encoding)


class LogisticBARTTSAgent(BARTTSAgent):
    """
    Refresh BART agent for binary outcomes using logistic regression.
    """
    def __init__(self, n_arms: int, n_features: int,
                 ndpost: int = 1000, nskip: int = 100,
                 n_trees: int = 200,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'native') -> None:
        if n_arms > 2 and encoding == 'native':
            raise NotImplementedError("RefreshLogisticBARTAgent: native encoding currently only supports n_arms = 2.")
        
        model_factory = lambda: LogisticBART(
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding)
