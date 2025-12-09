import numpy as np
from typing import Callable, List, Union, Optional, Dict, Any

from bart_playground.bandit.experiment_utils.encoder import BanditEncoder
from bart_playground.bart import DefaultBART, LogisticBART
from bart_playground.mcbart import MultiChainBART
from bart_playground.bandit.agents.agent import BanditAgent

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


class BARTAgent(BanditAgent):
    """
    A bandit agent that uses a Bayesian Additive Regression Trees (BART) model to choose an arm.
    BART models provide flexible nonparametric modeling of the reward function.
    """
    def __init__(self, n_arms: int, n_features: int, model_factory: Callable,
                 nadd: int = 3,
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
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        self.encoding = encoding
        self.encoder = BanditEncoder(self.n_arms, self.n_features, self.encoding)
        self.combined_dim = self.encoder.combined_dim
        
        # Initialize storage for all data
        self._clear_internal_data()
        
        self.model_factory = model_factory
        if self.encoding == 'separate':
            self.models = [model_factory() for _ in range(n_arms)]
        else:
            # Create a single BART model for all arms
            self.models = [model_factory()]
        self.is_logistic = isinstance(self.model, LogisticBART) or \
                        (isinstance(self.model, MultiChainBART) and self.model.bart_class == LogisticBART)
        # Track if model is fitted
        self.is_model_fitted = False
        # The number of additional posterior iterations to add when updating the model
        self.nadd = nadd
        
        self.cnt = 0  # Counter for number of data
        
    @property
    def model(self):
        return self.models[0]
    
    @property
    def separate_models(self) -> bool:
        return self.encoding == 'separate'

    def _mixing_bonus(self, iteration):
       return 1.0 / (1.0 + np.exp(-iteration))
    
    def total_iter(self, model_idx=0) -> int:
        return self.models[model_idx]._trace_length
    
    def _default_schedule(self, total_k) -> Callable[[int], float]:
        # raw_prob = self._mixing_bonus(np.arange(total_k))
        # raw_prob = raw_prob / np.sum(raw_prob)
        # return lambda k: raw_prob[k]
        return lambda k: 0.0 if k == 0 else 1.0 / (total_k - 1)
    
    def _get_action_estimates(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Get action estimates for all arms based on input features x.
        """
        # Ensure x is a 2D array with one row
        x = np.array(x).reshape(1, -1)
        
        # Thompson Sampling:
        # Get posterior sample from BART model
        
        def _reformat(post_sample):
            if self.is_logistic:
                # For LogisticBART 'multi' or 'one-hot', the shape is (n_arms, 2), we only need the second column (the probability of success)
                if self.encoding == 'multi' or self.encoding == 'one-hot':
                    post_sample = post_sample[:, 1] 
                # For LogisticBART 'native', the shape is (1, n_arms) with n_arms=2
                elif self.encoding == 'native':
                    post_sample = post_sample.flatten()
                # For LogisticBART 'separate', the shape is (1, 2) for each arm
                elif self.encoding == 'separate':
                    post_sample = post_sample[:, 1]
            return post_sample

        if not self.separate_models:
            prob_schedule = self._default_schedule(self.total_iter())
            x_combined = self.encoder.encode(x, arm=-1)  # Encode for all arms
            post_mean_samples = self.model.posterior_sample(x_combined, schedule=prob_schedule)
            action_estimates = _reformat(post_mean_samples)
        else:
            # For separate encoding, we need to get estimates for each arm separately
            action_estimates = np.zeros((self.n_arms))
            for arm in range(self.n_arms):
                prob_schedule = self._default_schedule(self.total_iter(model_idx=arm))
                post_mean_samples = self.models[arm].posterior_sample(x, schedule=prob_schedule)
                
                action_estimates[arm] = _reformat(post_mean_samples)

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
        
        # Choose the arm with the highest predicted outcome
        return int(np.argmax(action_estimates))
    
    def _clear_internal_data(self) -> None:
        """
        Clear internal data arrays after initial model fit.
        """
        self.ini_arms = np.empty((0, 1))
        self.ini_outcomes = np.empty((0, 1))    
        self.ini_enc_features = np.empty((0, self.combined_dim))
    
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
        new_enc_features = self.encoder.encode(x, arm=arm)
        
        if self.encoding == 'native':
            # Native encoding just uses the feature vector as is
            # However, we need to encode y (standing for chosing the correct arm)
            # this given y stands for "real y == arm"
            y = np.logical_not(np.logical_xor(y, arm)).astype(float)
        
        if not self.is_model_fitted:
            # Accumulation phase: collect data for initial fit
            self.ini_arms = np.vstack([self.ini_arms, arm])
            self.ini_enc_features = np.vstack([self.ini_enc_features, new_enc_features])
            self.ini_outcomes = np.vstack([self.ini_outcomes, y.reshape(-1, 1)])
            
            self.cnt += 1
            
            # Check if we have enough data for initial fit
            # we need more than one unique outcome
            # and at least some (say, 4) observations per arm (or overall if not separate models)
            def _enough_data(outcomes, min_obs=4):
                return outcomes.size >= min_obs # and np.unique(outcomes).size > 1

            if self.separate_models:
                criteria = all(
                    _enough_data(self.ini_outcomes[self.ini_arms.flatten() == arm])
                    for arm in range(self.n_arms)
                )
            else:
                criteria = _enough_data(self.ini_outcomes)
                
            if criteria:
                # Initial fit using all collected data
                print(f"Fitting initial BART model with first {self.cnt} observation(s)...", end="")
                # TODO
                if self.separate_models:
                    for arm in range(self.n_arms):
                        mask = (self.ini_arms.flatten() == arm)
                        self.models[arm].fit(
                            X=self.ini_enc_features[mask, :],
                            y=self.ini_outcomes[mask].flatten(),
                            quietly=True
                        )
                else: 
                    self.model.fit(
                        X=self.ini_enc_features,
                        y=self.ini_outcomes.flatten(),
                        quietly=True
                    )
                print(" Done.")
                self.is_model_fitted = True
                
                # Clear all data after initial fit since we don't need it anymore
                self._clear_internal_data()
        else:
            # Online update phase: process single observation without accumulation
            # Update the model with the new data point only
            if self.separate_models:
                # For separate encoding, we update the specific model for the chosen arm
                updatee = self.models[arm]
            else:
                updatee = self.model

            updatee.update_fit(
                X=new_enc_features, 
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
                 dirichlet_prior: bool = False,
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
        super().__init__(n_arms, n_features, model_factory, nadd, random_state, encoding)
        
class LogisticBARTAgent(BARTAgent):
    """
    BART agent for binary outcomes using logistic regression.
    """
    def __init__(self, n_arms: int, n_features: int, 
                 ndpost: int = 1000, nskip: int = 100, nadd: int = 3,
                 n_trees: int = 200, 
                 random_state: int = 42,
                 encoding: str = 'native') -> None:
        if n_arms > 2 and encoding == 'native':
            raise NotImplementedError("LogisticBARTAgent: native encoding currently only supports n_arms = 2.")
        model_factory = lambda: LogisticBART(
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        super().__init__(n_arms, n_features, model_factory, nadd, random_state, encoding)
        
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
        model_factory = lambda: MultiChainBART(
            n_ensembles=n_ensembles,  # Number of BART ensembles
            bart_class=bart_class,
            random_state=42,
            n_trees=n_trees,
            ndpost=ndpost,
            nskip=nskip,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        super().__init__(n_arms, n_features, model_factory, nadd, random_state, encoding)

    def clean_up(self) -> None:
        """
        Release resources held by the MultiChainBART agent.
        """
        for model in self.models:
            model.clean_up()
