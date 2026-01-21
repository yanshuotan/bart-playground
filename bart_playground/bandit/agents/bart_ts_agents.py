import numpy as np
import pandas as pd
from typing import Callable, List, Union, Tuple, cast, Optional, Dict, Any
import logging
from abc import abstractmethod
from bart_playground.bart import DefaultBART, LogisticBART, BART
from bart_playground.mcbart import MultiChainBART
from bart_playground.bandit.agents.agent import BanditAgent
from bart_playground.bandit.experiment_utils.encoder import BanditEncoder
from bart_playground.diagnostics import compute_diagnostics, MoveAcceptance
from bart_playground.bandit.agents.refresh_schedule import RefreshScheduleMixin
from bart_playground.bandit.agents.diagnostics_mixin import DiagnosticsMixin

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

def _prepare_bart_kwargs(default_kwargs: Dict[str, Any], 
                         user_kwargs: Optional[Dict[str, Any]]) -> Tuple[int, Optional[int], Dict[str, Any]]:
    """
    Helper to merge default and user kwargs.
    Returns: (base_ndpost, user_max_bins, merged_kwargs)
    """
    merged = dict(default_kwargs)
    if user_kwargs:
        merged.update(user_kwargs)
        
    # If quick_decay is True and tree_alpha is not specified, default tree_alpha to 0.45
    if merged.get("quick_decay", False) and "tree_alpha" not in merged:
        merged["tree_alpha"] = 0.45
        
    # Pop dynamic parameters to be explicitly passed to model_factory
    base_ndpost = int(merged.pop("ndpost", 1000))
    user_max_bins = merged.pop("max_bins", None)
    if user_max_bins is not None:
        user_max_bins = int(user_max_bins)
    merged.pop("random_state", None)
    
    return base_ndpost, user_max_bins, merged

class BARTTSAgent(BanditAgent, RefreshScheduleMixin, DiagnosticsMixin):
    """
    A BART agent that periodically re-fits the entire model from scratch,
    similar to the refresh strategy used in XGBoostTS and RandomForestTS agents.
    
    The model is re-trained when ceil(8*log(t)) > ceil(8*log(t-1)), where t is the time step.
    This ensures the refresh frequency increases logarithmically with time.
    """
    
    def __init__(self, n_arms: int, n_features: int, model_factory: Callable,
                 initial_random_selections: int = 5,
                 random_state: int = 42,
                 encoding: str = 'multi',
                 refresh_schedule: str = 'log',
                 feel_good_lambda: float = 0.0) -> None:
        """
        Initialize the RefreshBART agent.
        
        Parameters:
            n_arms (int): Number of arms
            n_features (int): Number of features
            model_factory (Callable): Factory function to create BART models
            initial_random_selections (int): Number of warm-start selections per arm before using model.
                Total warm-start steps = n_arms * initial_random_selections.
            random_state (int): Random seed
            encoding (str): Encoding strategy ('multi', 'one-hot', 'separate')
            refresh_schedule (str): Strategy to control model refresh frequency:
                - 'log': Standard (default). Refresh when ceil(8*log(t)) jumps. Aggressive early, scarce later.
                - 'sqrt': Balanced. Refresh when ceil(0.57*sqrt(t)) jumps. Smoother early phase.
                - 'hybrid': Sqrt -> Log. Reduced early overhead, standard late behavior.
                - 'rev_hybrid': Log -> Sqrt. Aggressive early learning, maintained alertness later.
            feel_good_lambda (float): Lambda for feel-good weights. If 0, caching is disabled.
        """
        super().__init__(n_arms, n_features)
        
        self.t = 1  # Time step counter
        self.refresh_schedule = refresh_schedule
        self.initial_random_selections = initial_random_selections
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

        # Build warm-start queue: initial_random_selections rounds of random permutations of all arms.
        self._warmstart_arms: List[int] = []
        for _ in range(self.initial_random_selections):
            self._warmstart_arms.extend(self.rng.permutation(self.n_arms).tolist())
        
        # Encoding setup
        self.encoding = encoding
        self.encoder = BanditEncoder(self.n_arms, self.n_features, self.encoding)
        self.combined_dim = self.encoder.combined_dim
        
        # Number of Thompson-sampling draws per decision step (can be overridden by callers)
        self.choices_per_iter: int = 1

        # Model setup
        self.model_factory = model_factory
        self.models: Union[list[BART], MultiChainBART] 
        if self.encoding == 'separate':
            first_model = model_factory()
            if isinstance(first_model, MultiChainBART) and getattr(first_model, "n_models", 1) == self.n_arms:
                self.models = first_model
            else:
                self.models = [first_model]
                for _ in range(1, n_arms):
                    self.models.append(model_factory())
        else:
            self.models = [model_factory()]
        
        # Record the maximum ndpost as the user-initialized value to cap future refreshes
        self.max_ndpost = int(getattr(self.model, 'ndpost', 0))

        # Check if model is logistic
        if isinstance(self.model, MultiChainBART):
            self.is_logistic = issubclass(self.model.bart_class, LogisticBART)
        else:
            self.is_logistic = isinstance(self.model, LogisticBART)
        
        # Storage for all training data
        self.all_arms = []
        self.all_features = []
        self.all_rewards = []
        self.all_encoded_features = []
        
        # Model state
        self.is_model_fitted = False
        
        # Posterior sample management
        self._post_indices = []
        self._post_idx_pos = 0
        
        # Flag for logging model params once
        self._has_logged_params = False
        
        # Fixed max_bins provided by caller (None -> use adaptive rule)
        self._fixed_max_bins_value: Optional[int] = None
        
        # Feel-good weights caching
        self.feel_good_lambda = float(feel_good_lambda)
        self._fg_S: Optional[np.ndarray] = None  # Cache for S(Theta), shape (n_post,)
        
    @property
    def model(self):
        return self.models[0]
    @model.setter
    def model(self, value):
        # Setter only called in non-separate branch where self.models is always a list; assertion always holds.
        assert isinstance(self.models, list), "Must be a list of BART models"
        self.models[0] = value
    
    @property
    def separate_models(self) -> bool:
        return self.encoding == 'separate'

    @staticmethod
    def _enough_data(outcomes, min_obs=5):
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
        
    def _ndpost_needed(self, max_needed: int) -> int:
        """Determine the number of posterior samples needed for the next refresh. At most self.max_ndpost. At least 100."""
        return int(max(100, min(self.max_ndpost, max_needed * (self.choices_per_iter+1))))
    
    def _model_factory_for_refresh(self) -> Callable:
        """Hook to determine the factory to use for THIS refresh."""
        return self.model_factory

    def _current_max_bins(self) -> int:
        """Return the max_bins value to use for the next refresh."""
        if self._fixed_max_bins_value is not None:
            return int(self._fixed_max_bins_value)
        bins = np.ceil(0.2 * max(self.t, 1)).astype(int)
        # soft cap at 2000 to avoid excessive quantile generation
        return int(min(2000, max(100, bins)))
    
    def _refresh_model(self) -> bool:
        """
        Re-fit the model from scratch using all historical data.
        
        Returns:
            bool: True if refresh succeeded, False if it failed
        """
        logger.info(f't = {self.t} - re-training BART model from scratch')

        X = np.array(self.all_encoded_features)
        y = np.array(self.all_rewards)
        current_max_bins = self._current_max_bins()
        if self._fixed_max_bins_value is None:
            logger.debug(f"Adaptive max_bins={current_max_bins} at t={self.t}")

        try:
            # Determine maximum number of actions until next refresh
            max_needed = self._steps_until_next_refresh(self.t)
            ndpost_need = self._ndpost_needed(max_needed)
            factory = self._model_factory_for_refresh()

            if self.separate_models:
                use_multichain = isinstance(self.models, MultiChainBART) and getattr(self.models, "n_models", 1) == self.n_arms
                if use_multichain:
                    self.models: MultiChainBART
                    self.models.set_ndpost(ndpost_need)
                    self.models.set_max_bins(current_max_bins)
                    for arm in range(self.n_arms):
                        arm_mask = np.array(self.all_arms) == arm
                        X_arm = X[arm_mask]
                        y_arm = y[arm_mask]
                        # Note: fit() trains all chains in parallel, but each actor only fits its active arm model
                        self.models[arm].fit(X_arm, y_arm, quietly=True)
                else:
                    # Train separate models for each arm
                    new_models: list[BART] = []
                    for arm in range(self.n_arms):
                        arm_mask = np.array(self.all_arms) == arm
                        X_arm = X[arm_mask]
                        y_arm = y[arm_mask]
                        model = factory(ndpost_need, max_bins=current_max_bins) 
                        model.fit(X_arm, y_arm, quietly=True)
                        new_models.append(model)

                    # Only replace if all fits succeeded
                    self.models = new_models

            else:
                new_model = factory(ndpost_need, max_bins=current_max_bins) 
                new_model.fit(X, y, quietly=True)
                # Only replace on success
                self.model = new_model

            self.is_model_fitted = True
            
            # Log model parameters on first successful fit
            if not self._has_logged_params:
                params = self.model.get_params()
                logger.info(f"First model refresh complete. Effective model parameters:")
                for k, v in params.items():
                    logger.info(f"  - {k}: {v}")
                self._has_logged_params = True

            # Reset sequential posterior index queue
            self._reset_post_queue()
            
            # Feel-good cache full recompute on posterior refresh
            if self.feel_good_lambda != 0.0:
                self._feel_good_full_recompute()
            
            # Memory tracking after refresh
            if PSUTIL_AVAILABLE and self.t >= 1000:
                try:
                    process = psutil.Process()
                    mem_main = process.memory_info().rss / (1024 ** 2)  # MB
                    
                    # Include memory from child processes (e.g., Ray actors)
                    # Handle race condition where child processes may exit during iteration
                    mem_children_bytes = 0
                    for child in process.children(recursive=True):
                        try:
                            mem_children_bytes += child.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                    mem_children = mem_children_bytes / (1024 ** 2)
                    mem_total = mem_main + mem_children
                    
                    if mem_children > 0:
                        logger.info(f"Memory: {mem_total:.1f} MB (main: {mem_main:.1f} MB, workers: {mem_children:.1f} MB)")
                    else:
                        logger.info(f"Memory: {mem_total:.1f} MB")
                except Exception:
                    # Don't let memory logging crash the agent
                    pass

            return True

        except Exception:
            logger.exception('Failed to refresh model; keeping previous model(s) in place')
            return False

    def _get_action_estimates(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Get action estimates for all arms based on input features x."""
        if not self.is_model_fitted:
            logger.error(f'Model is not fitted, returning 0.0')
            return np.zeros(self.n_arms, dtype=float)
            
        x = np.array(x).reshape(1, -1)
        
        def _reformat(post_sample):
            if self.is_logistic:
                post_sample = post_sample[:, 1]
            return post_sample
        
        # Select posterior draw index based on feel-good mode
        if self.feel_good_lambda != 0.0:
            k = self._sample_fg_post_index()
        else:
            # Non-feel-good: use non-repeating queue
            k = self._next_post_index()
        
        def _eval_at(model, X: np.ndarray, k: int) -> np.ndarray:
            return model.predict_trace(k, X, backtransform=True)
        
        if not self.separate_models:
            x_combined = self.encoder.encode(x, arm=-1)  # Encode for all arms
            post_mean_samples = _eval_at(self.model, x_combined, k)
            action_estimates = _reformat(post_mean_samples)
        else:
            # OPTIMIZATION: Use batch prediction if supported by the model (MultiChainBART)
            use_multichain = isinstance(self.models, MultiChainBART) and getattr(self.models, "n_models", 1) == self.n_arms
            
            if use_multichain and not self.is_logistic:
                # MultiChainBART batch path
                # Returns array of shape (n_arms, ...)
                batch_preds = self.models.predict_trace_batch(k, x, backtransform=True)
                action_estimates = np.array(batch_preds).flatten()
            else:
                # Standard path (loop over arms)
                action_estimates = np.zeros(self.n_arms)
                for arm in range(self.n_arms):
                    model = self.models[arm]
                    post_val = _eval_at(model, x, k)          # e.g. shape (1,) or (1, n_cat)
                    post_val = _reformat(post_val) 
                    action_estimates[arm] = float(np.ravel(post_val)[0])        
        return action_estimates

    def _reset_post_queue(self) -> None:
        """Reset the non-repeating posterior index queue after a refresh."""
        self._post_indices = []
        self._post_idx_pos = 0
        # Use any one model to infer the posterior range
        self._post_indices = list(self.model.range_post)
        self.rng.shuffle(self._post_indices)

    def _next_post_index(self) -> int:
        """Return the next posterior index to use without repetition (non-feel-good TS path).
        If the queue is exhausted or not initialized, re-initialize it."""
        if self._post_idx_pos < len(self._post_indices):
            k = self._post_indices[self._post_idx_pos]
            self._post_idx_pos += 1
            return k
        else: # Re-initialize the queue
            logger.info(f'Posterior index queue exhausted ({self.n_post} samples used), re-initializing.')
            self._reset_post_queue()
            return self._next_post_index()

    def _fg_softmax(self) -> np.ndarray:
        """Compute normalized probabilities p ~ exp(feel_good_lambda * S) via stable softmax."""
        if self._fg_S is None:
            raise ValueError("feel-good cache not initialized")
        
        logw = self.feel_good_lambda * self._fg_S
        m = float(np.max(logw))
        p = np.exp(logw - m)
        p = p / float(np.sum(p))
        return p

    def _sample_fg_post_index(self) -> int:
        """Sample a posterior draw index WITH replacement from p ~ exp(feel_good_lambda * S).
        Used for feel-good TS when feel_good_lambda != 0.0."""
        p = self._fg_softmax()
        
        # Sample a position and map to trace index k
        rp = self.model.range_post
        n_post = len(rp)
        pos = int(self.rng.choice(n_post, replace=True, p=p))
        return int(rp.start + pos * rp.step)

    def choose_arm(self, x: Union[np.ndarray, List[float]], **kwargs) -> int:
        """
        Choose an arm based on input features x.
        
        Parameters:
            x (array-like): Feature vector for which to choose an arm
            
        Returns:
            int: The index of the selected arm
        """
        # 1. Warm-start phase: use pre-generated permutation rounds
        idx = self.t - 1
        if idx < len(self._warmstart_arms):
            return int(self._warmstart_arms[idx])
            
        # 2. Fallback to random if model is not yet fitted
        if not self.is_model_fitted:
            return int(self.rng.integers(0, self.n_arms))
        
        # 3. Use BART model for arm selection
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
            
        # Store all data for potential refresh
        self.all_arms.append(arm)
        self.all_features.append(x.copy())
        self.all_rewards.append(y)
        self.all_encoded_features.append(encoded_x.copy())
        
        # Check if we should refresh the model
        refreshed = False
        if self._should_refresh():
            refreshed = self._refresh_model()
        if not refreshed and self.feel_good_lambda != 0.0 and self.is_model_fitted:
            self._feel_good_incremental_recompute(x)
        
        # Increment time counter
        self.t += 1
        
        return self

    @property
    def n_post(self) -> int:
        """Number of available posterior samples after burn-in.
        Returns 0 if model is not fitted yet.
        """
        if not self.is_model_fitted:
            return 0
        # Use model.range_post directly which now reflects total samples
        return len(self.model.range_post)

    def _preprocess_probes(self, X_probes: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
        """
        Preprocess probe covariates and return common parameters.

        Args:
            X_probes: Input probe covariates

        Returns:
            Tuple of (X_probes_processed, n_probes, n_arms, n_post)
            Returns (empty_X, n_probes, n_arms, 0) if n_post <= 0
        """
        # Ensure 2D X_probes
        X_probes = np.array(X_probes)
        if X_probes.ndim == 1:
            X_probes = X_probes.reshape(1, -1)

        n_probes = X_probes.shape[0]
        n_arms = self.n_arms
        n_post = self.n_post

        return X_probes, n_probes, n_arms, n_post

    def _posterior_f_by_arm(self, X_probes: np.ndarray, backtransform: bool = True) -> Tuple[np.ndarray, int, int, int]:
        """
        Get posterior f values with shape (n_probes, n_arms, n_post).
        Used by both posterior_draws_on_probes and feel_good_weights.

        Args:
            X_probes: Array of shape (n_probes, n_features)
            backtransform: Whether to backtransform to original scale (default True)

        Returns:
            f_by_arm: (n_probes, n_arms, n_post)
            n_probes, n_arms, n_post
        """
        X_probes, n_probes, n_arms, n_post = self._preprocess_probes(X_probes)

        if not self.separate_models:
            # Encode all probes for all arms
            encoded_list = [self.encoder.encode(X_probes[i, :], arm=-1) for i in range(n_probes)]
            X_all = np.vstack(encoded_list)  # (n_probes * n_arms, combined_dim)
            f_rows = self.model.posterior_f(X_all, backtransform=backtransform)  # (n_probes * n_arms, n_post)
            f_by_arm = f_rows.reshape(n_probes, n_arms, -1)  # (n_probes, n_arms, n_post)
        else:
            # Separate models: check for batch optimization
            use_batch = (isinstance(self.models, MultiChainBART) and 
                         getattr(self.models, "n_models", 1) == self.n_arms and
                         hasattr(self.models, 'posterior_f_batch'))
            
            if use_batch:
                # OPTIMIZED: Single parallel Ray call for all arms
                f_stack = self.models.posterior_f_batch(X_probes, backtransform=backtransform)
                # f_stack: (n_arms, n_probes, n_post)
                f_by_arm = np.transpose(f_stack, (1, 0, 2))  # (n_probes, n_arms, n_post)
            else:
                # STANDARD: Loop over arms (for list[BART] or old MultiChainBART)
                f_list = [self.models[arm].posterior_f(X_probes, backtransform=backtransform) 
                         for arm in range(n_arms)]
                f_by_arm = np.stack(f_list, axis=1)  # (n_probes, n_arms, n_post)

        return f_by_arm, n_probes, n_arms, n_post

    def posterior_draws_on_probes(self, X_probes: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute posterior draws over all arms for provided probe covariates.

        Args:
            X_probes: Array of shape (n_probes, n_features)

        Returns:
            draws: np.ndarray shaped (n_post, n_probes, n_arms)
            n_post: int, actual number of posterior draws used

        Notes:
            - For regression BART: draws correspond to posterior samples of f(x, arm).
            - For logistic BART: this method should be overridden (e.g. LogisticBARTTSAgent).
        """
        if self.is_logistic:
            raise NotImplementedError("posterior_draws_on_probes for logistic BART must be overridden.")
            
        f_by_arm, n_probes, n_arms, n_post = self._posterior_f_by_arm(X_probes)
        draws = np.transpose(f_by_arm, (2, 0, 1))  # (n_post, n_probes, n_arms)
        return draws, n_post

    def _feel_good_full_recompute(self) -> None:
        """
        Full recompute of feel-good cache S(Theta).
        Called after successful model refresh when feel_good_lambda != 0.
        """
        pass

    def _feel_good_incremental_recompute(self, x_new: np.ndarray) -> None:
        """
        Incremental update of feel-good cache with new observation x_new.
        Called after update_state when feel_good_lambda != 0 and model is fitted.
        
        Args:
            x_new: New feature vector, shape (n_features,)
        """
        pass


class DefaultBARTTSAgent(BARTTSAgent):
    """
    Refresh BART agent with default BART parameters.
    When n_chains > 1, uses MultiChainBART for ensemble modeling.
    """
    def __init__(self, n_arms: int, n_features: int,
                 initial_random_selections: int = 5,
                 random_state: int = 42,
                 encoding: str = 'multi',
                 n_chains: int = 1,
                 refresh_schedule: str = 'log',
                 bart_kwargs: Optional[Dict[str, Any]] = None,
                 feel_good_lambda: float = 0.0,
                 feel_good_eta: Optional[float] = None) -> None:
        """
        Initialize DefaultBARTTSAgent.
        
        Parameters:
            n_arms (int): Number of arms
            n_features (int): Number of features
            initial_random_selections (int): Number of warm-start selections per arm before using model.
                Total warm-start steps = n_arms * initial_random_selections.
            random_state (int): Random seed
            encoding (str): Encoding strategy ('multi', 'one-hot', 'separate')
            n_chains (int): Number of MCMC chains/ensembles
            refresh_schedule (str): Strategy to control model refresh frequency
            bart_kwargs (dict): Optional keyword arguments for BART models
            feel_good_lambda (float): Lambda for feel-good weights
            feel_good_eta (float): Optional eta for feel-good sigma2 computation
        """
        
        default_bart_kwargs: Dict[str, Any] = {
            "ndpost": 500,
            "nskip": 500,
            "n_trees": 100,
            "specification": "naive",
            "eps_nu": 1.0
        }
        
        base_ndpost, user_max_bins, merged_bart_kwargs = _prepare_bart_kwargs(default_bart_kwargs, bart_kwargs)
        
        # Compute fixed sigma2 from feel_good_eta if provided
        if feel_good_eta is not None and feel_good_eta > 0:
            fixed_sigma2 = 1.0 / (2.0 * feel_good_eta)
            merged_bart_kwargs["fixed_eps_sigma2"] = fixed_sigma2
        
        if n_chains > 1:
            def model_factory(new_ndpost=base_ndpost, max_bins=None):
                return MultiChainBART(
                    n_ensembles=n_chains,
                    bart_class=DefaultBART,
                    random_state=random_state,
                    ndpost=new_ndpost,
                    max_bins=max_bins,
                    n_models=(n_arms if encoding == 'separate' else 1),
                    **merged_bart_kwargs
                )
        else:
            def model_factory(new_ndpost=base_ndpost, max_bins=None):
                return DefaultBART(
                    random_state=random_state,
                    ndpost=new_ndpost,
                    max_bins=max_bins,
                    **merged_bart_kwargs
                )
        
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding, refresh_schedule,
                         feel_good_lambda)
        
        if user_max_bins is not None:
            self._fixed_max_bins_value = user_max_bins

    def _compute_fg_ess(self) -> float:
        """Compute Effective Sample Size (ESS) from feel-good weights: 1/sum(p^2)."""
        if self._fg_S is None:
            return float('nan')
        
        p = self._fg_softmax()
        return 1.0 / float(np.sum(p ** 2))

    def _feel_good_full_recompute(self) -> None:
        """
        Full recompute of feel-good cache S(Theta) using all historical data.
        Called after successful model refresh when feel_good_lambda != 0.
        """
        # Record ESS before refresh
        ess_before = self._compute_fg_ess()
        
        if len(self.all_features) == 0:
            # No data yet
            n_post = self.n_post
            self._fg_S = np.zeros(n_post)
            return
        
        X_probes = np.asarray(self.all_features)
        f_by_arm, _, _, _ = self._posterior_f_by_arm(X_probes, backtransform=False)
        max_over_arms = np.max(f_by_arm, axis=1)  # (n_probes, n_post)
        max_over_arms = np.minimum(0.5, max_over_arms)
        self._fg_S = np.sum(max_over_arms, axis=0)  # (n_post,)
        
        # Record ESS after refresh
        ess_after = self._compute_fg_ess()
        n_post = self.n_post
        
        if np.isnan(ess_before):
            logger.info(f"FG ESS: {ess_after:.1f}/{n_post} ({100*ess_after/n_post:.1f}%)")
        else:
            logger.info(f"FG ESS: {ess_before:.1f} -> {ess_after:.1f}/{n_post} ({100*ess_after/n_post:.1f}%)")

    def _feel_good_incremental_recompute(self, x_new: np.ndarray) -> None:
        """
        Incremental update of feel-good cache with new observation x_new.
        Called after update_state when feel_good_lambda != 0 and model is fitted.
        
        Args:
            x_new: New feature vector, shape (n_features,)
        """
        if self._fg_S is None:
            # Cache not initialized, do full recompute
            self._feel_good_full_recompute()
            return
        
        # Compute contribution of the new point
        x_new_2d = x_new.reshape(1, -1)  # (1, n_features)
        f_by_arm, _, _, _ = self._posterior_f_by_arm(x_new_2d, backtransform=False)
        max_over_arms = np.max(f_by_arm, axis=1)  # (1, n_post)
        max_over_arms = np.minimum(0.5, max_over_arms)
        delta = max_over_arms[0]  # (n_post,)
        self._fg_S += delta

    def feel_good_weights(self, lam: Callable[[int], float]):
        r"""
        Compute feel-good weights for posterior samples based on historical features.
        $$
        posterior_fg = prior * exp(-mu * residual^2) * exp(lambda * S(\Theta))
        $$
        If mu = 1/2sigma^2, then we have:
        $$
        posterior_fg = posterior * exp(lambda * S(\Theta))
        $$
        Inversely, in the feel-good TS setting, we may fix sigma = sqrt(2*mu).
        Use SIS
        $$
        S(\Theta)=\sum_{i} \min \left(b, \max _{a \in\{1, \ldots, K\}} f_{\Theta}\left(x_i, a\right)\right) \\
        w_j = \exp(\lambda S(\Theta_j))
        $$
        Here we assume b=1, i.e. [0, 1] range in the original paper. In our implementation, equivalent to [-0.5, 0.5], so we use 0.5 to clip f.
        
        Returns:
            Un-normalized log weights (log w_1, ..., log w_{n_post}) of shape (n_post,)
        """
        if self._fg_S is None:
            # Cache not initialized yet, return zeros
            return np.zeros(self.n_post)
        return lam(self.t) * self._fg_S

class LogisticBARTTSAgent(BARTTSAgent):
    """
    Refresh BART agent for binary outcomes using logistic regression.
    """
    def __init__(self, n_arms: int, n_features: int,
                 initial_random_selections: int = 5,
                 random_state: int = 42,
                 encoding: str = 'multi',
                 refresh_schedule: str = 'log',
                 bart_kwargs: Optional[Dict[str, Any]] = None,
                 feel_good_lambda: float = 0.0) -> None:
        
        default_bart_kwargs: Dict[str, Any] = {
            "ndpost": 500,
            "nskip": 500,
            "n_trees": 50
        }
        
        base_ndpost, user_max_bins, merged_bart_kwargs = _prepare_bart_kwargs(default_bart_kwargs, bart_kwargs)
        
        def model_factory(new_ndpost=base_ndpost, max_bins=None):
            return LogisticBART(
                random_state=random_state,
                ndpost=new_ndpost,
                max_bins=max_bins,
                **merged_bart_kwargs
            )
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding, refresh_schedule,
                         feel_good_lambda)
        
        if user_max_bins is not None:
            self._fixed_max_bins_value = user_max_bins

    def posterior_draws_on_probes(self, X_probes: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute posterior draws over all arms for provided probe covariates using LogisticBART.

        Args:
            X_probes: Array of shape (n_probes, n_features)

        Returns:
            draws: np.ndarray shaped (n_post, n_probes, n_arms)
            n_post: int, actual number of posterior draws used

        Notes:
            - Draws correspond to expected reward used in selection:
              * encoding in {'multi','one-hot'}: P(y=1 | x, arm)
              * encoding == 'separate': P(y=1 | x) per arm model
        """
        X_probes, n_probes, n_arms, n_post = self._preprocess_probes(X_probes)

        # Logistic BART pathway
        if not self.separate_models:
            # Encode each probe for all arms and stack rows
            encoded_list = []
            for i in range(n_probes):
                x = X_probes[i, :]
                X_enc = self.encoder.encode(x, arm=-1)  # (n_arms, combined_dim)
                encoded_list.append(X_enc)
            X_all = np.vstack(encoded_list)  # (n_probes*n_arms, combined_dim)
            model_logit = cast(LogisticBART, self.model)
            prob = model_logit.posterior_predict_proba(X_all)  # (n_rows, n_post, n_categories)
            # Select probability of positive class (category 1) per row
            if prob.shape[2] < 2:
                raise ValueError("LogisticBART with multi/one-hot encoding requires at least 2 categories.")
            p1_rows = prob[:, :, 1]  # (n_rows, n_post)
            # Reshape and transpose: (n_rows, n_post) -> (n_probes, n_arms, n_post) -> (n_post, n_probes, n_arms)
            p1_by_arm = p1_rows.reshape(n_probes, n_arms, -1)
            draws = np.transpose(p1_by_arm, (2, 0, 1))
            return draws, n_post
        else:
            # Separate logistic models per arm; use P(y=1|x) as arm value
            # Note: This assumes self.models[arm] is a single LogisticBART, not MultiChainBART (not supported yet)
            draws = np.zeros((n_post, n_probes, n_arms), dtype=float)
            for arm in range(n_arms):
                prob = self.models[arm].posterior_predict_proba(X_probes)  # (n_probes, n_post, n_categories)
                p1 = prob[:, :, 1]  # (n_probes, n_post)
                draws[:, :, arm] = p1.T
            return draws, n_post

