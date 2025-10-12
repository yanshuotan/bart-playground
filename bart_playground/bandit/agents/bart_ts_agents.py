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
        self.models : list[BART] = []
        if self.encoding == 'separate':
            self.models = [model_factory() for _ in range(n_arms)]
        else:
            self.models = [model_factory()]
        
        # Record the maximum ndpost as the user-initialized value to cap future refreshes
        self.max_ndpost = int(getattr(self.model, 'ndpost', 0))

        # Check if model is logistic
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
        
    @property
    def model(self):
        return self.models[0]
    @model.setter
    def model(self, value):
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
        """Determine the number of posterior samples needed for the next refresh. At most self.ndpost."""
        return int(min(self.max_ndpost, max_needed))
    
    def _refresh_model(self) -> None:
        """Re-fit the model from scratch using all historical data."""
        logger.info(f't = {self.t} - re-training BART model from scratch')

        X = np.array(self.all_encoded_features)
        y = np.array(self.all_rewards)

        try:
            # Determine maximum number of actions until next refresh
            max_needed = self._steps_until_next_refresh(self.t)
            if self.separate_models:
                # Train separate models for each arm
                new_models: list[BART] = []
                for arm in range(self.n_arms):
                    arm_mask = np.array(self.all_arms) == arm
                    X_arm = X[arm_mask]
                    y_arm = y[arm_mask]
                    model = self.model_factory(self._ndpost_needed(max_needed)) 
                    model.fit(X_arm, y_arm, quietly=True)
                    new_models.append(model)

                # Only replace if all fits succeeded
                self.models = new_models

            else:
                new_model = self.model_factory(self._ndpost_needed(max_needed)) 
                new_model.fit(X, y, quietly=True)

            # Only replace on success
                self.model = new_model

            self.is_model_fitted = True

            # Reset sequential posterior index queue
            self._reset_post_queue()

        except Exception:
            logger.exception('Failed to refresh model; keeping previous model(s) in place')
            # self.models or self.model and is_model_fitted remain as they were

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
        
        # Use a precomputed, non-repeating posterior draw index
        k = self._next_post_index()
        def _eval_at(model, X: np.ndarray, k: int) -> np.ndarray:
            return model.predict_trace(k, X, backtransform=True)
        
        if not self.separate_models:
            x_combined = self.encoder.encode(x, arm=-1)  # Encode for all arms
            post_mean_samples = _eval_at(self.model, x_combined, k)
            action_estimates = _reformat(post_mean_samples)
        else:
            action_estimates = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                if self.is_model_fitted:
                    model = self.models[arm]
                    post_val = _eval_at(model, x, k)
                    action_estimates[arm] = _reformat(post_val)
                else:
                    logger.warning(f'Model for arm {arm} is not fitted, returning 0.0')
                    action_estimates[arm] = 0.0
        
        return action_estimates

    @staticmethod
    def _refresh_idx(t: int) -> int:
        """Determine the time step at which the model should be refreshed."""
        return int(np.ceil(8 * np.log(t)))

    def _should_refresh(self) -> bool:
        """Check if model should be refreshed based on time step."""
        if self.t < self.initial_random_selections or not self._has_sufficient_data():
            return False
        return self._refresh_idx(self.t) > self._refresh_idx(self.t - 1)
    
    def _steps_until_next_refresh(self, t_now: int) -> int:
        """Compute the number of selections until the next refresh trigger based on the schedule."""
        # We assume we are in steady-state (enough data and past initial random phase)
        steps = 1
        while self._refresh_idx(t_now + steps) <= self._refresh_idx(t_now + steps - 1):
            steps += 1
        return steps

    def _reset_post_queue(self) -> None:
        """Reset the non-repeating posterior index queue after a refresh."""
        self._post_indices = []
        self._post_idx_pos = 0
        # Use any one model to infer the posterior range
        self._post_indices = list(self.model.range_post)
        self.rng.shuffle(self._post_indices)

    def _next_post_index(self) -> int:
        """Return the next posterior index to use without repetition.
        If the queue is exhausted or not initialized, re-initialize it."""
        if self._post_idx_pos < len(self._post_indices):
            k = self._post_indices[self._post_idx_pos]
            self._post_idx_pos += 1
            return k
        else: # Re-initialize the queue
            logger.info(f'Posterior index queue exhausted ({self.n_post} samples used), re-initializing.')
            self._reset_post_queue()
            return self._next_post_index()

    def diagnostics_chain(self, key: str = "eps_sigma2") -> Dict[str, Any]:
        """
        Compute chain-level MCMC diagnostics for the underlying model.
        """
        if not self.is_model_fitted or self.n_post <= 0:
            return { }
        return compute_diagnostics(self.model, key=key)

    def diagnostics_probes(self, X_probes: np.ndarray) -> Dict[str, Any]:
        """
        Compute f(X) diagnostics for probes.
        Raises NotImplementedError if model is logistic and encoding is native.
        Raises ValueError if model is not fitted or has no post-burn-in draws (from `compute_diagnostics`).
        """
        if self.is_logistic:
            raise NotImplementedError("diagnostics_probes: logistic handling not implemented yet")

        # Preprocess probes and check posterior draws
        X_probes, n_probes, n_arms, n_post = self._preprocess_probes(X_probes)

        diag: dict[str, Any] = {"meta": {}, "metrics": {}, "acceptance": {}}

        if not self.separate_models:
            # Build rows for all (probe, arm) pairs
            encoded_list: List[np.ndarray] = []
            for p in range(n_probes):
                x = X_probes[p, :]
                X_enc = self.encoder.encode(x, arm=-1)  # (n_arms, combined_dim)
                encoded_list.append(X_enc)
            X_all = np.vstack(encoded_list)  # (n_probes*n_arms, combined_dim)

            raw_diag = compute_diagnostics(self.model, X=X_all)
            # raw_diag["metrics"] is a pandas DataFrame with columns [...metrics...]
            metrics_df = raw_diag["metrics"]
            # Build tidy DataFrame with probe/arm columns
            if len(metrics_df) != n_probes * n_arms:
                logger.warning("diagnostics_probes: unexpected metrics length vs n_probes*n_arms")
            probes = np.repeat(np.arange(n_probes, dtype=int), n_arms)
            arms = np.tile(np.arange(n_arms, dtype=int), n_probes)
            out_df = pd.DataFrame({"probe_idx": probes, "arm": arms})
            out_df = pd.concat([out_df, metrics_df], axis=1)

            diag["meta"]["model"] = [raw_diag["meta"]]
            diag["metrics"] = out_df
            diag["acceptance"] = raw_diag["acceptance"]
        else:
            # Separate models: run per-arm and build tidy DataFrame with probe/arm columns
            df_list: List[pd.DataFrame] = []
            diag["meta"]["model"] = []
            for a in range(n_arms):
                model_a = self.models[a]
                diag_a = compute_diagnostics(model_a, X=X_probes)
                diag["meta"]["model"].append(diag_a["meta"])
                metrics_df_a = diag_a["metrics"]
                df_a = pd.DataFrame({"probe_idx": np.arange(n_probes, dtype=int), "arm": a})
                df_a = pd.concat([df_a, metrics_df_a], axis=1)
                df_list.append(df_a)
                for move, move_acceptance in diag_a["acceptance"].items():
                    if move not in diag["acceptance"]:
                        diag["acceptance"][move] = MoveAcceptance(selected=0, proposed=0, accepted=0)
                    diag["acceptance"][move] = diag["acceptance"][move].combine(move_acceptance)
            # This dataframe is of different sort order than the one from the non-separate models
            # This is fine for most purposes; the caller should not rely on the order
            diag["metrics"] = pd.concat(df_list, ignore_index=True)

        diag["meta"]["n_probes"] = n_probes
        diag["meta"]["n_arms"] = n_arms
        diag["meta"]["n_post"] = n_post

        return diag
    
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

    @property
    def n_post(self) -> int:
        """Number of available posterior samples after burn-in.
        Returns 0 if model is not fitted yet.
        """
        try:
            if not self.is_model_fitted:
                return 0
            # Use any one model to infer posterior range
            n_post_per_chain = len(list(self.model.range_post))
            # If using MultiChainBART, effective posterior samples equal per-chain times number of ensembles
            if isinstance(self.model, MultiChainBART):
                return int(n_post_per_chain * self.model.n_ensembles)
            return int(n_post_per_chain)
        except Exception:
            return 0

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

    def _get_empty_draws(self, n_probes: int, n_arms: int) -> Tuple[np.ndarray, int]:
        """Return empty draws array when model is not fitted."""
        logger.warning('Model is not fitted, returning empty draws for probes')
        return np.zeros((0, n_probes, n_arms), dtype=float), 0

    def _rows_to_draws(self, rows: np.ndarray, n_probes: int, n_arms: int) -> np.ndarray:
        """
        Helper to reshape stacked rows (n_probes * n_arms, n_post) -> (n_post, n_probes, n_arms)

        Args:
            rows: Array of shape (n_probes * n_arms, n_post)
            n_probes: Number of probe points
            n_arms: Number of arms

        Returns:
            Array of shape (n_post, n_probes, n_arms)
        """
        arr = rows.reshape(n_probes, n_arms, -1)  # -1 infers n_post
        return np.transpose(arr, (2, 0, 1))

    @abstractmethod
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
            - For logistic BART: draws correspond to expected reward used in selection:
              * encoding in {'multi','one-hot'}: P(y=1 | x, arm)
              * encoding == 'separate': P(y=1 | x) per arm model
              * encoding == 'native' (binary arms only): category probabilities per arm
            - Returns (empty array with shape (0, n_probes, n_arms), 0) when not fitted.
        """
        pass


class DefaultBARTTSAgent(BARTTSAgent):
    """
    Refresh BART agent with default BART parameters.
    When n_chains > 1, uses MultiChainBART for ensemble modeling.
    """
    def __init__(self, n_arms: int, n_features: int,
                 ndpost: int = 1000, nskip: int = 100,
                 n_trees: int = 200,
                 dirichlet_prior: bool = False,
                 initial_random_selections: int = 10,
                 random_state: int = 42,
                 encoding: str = 'multi',
                 n_chains: int = 1,
                 tree_alpha: float = 0.95,
                 tree_beta: float = 2.0,
                 quick_decay: bool = False) -> None:
        
        if n_chains > 1:
            # Use MultiChainBART for ensemble modeling
            model_factory = lambda new_ndpost=ndpost: MultiChainBART(
                n_ensembles=n_chains,
                bart_class=DefaultBART,
                n_trees=n_trees,
                ndpost=new_ndpost,
                nskip=nskip,
                random_state=random_state,
                proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1},
                dirichlet_prior=dirichlet_prior,
                tree_alpha=tree_alpha,
                tree_beta=tree_beta,
                quick_decay=quick_decay
            )
        else:
            # Use single DefaultBART instance
            model_factory = lambda new_ndpost=ndpost: DefaultBART(
                n_trees=n_trees,
                ndpost=new_ndpost,
                nskip=nskip,
                random_state=random_state,
                proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1},
                dirichlet_prior=dirichlet_prior,
                tree_alpha=tree_alpha,
                tree_beta=tree_beta,
                quick_decay=quick_decay
            )
        
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding)

    def posterior_draws_on_probes(self, X_probes: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Compute posterior draws over all arms for provided probe covariates using DefaultBART.

        Args:
            X_probes: Array of shape (n_probes, n_features)

        Returns:
            draws: np.ndarray shaped (n_post, n_probes, n_arms)
            n_post: int, actual number of posterior draws used

        Notes:
            - Draws correspond to posterior samples of f(x, arm).
            - Returns (empty array with shape (0, n_probes, n_arms), 0) when not fitted.
        """
        X_probes, n_probes, n_arms, n_post = self._preprocess_probes(X_probes)

        if n_post <= 0:
            return self._get_empty_draws(n_probes, n_arms)

        # Regression BART pathway (DefaultBART):
        if not self.separate_models:
            # Encode each probe for all arms and stack
            encoded_list: list[np.ndarray] = []
            for i in range(n_probes):
                x = X_probes[i, :]
                X_enc = self.encoder.encode(x, arm=-1)  # (n_arms, combined_dim)
                encoded_list.append(X_enc)
            X_all = np.vstack(encoded_list)  # (n_probes*n_arms, combined_dim)
            # Posterior samples f(X)
            f_rows = self.model.posterior_f(X_all, backtransform=True)  # (n_rows, n_post)
            # Convert to (n_post, n_probes, n_arms)
            draws = self._rows_to_draws(f_rows, n_probes, n_arms)
            return draws, n_post
        else:
            # Separate models per arm
            draws = np.zeros((n_post, n_probes, n_arms), dtype=float)
            for arm in range(n_arms):
                model = self.models[arm]
                f = model.posterior_f(X_probes, backtransform=True)  # (n_probes, n_post)
                n_post_model = int(f.shape[1])
                if n_post_model != n_post:
                    logger.error(f"posterior_f produced {n_post_model} draws for arm {arm}, expected {n_post}")
                draws[:, :, arm] = f.T  # (n_post, n_probes)
            return draws, n_post


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
        
        model_factory = lambda new_ndpost=ndpost: LogisticBART(
            n_trees=n_trees,
            ndpost=new_ndpost,
            nskip=nskip,
            random_state=random_state,
            proposal_probs={"grow": 0.4, "prune": 0.4, "change": 0.1, "swap": 0.1}
        )
        super().__init__(n_arms, n_features, model_factory,
                         initial_random_selections, random_state, encoding)

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
              * encoding == 'native' (binary arms only): category probabilities per arm
            - Returns (empty array with shape (0, n_probes, n_arms), 0) when not fitted.
        """
        X_probes, n_probes, n_arms, n_post = self._preprocess_probes(X_probes)

        if n_post <= 0:
            return self._get_empty_draws(n_probes, n_arms)

        # Logistic BART pathway
        if not self.separate_models:
            if self.encoding in ['multi', 'one-hot']:
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
                draws = self._rows_to_draws(p1_rows, n_probes, n_arms)
                return draws, n_post
            else:
                raise NotImplementedError(f"Unsupported encoding for logistic BARTTSAgent probes: {self.encoding}")
        else:
            # Separate logistic models per arm; use P(y=1|x) as arm value
            draws = np.zeros((n_post, n_probes, n_arms), dtype=float)
            for arm in range(n_arms):
                model = cast(LogisticBART, self.models[arm])
                prob = model.posterior_predict_proba(X_probes)  # (n_probes, n_post, n_categories)
                p1 = prob[:, :, 1]  # (n_probes, n_post)
                draws[:, :, arm] = p1.T
            return draws, n_post
