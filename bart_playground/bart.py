from warnings import warn
import numpy as np
from typing import Optional, Callable, Dict, Any, Sequence
from scipy.stats import norm
from tqdm import tqdm

from .samplers import Sampler, DefaultSampler, MultiSampler, ProbitSampler, LogisticSampler, TemperatureSchedule, default_proposal_probs
from .priors import ComprehensivePrior, ProbitPrior, LogisticPrior
from .util import Preprocessor, DefaultPreprocessor, ClassificationPreprocessor, Dataset


class BART:
    """
    API for the BART model.
    """
    preprocessor_class = None  # Must be overridden by subclasses
    
    def __init__(self, preprocessor : Preprocessor, sampler : Sampler, 
                 ndpost=1000, nskip=100):
        """
        Initialize the BART model.
        """
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.ndpost = int(ndpost)
        self.nskip = int(nskip)
        self.trace = []
        self.is_fitted = False
        self.data = None

    def get_params(self) -> Dict[str, Any]:
        """Get effective parameters for this model instance."""
        return {"ndpost": self.ndpost, "nskip": self.nskip}

    def fit(self, X, y, quietly = False):
        """
        Fit the BART model.
        """
        data = self.preprocessor.fit_transform(X, y)
        return self.fit_with_data(data, quietly=quietly)
    
    def fit_with_data(self, data: Dataset, quietly=False):
        """
        Fit the BART model using a preprocessed dataset.
        """
        self.data = data
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.trace = self.sampler.run(self.ndpost + self.nskip, quietly=quietly, n_skip=self.nskip)
        self.is_fitted = True
        return self
    
    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        """
        Update an existing fitted model with new data points.
        
        Parameters:
            X: New feature data to add
            y: New target data to add
            add_ndpost: Number of more posterior samples to draw
            quietly: Whether to suppress output
            
        Returns:
            self
        """
        if self.data is None:
            self.fit(X, y, quietly=quietly)
            return self
        if not self.is_fitted: # or self.data.n <= 10:
            # If not fitted yet, or data is empty, or not enough data, just do a regular fit
            X_combined = np.vstack((self.data.X, X))
            y_combined = np.hstack((self.data.y, y))
            self.fit(X_combined, y_combined, quietly=quietly)
            return self

        updated_data = self.preprocessor.update_transform(X, y, self.data)
        return self.update_fit_with_data(updated_data, add_ndpost=add_ndpost, quietly=quietly)
    
    def update_fit_with_data(self, data: Dataset, add_ndpost=20, quietly=False):
        """
        Update an existing fitted model with a new preprocessed dataset.
        """
        if self.data is None or not self.is_fitted:
            return self.fit_with_data(data, quietly=quietly)
        additional_iters = add_ndpost
        # Set all previous iterations as burn-in
        self.nskip += self.ndpost
        # Set new add_ndpost iterations as post-burn-in
        self.ndpost = add_ndpost

        self.data = data
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        
        # Run the sampler for additional iterations
        new_trace = self.sampler.continue_run(additional_iters, new_data=self.data, quietly=quietly)
        # Previous samples are treated as burn-in (via nskip adjustment above), so only the latest posterior samples are kept.
        self.trace = new_trace
        
        return self
    
    @property
    def _trace_length(self):
        return len(self.trace)
    
    @property
    def range_post(self):
        """
        Get the range of posterior samples.
        """
        total_iterations = self._trace_length
        if total_iterations < self.ndpost:
            raise ValueError(f"Not enough posterior samples: {total_iterations} < {self.ndpost} (provided ndpost).")
        return range(total_iterations - self.ndpost, total_iterations)
    
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for i, k in enumerate(self.range_post):
            preds[:, i] = self.predict_trace(k, X, backtransform=backtransform)
        return preds
    
    # WeightSchedule: Callable that takes a trace index k and returns a normalized probability (sum over all k must equal 1.0)
    WeightSchedule = Callable[[int], float]
    def posterior_sample(self, X, schedule: WeightSchedule, backtransform=True):
        """
        Get a posterior sample of f(x) for each row in X.
        """
        pred = np.zeros((X.shape[0]))
        # sample a k using the schedule
        k = self.sampler.generator.choice(
            range(self._trace_length), 
            p=[schedule(k) for k in range(self._trace_length)]
        )
        y_eval = self.trace[k].evaluate(X)
        if backtransform:
            pred = self.preprocessor.backtransform_y(y_eval)
        else:
            pred = y_eval
        return pred
    
    def predict(self, X):
        """
        Predict using the BART model.
        """
        return np.mean(self.posterior_f(X), axis=1)
    
    def predict_trace(self, k: int, X, backtransform=True):
        """
        Predict using a single trace state.
        """
        y_eval = self.trace[k].evaluate(X)
        if backtransform:
            return self.preprocessor.backtransform_y(y_eval)
        else:
            return y_eval
    
    def posterior_predict(self, X):
        """
        Get the full posterior distribution of predictions.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with posterior samples
        """
        preds = self.posterior_f(X, backtransform=False)
        for k in range(self.ndpost):
            eps_sigma2 = self.trace[k].global_params['eps_sigma2']
            preds[:, k] += self.sampler.generator.normal(0, np.sqrt(eps_sigma2), size=preds[:, k].shape)
            preds[:, k] = self.preprocessor.backtransform_y(preds[:, k])
        return preds

    def init_from_xgboost(
            self,
            xgb_model,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            xgb_kwargs: dict | None = None,
            debug: bool = False
    ) -> "BART":
        # Ensure self.data is correctly populated. 
        # If X, y are different from self.data, an update or re-fit might be needed.
        # We assume that X and y are train_data.X and train_data.y,
        # and self.data is already train_data.
        if self.data is None: 
            self.data = self.preprocessor.fit_transform(X,y)
        elif X is not self.data.X or y is not self.data.y: # Check if X,y are different objects
            # This path is taken if X, y are new/different from what self.data currently holds.
            # If they are actually different datasets, a full re-fit or careful update is needed.
            print("[WARN BART.init_from_xgboost] X or y are different objects than self.data.X/y. Calling update_transform.")
            self.data = self.preprocessor.update_transform(X, y, self.data)

        dataX = self.data.X # Use self.data which should be correctly set

        from .xgb_init import fit_and_init_trees
        xgb_kwargs = xgb_kwargs or {}

        n_trees = self.sampler.tree_prior.n_trees

        model, init_trees = fit_and_init_trees(
            X, y,
            model=xgb_model,
            dataX=dataX,
            n_estimators=n_trees,
            debug=debug,
            **xgb_kwargs
        )

        self.sampler = DefaultSampler(
            prior=self.sampler.prior,
            proposal_probs=self.sampler.proposals,
            generator=self.sampler.generator,
            temp_schedule=self.sampler.temp_schedule,
            tol=self.sampler.tol,
            init_trees=init_trees
        )

        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)

        # ——— warm-start a BART draw by resampling leaf-values & global params ———
        init_state = self.sampler.get_init_state()
        if debug: # Check if debug flag is True
            print(f"[DEBUG XGB_INIT] Initial state from get_init_state():")
            print(f"[DEBUG XGB_INIT]   Tree 0 Leaf Vals (from XGB): {init_state.trees[0].leaf_vals[init_state.trees[0].leaves]}")
            print(f"[DEBUG XGB_INIT]   Global eps_sigma2: {init_state.global_params['eps_sigma2']}")

        # 1) for each tree, draw new leaf-values under BART's posterior
        for k in range(self.sampler.tree_prior.n_trees):
            new_leaf_vals = self.sampler.tree_prior.resample_leaf_vals(
                init_state,
                data_y=self.data.y,
                tree_ids=[k],
            )
            if debug:
                print(f"[DEBUG XGB_INIT] Resampled Leaf Vals for tree {k}: {new_leaf_vals}")
            init_state.update_leaf_vals([k], new_leaf_vals)
        # 2) draw the global μ/σ
        init_state.global_params = self.sampler.global_prior.resample_global_params(
            init_state,
            data_y=self.data.y
        )
        if debug:
            print(f"[DEBUG XGB_INIT] Resampled Global eps_sigma2: {init_state.global_params['eps_sigma2']}")
            print(f"[DEBUG XGB_INIT] Final state for trace - Tree 0 Leaf Vals: {init_state.trees[0].leaf_vals[init_state.trees[0].leaves]}")

        # 3) overwrite the sampler's "trace" so .run() will start from a BART-sampled state
        self.sampler.trace = [init_state]

        return self

    def _check_temperature(self, temperature):
        """
        Check if the temperature is a valid type.
        """
        is_temperature_number = type(temperature) in [float, int]
        if is_temperature_number:
            temp_func = lambda x: temperature
            return TemperatureSchedule(temp_func)
        elif type(temperature) == TemperatureSchedule:
            return temperature
        else:
            raise ValueError("Invalid temperature type ", type(temperature))
        
    def clean_trace(self, k, keep_indices=True):
        """
        Clean the trace by removing the k-th element.
        If keep_indices is True, it will set the k-th element to None and keep the originial indices.
        If keep_indices is False, it will remove the k-th element from the trace.
        """
        if not keep_indices:
            self.trace = [t for i, t in enumerate(self.trace) if i != k]
        else:
            self.trace[k] = None

class DefaultBART(BART):
    preprocessor_class = DefaultPreprocessor

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, dirichlet_prior=False, quick_decay: bool = False,
                 s_alpha: float = 1.0, fixed_eps_sigma2: Optional[float] = None,
                 init_trees=None, init_sigma2=None):
        if max_bins is None:
            max_bins = 100
        preprocessor = self.preprocessor_class(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay, s_alpha=s_alpha, fixed_eps_sigma2=fixed_eps_sigma2)
        temp_schedule = self._check_temperature(temperature)
        sampler = DefaultSampler(prior=prior, proposal_probs=proposal_probs, generator=rng, 
                                 tol=tol, temp_schedule=temp_schedule, init_trees=init_trees)
        super().__init__(preprocessor, sampler, ndpost, nskip)
        
    def get_params(self) -> Dict[str, Any]:
        """Get all effective parameters for this model instance."""
        return {
            "model_type": "DefaultBART",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "n_trees": self.sampler.tree_prior.n_trees,
            "tree_alpha": self.sampler.tree_prior.alpha,
            "tree_beta": self.sampler.tree_prior.beta,
            "f_k": self.sampler.tree_prior.f_k,
            "eps_nu": self.sampler.prior.global_prior.eps_nu,
            "eps_q": self.sampler.prior.global_prior.eps_q,
            "specification": self.sampler.prior.global_prior.specification,
            "dirichlet_prior": self.sampler.prior.global_prior.dirichlet_prior,
            "quick_decay": self.sampler.tree_prior.quick_decay,
            "proposal_probs": self.sampler.proposals,
            "fixed_eps_sigma2": self.sampler.prior.global_prior.fixed_eps_sigma2
        }
        
    def predict_proba(self, X):
        """
        DefaultBART doesn't support classification probabilities.
        Use naive prediction instead.
        Returns:
            Array of shape (n_samples, 1) with predicted values
        """
        warn("predict_proba not recommended for regression BART. Use LogisticBART for classification.")
        prob_1 = np.clip(self.predict(X).reshape(-1, 1), 0.0, 1.0)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

    def feature_inclusion_probability(self):
        """
        Compute posterior inclusion probability for each feature.

        For each posterior draw k in range_post, mark 1 if feature i is used
        at least once as a split variable in any tree (hist_k[i] > 0), else 0.
        Returns the average over posterior draws.

        Returns
        -------
        np.ndarray
            Array of shape (p,) where p is the number of features.
        """
        if not self.is_fitted or self.data is None:
            raise ValueError("Model must be fitted before computing inclusion probability.")

        p = self.data.X.shape[1]
        probs = np.zeros(p, dtype=float)

        for k in self.range_post:
            # trace[k] is Parameters for regression BART
            # vars_histogram is now a numpy array of shape (p,)
            hist = self.trace[k].vars_histogram
            if hist.size == 0:
                continue
            # Mark features that were used at least once
            probs += (hist > 0).astype(float)

        probs /= float(self.ndpost)
        return probs

    def feature_inclusion_frequency(self, normalize: str = 'split'):
        """
        Compute feature inclusion frequency (VIP-style) across posterior draws.

        Parameters
        ----------
        normalize : str, default 'split'
            - 'split': aggregate counts across draws then divide by total split count.
            - 'per_draw': normalize each draw's histogram to sum 1, then average over draws.

        Returns
        -------
        np.ndarray
            Array of shape (p,) with frequencies summing to 1 when normalize='split'.
        """
        if not self.is_fitted or self.data is None:
            raise ValueError("Model must be fitted before computing inclusion frequency.")

        if normalize not in ('split', 'per_draw'):
            raise ValueError("normalize must be one of {'split', 'per_draw'}.")

        p = self.data.X.shape[1]
        freq = np.zeros(p, dtype=float)

        if normalize == 'split':
            total_splits = 0.0
            for k in self.range_post:
                # vars_histogram is now a numpy array of shape (p,)
                hist = self.trace[k].vars_histogram
                if hist.size == 0:
                    continue
                freq += hist.astype(float)
                total_splits += float(hist.sum())
            if total_splits > 0.0:
                freq /= total_splits
            else:
                # no splits observed; return zeros
                freq[:] = 0.0
            return freq

        # per_draw: average normalized-per-draw histograms
        draws_count = 0
        for k in self.range_post:
            hist = self.trace[k].vars_histogram
            if hist.size == 0:
                continue
            draw_total = float(hist.sum())
            if draw_total <= 0.0:
                continue
            freq += hist.astype(float) / draw_total
            draws_count += 1

        if draws_count > 0:
            freq /= float(draws_count)
        else:
            freq[:] = 0.0
        return freq


class ParallelTemperingBART(BART):
    """
    Regression BART with parallel tempering (PT).

    Temperature affects likelihood only; tree/global priors are untouched.
    """
    preprocessor_class = DefaultPreprocessor

    def __init__(
        self,
        ndpost=1000,
        nskip=100,
        n_trees=200,
        tree_alpha: float = 0.95,
        tree_beta: float = 2.0,
        f_k=2.0,
        eps_q: float = 0.9,
        eps_nu: float = 3,
        specification="linear",
        proposal_probs=default_proposal_probs,
        tol=100,
        max_bins=100,
        random_state=42,
        temperatures: Optional[Sequence[float]] = None,
        n_temperatures: int = 4,
        max_temperature: float = 5.0,
        swap_interval: int = 5,
        dirichlet_prior=False,
        quick_decay: bool = False,
        s_alpha: float = 1.0,
        fixed_eps_sigma2: Optional[float] = None,
        init_trees=None,
        init_sigma2=None,
        store_chain_traces: bool = False,
        store_swap_diagnostics: bool = False,
        print_swap_diagnostics: bool = False,
    ):
        if max_bins is None:
            max_bins = 100
        if swap_interval <= 0:
            raise ValueError("swap_interval must be a positive integer.")

        preprocessor = self.preprocessor_class(max_bins=max_bins)

        seed_seq = random_state if isinstance(random_state, np.random.SeedSequence) else np.random.SeedSequence(int(random_state))
        temps = self._build_temperature_ladder(
            temperatures=temperatures,
            n_temperatures=n_temperatures,
            max_temperature=max_temperature,
        )
        child_seeds = seed_seq.spawn(len(temps))

        chain_samplers = []
        for chain_idx, chain_seed in enumerate(child_seeds):
            rng = np.random.default_rng(chain_seed)
            prior = ComprehensivePrior(
                n_trees,
                tree_alpha,
                tree_beta,
                f_k,
                eps_q,
                eps_nu,
                specification,
                rng,
                dirichlet_prior,
                quick_decay=quick_decay,
                s_alpha=s_alpha,
                fixed_eps_sigma2=fixed_eps_sigma2,
                init_sigma2=init_sigma2,
            )
            # In PT, each chain has a fixed temperature.
            chain_temp = temps[chain_idx]
            temp_schedule = TemperatureSchedule(lambda _t, _temp=chain_temp: _temp)
            sampler = DefaultSampler(
                prior=prior,
                proposal_probs=proposal_probs,
                generator=rng,
                tol=tol,
                temp_schedule=temp_schedule,
                init_trees=init_trees,
            )
            chain_samplers.append(sampler)

        # Keep BART base compatibility via the cold-chain sampler.
        super().__init__(preprocessor, chain_samplers[0], ndpost, nskip)

        self.temperatures = temps
        self.n_temperatures = len(temps)
        self.swap_interval = int(swap_interval)
        self.chain_samplers = chain_samplers
        self.store_chain_traces = bool(store_chain_traces)
        self.store_swap_diagnostics = bool(store_swap_diagnostics)
        self.print_swap_diagnostics = bool(print_swap_diagnostics)

        self.swap_attempt_counts = np.zeros(max(0, self.n_temperatures - 1), dtype=np.int64)
        self.swap_accept_counts = np.zeros(max(0, self.n_temperatures - 1), dtype=np.int64)
        self.chain_traces = [[] for _ in range(self.n_temperatures)] if self.store_chain_traces else None
        self.swap_diagnostics = []

    @staticmethod
    def _build_temperature_ladder(
        temperatures: Optional[Sequence[float]],
        n_temperatures: int,
        max_temperature: float,
    ) -> list[float]:
        if temperatures is not None:
            if len(temperatures) == 0:
                raise ValueError("temperatures cannot be empty.")
            temps = sorted(float(t) for t in temperatures)
            if any(t <= 0 for t in temps):
                raise ValueError("All temperatures must be strictly positive.")
            if temps[0] != 1.0:
                temps = [1.0] + [t for t in temps if t != 1.0]
            return temps

        if n_temperatures < 1:
            raise ValueError("n_temperatures must be >= 1.")
        if max_temperature < 1.0:
            raise ValueError("max_temperature must be >= 1.0.")
        if n_temperatures == 1:
            return [1.0]
        return list(np.geomspace(1.0, float(max_temperature), int(n_temperatures)).astype(float))

    def _compress_state_for_trace(self, state):
        state_out = state.copy(copy_cache=False)
        state_out.clear_cache()
        return state_out

    def _state_loglik_rss_eps(self, state):
        if state.cache is not None:
            fitted = state.cache
        else:
            fitted = state.evaluate()
        residuals = self.data.y - fitted
        rss = float(np.sum(residuals ** 2))
        eps_sigma2 = float(state.global_params["eps_sigma2"][0])
        if eps_sigma2 <= 0.0:
            raise ValueError("eps_sigma2 must be strictly positive.")
        n = residuals.shape[0]
        loglik = float(-0.5 * (n * np.log(eps_sigma2) + rss / eps_sigma2))
        return loglik, rss, eps_sigma2

    def _state_collapsed_loglik(self, state, temp: float) -> float:
        return float(
            self.sampler.likelihood.trees_log_marginal_lkhd(
                state,
                self.data.y,
                np.arange(state.n_trees),
                temp=temp,
            )
        )

    def _refresh_state_tempered_params(self, state, chain_id: int) -> None:
        sampler = self.chain_samplers[chain_id]
        temp = float(self.temperatures[chain_id])
        tree_ids = np.arange(state.n_trees, dtype=int)

        # Re-draw all leaf values under the chain's current temperature.
        new_leaf_vals = sampler.tree_prior.resample_leaf_vals(
            state,
            data_y=self.data.y,
            tree_ids=tree_ids,
            temp=temp,
        )
        state.update_leaf_vals(tree_ids.tolist(), new_leaf_vals)

        # Re-draw global sigma2 under the same tempered conditional.
        state.global_params = sampler.global_prior.resample_global_params(
            state,
            data_y=self.data.y,
            temp=temp,
        )

    def _attempt_adjacent_swap(
        self,
        states,
        i: int,
        j: int,
        iteration: int | None = None,
        sweep: int | None = None,
        swap_step: int | None = None,
        count_for_stats: bool = True,
    ) -> bool:
        sampler = self.chain_samplers[i]
        temp_a = float(self.temperatures[i])
        temp_b = float(self.temperatures[j])

        ll_aa = self._state_collapsed_loglik(states[i], temp_a)
        ll_bb = self._state_collapsed_loglik(states[j], temp_b)
        ll_ab = self._state_collapsed_loglik(states[j], temp_a)
        ll_ba = self._state_collapsed_loglik(states[i], temp_b)
        delta = float(ll_ab + ll_ba - ll_aa - ll_bb)

        if count_for_stats:
            self.swap_attempt_counts[i] += 1
        u = sampler.generator.uniform(0.0, 1.0)
        accepted = bool(np.log(u) < delta)

        if self.store_swap_diagnostics or self.print_swap_diagnostics:
            diag = {
                "pair_index": int(i),
                "iteration": None if iteration is None else int(iteration),
                "swap_step": None if swap_step is None else int(swap_step),
                "sweep": None if sweep is None else int(sweep),
                "temp_a": temp_a,
                "temp_b": temp_b,
                "ll_aa": ll_aa,
                "ll_bb": ll_bb,
                "ll_ab": ll_ab,
                "ll_ba": ll_ba,
                "delta": delta,
                "accepted": accepted,
            }
            if self.store_swap_diagnostics:
                self.swap_diagnostics.append(diag)
            if self.print_swap_diagnostics:
                print(
                    "[PT swap collapsed] "
                    f"iter={diag['iteration']} "
                    f"step={diag['swap_step']} "
                    f"sweep={diag['sweep']} "
                    f"pair={i}-{j} "
                    f"temp_a={temp_a:.6g} temp_b={temp_b:.6g} "
                    f"ll_aa={ll_aa:.6g} ll_bb={ll_bb:.6g} "
                    f"ll_ab={ll_ab:.6g} ll_ba={ll_ba:.6g} "
                    f"delta={delta:.6g} "
                    f"accepted={accepted}"
                )

        if accepted:
            states[i], states[j] = states[j], states[i]
            self._refresh_state_tempered_params(states[i], i)
            self._refresh_state_tempered_params(states[j], j)
            if count_for_stats:
                self.swap_accept_counts[i] += 1
            return True
        return False

    def fit(self, X, y, quietly=False):
        data = self.preprocessor.fit_transform(X, y)
        return self.fit_with_data(data, quietly=quietly)

    def fit_with_data(self, data: Dataset, quietly=False):
        self.data = data
        self.trace = []
        if self.chain_traces is not None:
            self.chain_traces = [[] for _ in range(self.n_temperatures)]
        self.swap_attempt_counts[:] = 0
        self.swap_accept_counts[:] = 0
        if self.store_swap_diagnostics:
            self.swap_diagnostics = []

        current_states = []
        for sampler in self.chain_samplers:
            sampler.add_data(self.data)
            sampler.add_thresholds(self.preprocessor.thresholds)
            current_states.append(sampler.get_init_state())

        total_iters = self.ndpost + self.nskip

        iterator = range(total_iters) if quietly else tqdm(range(total_iters), desc="Iterations")

        for it in iterator:

            # Local updates at each chain temperature.
            for chain_id, sampler in enumerate(self.chain_samplers):
                current_states[chain_id] = sampler.one_iter(
                    current_states[chain_id],
                    temp=self.temperatures[chain_id],
                    return_trace=False,
                )

            # At each swap point, run a full odd-even swap sweep of length (n_temperatures - 1)
            # so information can traverse across the ladder within one interval.
            if self.n_temperatures > 1 and ((it + 1) % self.swap_interval == 0):
                swap_step = (it + 1) // self.swap_interval
                base_offset = ((it + 1) // self.swap_interval) % 2
                for sweep in range(self.n_temperatures - 1):
                    offset = (base_offset + sweep) % 2
                    for left in range(offset, self.n_temperatures - 1, 2):
                        # Count swap rates on posterior samples only (it >= nskip).
                        in_posterior = it >= self.nskip
                        self._attempt_adjacent_swap(
                            current_states,
                            left,
                            left + 1,
                            iteration=it + 1,
                            sweep=sweep + 1,
                            swap_step=swap_step,
                            count_for_stats=in_posterior,
                        )

            if it >= self.nskip:
                self.trace.append(self._compress_state_for_trace(current_states[0]))
                if self.chain_traces is not None:
                    for chain_id in range(self.n_temperatures):
                        self.chain_traces[chain_id].append(
                            self._compress_state_for_trace(current_states[chain_id])
                        )

        self.is_fitted = True
        self.sampler = self.chain_samplers[0]
        return self

    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        # For PT, a full re-fit is the safest behavior to keep chain coupling coherent.
        warn("ParallelTemperingBART.update_fit currently refits from scratch with updated data.")
        X_combined = X if self.data is None else np.vstack((self.data.X, X))
        y_combined = y if self.data is None else np.hstack((self.data.y, y))
        self.ndpost = int(add_ndpost)
        self.nskip = 0
        return self.fit(X_combined, y_combined, quietly=quietly)

    def get_params(self) -> Dict[str, Any]:
        base = {
            "model_type": "ParallelTemperingBART",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "n_temperatures": self.n_temperatures,
            "temperatures": list(self.temperatures),
            "swap_interval": self.swap_interval,
            "store_chain_traces": self.store_chain_traces,
            "store_swap_diagnostics": self.store_swap_diagnostics,
            "print_swap_diagnostics": self.print_swap_diagnostics,
        }
        if self.swap_attempt_counts.size > 0:
            rates = np.divide(
                self.swap_accept_counts,
                np.maximum(1, self.swap_attempt_counts),
            )
            base["swap_attempts"] = self.swap_attempt_counts.tolist()
            base["swap_accepts"] = self.swap_accept_counts.tolist()
            base["swap_accept_rates"] = rates.tolist()
        if self.store_swap_diagnostics:
            base["swap_diagnostics"] = self.swap_diagnostics
        return base

    def predict_proba(self, X):
        warn("predict_proba not recommended for regression BART. Use LogisticBART for classification.")
        prob_1 = np.clip(self.predict(X).reshape(-1, 1), 0.0, 1.0)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

class ProbitBART(BART):
    """
    Binary BART implementation using Albert-Chib data augmentation and probit link.
    """
    preprocessor_class = ClassificationPreprocessor

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95,
                 tree_beta: float=2.0,
                 f_k=2.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, quick_decay: bool = False):
        preprocessor = self.preprocessor_class(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ProbitPrior(n_trees, tree_alpha, tree_beta, f_k, rng, quick_decay=quick_decay)
        temp_schedule = self._check_temperature(temperature)
        sampler = ProbitSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        super().__init__(preprocessor, sampler, ndpost, nskip)
    
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        For binary BART, this returns the latent function values.
        Sort of categories: lexicographical, the same as np.unique
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for i, k in enumerate(self.range_post):
            y_eval = self.trace[k].evaluate(X)
            preds[:, i] = y_eval
        return preds
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the probit link.
        
        Returns:
            Array of shape (n_samples, 2) with probabilities for classes 0 and 1
        """
        # Get posterior samples of probabilities
        prob_1 = self.posterior_predict_proba(X)
        
        # Average over posterior samples
        mean_prob_1 = np.mean(prob_1, axis=1)
        mean_prob_0 = 1 - mean_prob_1
        
        return np.column_stack([mean_prob_0, mean_prob_1])
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary classes.
        
        Parameters:
            X: Input features
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def posterior_predict_proba(self, X):
        """
        Get full posterior distribution of predicted probabilities.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with probability samples
        """
        f_samples = self.posterior_f(X)
        return norm.cdf(f_samples)
    
    def posterior_predict(self, X):
        """
        Get full posterior distribution of predicted classes.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with class samples
        """
        prob_samples = self.posterior_predict_proba(X)
        draws = self.sampler.generator.binomial(1, prob_samples, size=prob_samples.shape).astype(int)
        y_labels = np.zeros((draws.shape[0], draws.shape[1]), dtype=int)
        for k in range(draws.shape[1]):
            y_labels[:, k] = self.preprocessor.backtransform_y(draws[:, k])
        return y_labels
    
class LogisticBART(BART):
    """
    Logistic BART implementation using logistic link function.
    """
    preprocessor_class = ClassificationPreprocessor

    def __init__(self, ndpost=1000, nskip=100, n_trees=25, tree_alpha: float=0.95,
                 tree_beta: float=2.0, 
                 c: float = 0.0, d: float = 0.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, quick_decay: bool = False):
        if max_bins is None:
            max_bins = 100
        preprocessor = self.preprocessor_class(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = LogisticPrior(n_trees, tree_alpha, tree_beta, c, d, rng, quick_decay=quick_decay)
        temp_schedule = self._check_temperature(temperature)
        sampler = LogisticSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        self.sampler : LogisticSampler
        super().__init__(preprocessor, sampler, ndpost, nskip)

    def get_params(self) -> Dict[str, Any]:
        """Get all effective parameters for this model instance."""
        return {
            "model_type": "LogisticBART",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "n_trees": self.sampler.tree_prior.n_trees,
            "tree_alpha": self.sampler.tree_prior.alpha,
            "tree_beta": self.sampler.tree_prior.beta,
            "c": self.sampler.tree_prior.c,
            "d": self.sampler.tree_prior.d,
            "quick_decay": self.sampler.tree_prior.quick_decay,
            "proposal_probs": self.sampler.proposals
        }
        
    @property
    def n_categories(self):
        return self.sampler.n_categories
    @n_categories.setter
    def n_categories(self, value):
        self.sampler.n_categories = value
        
    def fit(self, X, y, quietly=False):
        y = y.flatten()
        self.sampler.n_categories = np.unique(y).size
        return super().fit(X, y, quietly=quietly)

    def fit_with_data(self, data: Dataset, quietly=False):
        # data.y is already encoded to 0..K-1 by ClassificationPreprocessor
        self.sampler.n_categories = int(np.max(data.y)) + 1
        return super().fit_with_data(data, quietly=quietly)
        
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        For logistic BART, this returns the latent function values.
        """
        preds = np.zeros((X.shape[0], self.ndpost, self.n_categories))
        for i, k in enumerate(self.range_post):
            for category in range(self.n_categories):
                y_eval = self.trace[k][category].evaluate(X)
                preds[:, i, category] = y_eval
        return preds
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the logistic link.
        """
        prob = self.posterior_predict_proba(X)
        
        # Average over posterior samples
        mean_prob = np.mean(prob, axis=1)
        return mean_prob
    
    def predict(self, X):
        """
        Predict classes.
        
        Parameters:
            X: Input features

        Returns:
            Class predictions
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def posterior_predict_proba(self, X):
        """
        Get full posterior distribution of predicted probabilities.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples, n_categories) with probability samples
        """
        f_samples = self.posterior_f(X)
        prob = np.zeros_like(f_samples)
        for category in range(self.n_categories):
            prob[:, :, category] = np.exp(f_samples[:, :, category])
        # Normalize to get probabilities
        prob_sum = np.sum(prob, axis=2, keepdims=True)
        prob /= prob_sum
        return prob
    
    def posterior_sample(self, X, schedule: Callable[[int], float], backtransform=False):
        """
        Get a posterior sample of predicted probabilities (posterior mean) for each row in X.
        
        Parameters:
            X: Input features
            schedule: Callable that returns a temperature for sampling
            
        Returns:
            Sampled predictions
        """
        pred = np.zeros((X.shape[0], self.n_categories))
        # sample a k using the schedule
        k = self.sampler.generator.choice(
            range(len(self.trace)), 
            p=[schedule(k) for k in range(len(self.trace))]
        )
        f_sample = np.zeros((X.shape[0], self.n_categories))
        for category in range(self.n_categories):
            f_sample[:, category] = self.trace[k][category].evaluate(X)
        prob = np.exp(f_sample)
        # Normalize to get probabilities
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob /= prob_sum
        if backtransform:
            raise NotImplementedError("Backtransform not implemented for LogisticBART")
        else:
            pred = prob
        return pred
    
    def posterior_predict(self, X):
        """
        Get full posterior distribution of predicted classes.
        
        Returns:
            Array of shape (n_samples, n_posterior_samples) with class samples
        """
        prob_samples = self.posterior_predict_proba(X)
        draws = self.sampler.generator.multinomial(
            n=1, pvals=prob_samples,
            size=(prob_samples.shape[0], prob_samples.shape[1])
        )
        labels = np.argmax(draws, axis=2)
        y_labels = np.zeros((labels.shape[0], labels.shape[1]), dtype=int)
        for k in range(labels.shape[1]):
            y_labels[:, k] = self.preprocessor.backtransform_y(labels[:, k])
        return y_labels

    def predict_trace(self, k: int, X, backtransform=True):
        """
        Predict class probabilities using a single trace state for LogisticBART.
        Returns an array shaped (n_samples, n_categories).
        """
        n_categories = self.n_categories
        f_sample = np.zeros((X.shape[0], n_categories))
        for category in range(n_categories):
            f_sample[:, category] = self.trace[k][category].evaluate(X)
        prob = np.exp(f_sample)
        prob_sum = np.sum(prob, axis=1, keepdims=True)
        prob /= prob_sum
        if backtransform:
            # Nothing to backtransform for probabilities
            return prob
        return prob
    
class MultiBART(BART):

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=default_proposal_probs, tol=1, max_bins=100,
                 random_state=42, temperature=1.0, multi_tries=10, dirichlet_prior=False, 
                 s_alpha: float = 1.0, fixed_eps_sigma2: Optional[float] = None,
                 quick_decay: bool = False, init_trees=None, init_sigma2=None):
        preprocessor = DefaultPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay, s_alpha=s_alpha, fixed_eps_sigma2=fixed_eps_sigma2, init_sigma2=init_sigma2)
        temp_schedule = self._check_temperature(temperature)
        sampler = MultiSampler(
            prior=prior, proposal_probs=proposal_probs, generator=rng, tol=tol, 
            temp_schedule=temp_schedule, multi_tries=multi_tries, init_trees=init_trees)
        super().__init__(preprocessor, sampler, ndpost, nskip)

    def predict_proba(self, X):
        """
        MultiBART doesn't support classification probabilities.
        Use naive prediction instead.
        Returns:
            Array of shape (n_samples, 1) with predicted values
        """
        warn("predict_proba not recommended for regression BART. Use LogisticBART for classification.")
        prob_1 = np.clip(self.predict(X).reshape(-1, 1), 0.0, 1.0)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

class PipelineBART(BART):
    """
    A BART model that first uses MultiSampler and then DefaultSampler.
    """
    def __init__(self, ndpost=1000, nskip=0, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, eps_nu: float=3, 
                 specification="linear", multi_proposal_probs=default_proposal_probs, 
                 proposal_probs=default_proposal_probs, tol=100, 
                 max_bins=100, random_state=42, temperature=1.0, multi_tries=10, dirichlet_prior=False, 
                 quick_decay: bool = False, init_trees=None):
        """
        Initialize the PipelineBART model.

        Parameters:
            ndpost (int): Number of posterior samples to draw.
            nskip (int): Number of burn-in iterations to skip.
            n_trees (int): Number of trees in the model.
            tree_alpha, tree_beta, f_k, eps_q, eps_nu: Prior parameters.
            specification (str): Model specification.
            proposal_probs (dict): Proposal probabilities for moves.
            tol (int): Tolerance for samplers.
            max_bins (int): Maximum number of bins for preprocessing.
            random_state (int): Random seed.
            temperature (float): Temperature for the sampler.
            multi_tries (list[int]): Multi-try MCMC parameters for MultiSampler.
        """
        # Initialize preprocessor
        preprocessor = DefaultPreprocessor(max_bins=max_bins)

        # Initialize random generator
        rng = np.random.default_rng(random_state)

        # Initialize prior
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay)

        # Initialize temperature schedule
        temp_schedule = self._check_temperature(temperature)

        # Initialize MultiSampler
        self.multi_sampler = MultiSampler(
            prior=prior,
            proposal_probs=multi_proposal_probs,
            generator=rng,
            temp_schedule=temp_schedule,
            tol=1,
            multi_tries=multi_tries, 
            init_trees=init_trees
        )

        # Initialize DefaultSampler
        self.default_sampler = DefaultSampler(
            prior=prior,
            proposal_probs=proposal_probs,
            generator=rng,
            temp_schedule=temp_schedule,
            tol=tol
        )

        # Call the parent constructor with DefaultSampler
        super().__init__(preprocessor, self.default_sampler, ndpost, nskip)

    def fit(self, X, y, multi_iter=1000, quietly=False):
        """
        Fit the PipelineBART model.

        Parameters:
            X: Feature matrix.
            y: Target vector.
            multi_iter (int): Number of iterations for MultiSampler.
            quietly (bool): Whether to suppress output.
        """
        # Step 1: Preprocess the data
        self.data = self.preprocessor.fit_transform(X, y)
        self.multi_sampler.add_data(self.data)
        self.multi_sampler.add_thresholds(self.preprocessor.thresholds)
        self.multi_iter = multi_iter

        # Step 2: Run MultiSampler
        if not quietly:
            print(f"Running MultiSampler for {self.multi_iter + self.nskip} iterations...")
        self.multi_sampler.run(self.multi_iter + self.nskip, quietly=quietly, n_skip=self.nskip)

        # Step 3: Get the final state from MultiSampler
        final_state = self.multi_sampler.trace[-1]

        # Step 4: Initialize DefaultSampler with the final state
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.sampler.trace = [final_state]

        # Step 5: Run DefaultSampler
        if not quietly:
            print(f"Running DefaultSampler for {self.ndpost} iterations...")
        self.trace = self.multi_sampler.trace[:-1] + self.sampler.run(self.ndpost, quietly=quietly)
        self.is_fitted = True

    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.multi_iter + self.ndpost))
        for k in range(self.multi_iter + self.ndpost):
            y_eval = self.trace[k].evaluate(X)
            if backtransform:
                preds[:, k] = self.preprocessor.backtransform_y(y_eval)
            else:
                preds[:, k] = y_eval
        return preds