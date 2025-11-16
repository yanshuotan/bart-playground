from warnings import warn
import numpy as np
from typing import Optional, Callable
from scipy.stats import norm

from .samplers import Sampler, DefaultSampler, ProbitSampler, LogisticSampler, TemperatureSchedule, default_proposal_probs
from .priors import ComprehensivePrior, ProbitPrior, LogisticPrior
from .util import Preprocessor, DefaultPreprocessor, ClassificationPreprocessor


class BART:
    """
    API for the BART model.
    """
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

    def fit(self, X, y, quietly = False):
        """
        Fit the BART model.
        """
        self.data = self.preprocessor.fit_transform(X, y)
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.trace = self.sampler.run(self.ndpost + self.nskip, quietly=quietly, n_skip=self.nskip)
        self.is_fitted = True
    
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

        additional_iters = add_ndpost
        # Set all previous iterations as burn-in
        self.nskip += self.ndpost
        # Set new add_ndpost iterations as post-burn-in
        self.ndpost = add_ndpost
        
        # Update the dataset using the appropriate preprocessor method
        self.data = self.preprocessor.update_transform(X, y, self.data)
            
        # Update thresholds 
        # if needed TODO
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

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, dirichlet_prior=False, quick_decay: bool = False):
        preprocessor = DefaultPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng, dirichlet_prior, quick_decay=quick_decay)
        temp_schedule = self._check_temperature(temperature)
        sampler = DefaultSampler(prior=prior, proposal_probs=proposal_probs, generator=rng, tol=tol, temp_schedule=temp_schedule)
        super().__init__(preprocessor, sampler, ndpost, nskip)
        
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
            hist = self.trace[k].vars_histogram
            if not hist:
                continue
            # Counter only contains keys with count > 0, so no need to check count
            for var_idx in hist.keys():
                if 0 <= var_idx < p:
                    probs[var_idx] += 1.0

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
                hist = self.trace[k].vars_histogram
                if not hist:
                    continue
                for var_idx, count in hist.items():
                    if 0 <= var_idx < p:
                        freq[var_idx] += float(count)
                        total_splits += float(count)
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
            if not hist:
                continue
            draw_total = float(sum(hist.values()))
            if draw_total <= 0.0:
                continue
            for var_idx, count in hist.items():
                if 0 <= var_idx < p:
                    freq[var_idx] += float(count) / draw_total
            draws_count += 1

        if draws_count > 0:
            freq /= float(draws_count)
        else:
            freq[:] = 0.0
        return freq

class ProbitBART(BART):
    """
    Binary BART implementation using Albert-Chib data augmentation and probit link.
    """

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95,
                 tree_beta: float=2.0,
                 f_k=2.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, quick_decay: bool = False):
        preprocessor = ClassificationPreprocessor(max_bins=max_bins)
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
    def __init__(self, ndpost=1000, nskip=100, n_trees=25, tree_alpha: float=0.95,
                 tree_beta: float=2.0, 
                 c: float = 0.0, d: float = 0.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0, quick_decay: bool = False):
        preprocessor = ClassificationPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = LogisticPrior(n_trees, tree_alpha, tree_beta, c, d, rng, quick_decay=quick_decay)
        temp_schedule = self._check_temperature(temperature)
        sampler = LogisticSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        self.sampler : LogisticSampler
        super().__init__(preprocessor, sampler, ndpost, nskip)
        
    @property
    def n_categories(self):
        return self.sampler.n_categories
    @n_categories.setter
    def n_categories(self, value):
        self.sampler.n_categories = value
        
    def fit(self, X, y, quietly=False):
        y = y.flatten()
        self.sampler.n_categories = np.unique(y).size
        super().fit(X, y, quietly=quietly)
        
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
    