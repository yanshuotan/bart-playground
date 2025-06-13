import random
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
        self.ndpost = ndpost
        self.nskip = nskip
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
        if not self.is_fitted or self.data.n <= 10:
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
        self.trace = new_trace
        # self.trace = self.trace + new_trace[1:]
        
        return self
    
    @property
    def _range_post(self):
        """
        Get the range of posterior samples.
        """
        total_iterations = len(self.trace)
        if total_iterations < self.ndpost:
            raise ValueError(f"Not enough posterior samples: {total_iterations} < {self.ndpost} (provided ndpost).")
        return range(total_iterations - self.ndpost, total_iterations)
    
    def posterior_f(self, X, backtransform=True):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for i, k in enumerate(self._range_post):
            y_eval = self.trace[k].evaluate(X)
            if backtransform:
                preds[:, i] = self.preprocessor.backtransform_y(y_eval)
            else:
                preds[:, i] = y_eval
        return preds
    
    WeightSchedule = Callable[[int], float]
    def posterior_sample(self, X, schedule: WeightSchedule, backtransform=True):
        """
        Get a posterior sample of f(x) for each row in X.
        """
        pred = np.zeros((X.shape[0]))
        # sample a k using the schedule
        k = self.sampler.generator.choice(
            range(len(self.trace)), 
            p=[schedule(k) for k in range(len(self.trace))]
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
            xgb_kwargs: dict = None,
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
    

class DefaultBART(BART):

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0):
        preprocessor = DefaultPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng)
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

class ProbitBART(BART):
    """
    Binary BART implementation using Albert-Chib data augmentation and probit link.
    """

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95,
                 tree_beta: float=2.0,
                 f_k=2.0,
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42, temperature=1.0):
        preprocessor = ClassificationPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ProbitPrior(n_trees, tree_alpha, tree_beta, f_k, rng)
        temp_schedule = self._check_temperature(temperature)
        sampler = ProbitSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        super().__init__(preprocessor, sampler, ndpost, nskip)
    
    def posterior_f(self, X):
        """
        Get the posterior distribution of f(x) for each row in X.
        For binary BART, this returns the latent function values.
        Sort of categories: lexicographical, the same as np.unique
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for i, k in enumerate(self._range_post):
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
                 random_state=42, temperature=1.0):
        preprocessor = ClassificationPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = LogisticPrior(n_trees, tree_alpha, tree_beta, c, d, rng)
        temp_schedule = self._check_temperature(temperature)
        sampler = LogisticSampler(prior=prior, proposal_probs=proposal_probs, 
                               generator=rng, tol=tol, temp_schedule=temp_schedule)
        self.sampler : LogisticSampler
        super().__init__(preprocessor, sampler, ndpost, nskip)
        
    def fit(self, X, y, quietly=False):
        self.sampler.n_categories = np.unique(y).size
        super().fit(X, y, quietly=quietly)
        
    def posterior_f(self, X):
        """
        Get the posterior distribution of f(x) for each row in X.
        For logistic BART, this returns the latent function values.
        """
        preds = np.zeros((X.shape[0], self.ndpost, self.sampler.n_categories))
        for i, k in enumerate(self._range_post):
            for category in range(self.sampler.n_categories):
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
        for category in range(self.sampler.n_categories):
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
        pred = np.zeros((X.shape[0], self.sampler.n_categories))
        # sample a k using the schedule
        k = self.sampler.generator.choice(
            range(len(self.trace)), 
            p=[schedule(k) for k in range(len(self.trace))]
        )
        f_sample = np.zeros((X.shape[0], self.sampler.n_categories))
        for category in range(self.sampler.n_categories):
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
    
from joblib import Parallel, delayed    

class MultiChainBART:
    """
    Multi-chain BART model that runs multiple BART chains.
    This allows for embarrassing parallelism.
    """
    def __init__(self, n_ensembles, bart_class=DefaultBART, random_state=42, parallel=True, **kwargs):
        # Don't call super().__init__ since we manage multiple instances
        self.n_ensembles = n_ensembles
        self.bart_class = bart_class
        self.parallel = parallel

        # Generate children random states
        self.rng = np.random.default_rng(random_state)
        random_states = [self.rng.integers(0, 2**32 - 1) for _ in range(n_ensembles)]
        self.bart_instances = [bart_class(random_state=random_states[i], **kwargs) 
                              for i in range(n_ensembles)]
        
        # Initialize BART attributes
        self.is_fitted = False
        self.ndpost = self.bart_instances[0].ndpost * n_ensembles  # Combined posterior samples
        self.nskip = self.bart_instances[0].nskip
        self.preprocessor = self.bart_instances[0].preprocessor
        self.sampler = self.bart_instances[0].sampler  # Reference for compatibility
        self.trace = []
        self.data = None

    def fit(self, X, y, quietly=False):
        """Fit all BART instances."""
        if not self.parallel:
            for bart in self.bart_instances:
                bart.fit(X, y, quietly=quietly)
        else:
            # Parallel fitting using joblib
            Parallel(n_jobs=self.n_ensembles)(
                delayed(bart.fit)(X, y, quietly=quietly) for bart in self.bart_instances
            )
        self.is_fitted = True
        self.data = self.bart_instances[0].data  # All should have same preprocessed data
        
        # Combine traces for compatibility
        self.trace = []
        for bart in self.bart_instances:
            self.trace.extend(bart.trace)
    
    def predict(self, X):
        """Predict using all BART instances and average the results."""
        preds = np.array([bart.predict(X) for bart in self.bart_instances])
        return np.mean(preds, axis=0)
    
    def posterior_predict(self, X):
        """
        Get full posterior distribution from all instances.
        Returns: Array of shape (n_samples, n_ensembles * ndpost_per_instance)
        """
        preds_list = [bart.posterior_predict(X) for bart in self.bart_instances]
        return np.concatenate(preds_list, axis=1)
    
    def posterior_f(self, X, backtransform=True):
        """
        Get posterior distribution of f(x) from all instances.
        Returns: Array of shape (n_samples, n_ensembles * ndpost_per_instance)
        """
        preds_list = [bart.posterior_f(X, backtransform=backtransform) for bart in self.bart_instances]
        return np.concatenate(preds_list, axis=1)
    
    def posterior_sample(self, X, schedule, backtransform=True):
        """
        Get posterior sample from randomly selected instance and iteration.
        """
        # Randomly select an instance
        instance_idx = self.rng.integers(0, self.n_ensembles)
        return self.bart_instances[instance_idx].posterior_sample(X, schedule, backtransform)
    
    def predict_proba(self, X):
        """For classification variants - predict class probabilities."""
        if not hasattr(self.bart_instances[0], 'predict_proba'):
            raise AttributeError(f"{self.bart_class.__name__} doesn't support predict_proba")
        
        probs = np.array([bart.predict_proba(X) for bart in self.bart_instances])
        return np.mean(probs, axis=0)
    
    def posterior_predict_proba(self, X):
        """For classification variants - get full posterior distribution of probabilities."""
        if not hasattr(self.bart_instances[0], 'posterior_predict_proba'):
            raise AttributeError(f"{self.bart_class.__name__} doesn't support posterior_predict_proba")
        
        prob_samples = [bart.posterior_predict_proba(X) for bart in self.bart_instances]
        return np.concatenate(prob_samples, axis=1)
    
    def update_fit(self, X, y, add_ndpost=20, quietly=False):
        """Update all BART instances with new data points."""
        for bart in self.bart_instances:
            bart.update_fit(X, y, add_ndpost=add_ndpost, quietly=quietly)
        
        # Update combined attributes
        self.data = self.bart_instances[0].data
        self.trace = []
        for bart in self.bart_instances:
            self.trace.extend(bart.trace)
        
        return self       
    