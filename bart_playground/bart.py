import numpy as np

from .samplers import Sampler, DefaultSampler, default_proposal_probs
from .priors import *
from .priors import *
from .util import Preprocessor, DefaultPreprocessor
from .params import Tree, Parameters

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
        self.trace = self.sampler.run(self.ndpost + self.nskip, quietly=quietly)
        self.is_fitted = True

    def update_fit(self, X, y, additional_iters=10, quietly=False):
        """
        Update an existing fitted model with new data points.
        
        Parameters:
            X: New feature data to add
            y: New target data to add
            additional_iters: Number of additional MCMC iterations to run
            quietly: Whether to suppress output
            
        Returns:
            self
        """
        if not self.is_fitted:
            # If not fitted yet, just do a regular fit
            self.fit(X, y, quietly)
            return self
        
        assert self.data, "Data has not been added yet."
        X_combined = np.vstack([self.data.X, X])
        # if old data's sample size is small, recompute thresholds and y
        if self.data.n < 10:
            y_combined = np.vstack([self.preprocessor.backtransform_y(self.data.y).reshape(-1, 1), y.reshape(-1, 1)]).flatten()
            self.data = self.preprocessor.fit_transform(X_combined, y_combined)
        # if old data's sample size is large, just update y
        else:
            # Transform new data using already fitted preprocessor
            y_new = self.preprocessor.transform_y(y)
            y_combined = np.vstack([self.data.y.reshape(-1, 1), y_new.reshape(-1, 1)]).flatten()
            self.data.X = X_combined
            self.data.y = y_combined
            
        # Update the data in the sampler
        self.sampler.add_data(self.data)
        
        # Get the last state from the trace and rebuild it with updated data
        last_state : Parameters = self.trace[-1]
        current_state = last_state.add_data_points(X)
        
        # Run the sampler for additional iterations
        new_trace = [current_state]
        
        for i in range(additional_iters):
            if not quietly and i % 10 == 0:
                print(f"Running update iteration {i}/{additional_iters}")
            current_state = self.sampler.one_iter(current_state, temp=1.0, return_trace=False)
            new_trace.append(current_state)
        
        # Update the trace
        self.trace = self.trace + new_trace
        
        return self

    def posterior_f(self, X):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for k in range(self.ndpost):
            y_eval = self.sampler.trace[self.nskip + k].evaluate(X)
            preds[:, k] = self.preprocessor.backtransform_y(y_eval)
        return preds
    
    def predict(self, X):
        """
        Predict using the BART model.
        """
        return np.mean(self.posterior_f(X), axis=1)
    
class DefaultBART(BART):

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 proposal_probs=default_proposal_probs, tol=100, max_bins=100,
                 random_state=42):
        preprocessor = DefaultPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng)
        sampler = DefaultSampler(prior = prior, proposal_probs = proposal_probs, generator = rng, tol = tol)
        super().__init__(preprocessor, sampler, ndpost, nskip)
