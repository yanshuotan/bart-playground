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
    
    def update_fit(self, X, y, add_ndpost=20, add_nskip=10, quietly=False):
        """
        Update an existing fitted model with new data points.
        
        Parameters:
            X: New feature data to add
            y: New target data to add
            add_ndpost: Number of posterior samples to draw
            add_nskip: Number of burn-in iterations to skip
            quietly: Whether to suppress output
            
        Returns:
            self
        """
        if not self.is_fitted or self.data is None or self.data.n <= 10:
            # If not fitted yet, or data is empty, or not enough data, just do a regular fit
            self.fit(X, y, quietly=quietly)
            return self
            
        additional_iters = add_ndpost + add_nskip
        # Set all previous iterations + add_nskip as burn-in
        self.nskip += self.ndpost + add_nskip
        # Set new add_ndpost iterations as post-burn-in
        self.ndpost = add_ndpost
        
        # Update the dataset using the appropriate preprocessor method
        self.data = self.preprocessor.update_transform(X, y, self.data)
            
        # Update thresholds 
        # if needed TODO
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        
        # Run the sampler for additional iterations
        new_trace = self.sampler.continue_run(additional_iters, new_data=self.data, quietly=quietly)
        self.trace = self.trace + new_trace[1:]
        
        return self
    
    def posterior_f(self, X):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for k in range(self.ndpost):
            y_eval = self.trace[self.nskip + k + 1].evaluate(X)
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
