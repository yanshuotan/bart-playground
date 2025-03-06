import numpy as np

from .samplers import Sampler, DefaultSampler, default_proposal_probs, NTreeSampler
from .priors import *
from .util import Preprocessor, DefaultPreprocessor

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

    def fit(self, X, y):
        """
        Fit the BART model.
        """
        data = self.preprocessor.fit_transform(X, y)
        self.sampler.add_data(data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.trace = self.sampler.run(self.ndpost + self.nskip)

    def posterior_f(self, X):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        for k in range(self.ndpost):
            preds[:, k] = self.preprocessor.backtransform_y(
                self.sampler.trace[self.nskip + k].evaluate(X))
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


class ChangeNumTreeBART(BART):

    def __init__(self, ndpost=1000, nskip=100, n_trees=200, tree_alpha: float=0.95, 
                 tree_beta: float=2.0, f_k=2.0, eps_q: float=0.9, 
                 eps_nu: float=3, specification="linear", 
                 theta_0 = 200, theta_df = 100, tau_k = 2.0,
                 proposal_probs=default_proposal_probs, break_prob: float=0.5, tol=100, max_bins=100,
                 random_state=42):
        preprocessor = DefaultPreprocessor(max_bins=max_bins)
        rng = np.random.default_rng(random_state)
        prior = ComprehensivePrior(n_trees, tree_alpha, tree_beta, f_k, eps_q, 
                             eps_nu, specification, rng, theta_0, theta_df, tau_k)
        sampler = NTreeSampler(prior = prior, proposal_probs = proposal_probs, break_prob = break_prob, generator = rng, tol = tol)
        super().__init__(preprocessor, sampler, ndpost, nskip)