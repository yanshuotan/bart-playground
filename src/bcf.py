
from .bart import BART
from .util import Dataset, DefaultPreprocessor
from .bcf_prior import BCFPrior
from .bcf_sampler import BCFSampler

import numpy as np

class BCF(BART):
    """Bayesian Causal Forest implementation"""
    def __init__(self, ndpost=1000, nskip=100,
                 n_mu_trees=200, n_tau_trees=50,
                 mu_alpha=0.95, mu_beta=2.0,
                 tau_alpha=0.25, tau_beta=3.0,
                 tau_k=1.0, **kwargs):
        
        # Initialize specialized BCF components
        preprocessor = BCFPreprocessor()
        prior = BCFPrior(n_mu_trees=n_mu_trees, n_tau_trees=n_tau_trees,
                        mu_alpha=mu_alpha, mu_beta=mu_beta,
                        tau_alpha=tau_alpha, tau_beta=tau_beta,
                        tau_k=tau_k)
        
        rng = np.random.default_rng(kwargs.get('random_state', 42))
        sampler = BCFSampler(prior, tau_update_prob=0.3, generator=rng)
        
        super().__init__(preprocessor, sampler, ndpost, nskip)

    def fit(self, X, y, z):
        """Extend fit to handle treatment indicator z"""
        data = self.preprocessor.fit_transform(X, y)
        data.z = z  # Store treatment vector
        self.sampler.prior.fit(data)
        self.sampler.add_data(data)
        self.sampler.run(self.ndpost + self.nskip)

    def predict_components(self, X):
        """Return separate mu and tau predictions"""
        post_mu = np.zeros((X.shape[0], self.ndpost))
        post_tau = np.zeros_like(post_mu)
        
        for k in range(self.ndpost):
            params = self.sampler.trace[self.nskip + k]
            post_mu[:,k] = np.sum([t.evaluate(X) for t in params.mu_trees], axis=0)
            post_tau[:,k] = np.sum([t.evaluate(X) for t in params.tau_trees], axis=0)
            
        return post_mu, post_tau

class BCFPreprocessor(DefaultPreprocessor):
    """Add treatment indicator handling"""
    def transform(self, X, y):
        dataset = super().transform(X, y)
        # Treatment indicator z should be passed separately
        return dataset
    