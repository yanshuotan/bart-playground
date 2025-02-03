
from ..bart import BART
from ..util import Dataset, DefaultPreprocessor
from .bcf_prior import BCFPrior
from .bcf_sampler import BCFSampler
from .bcf_util import BCFDataset

import numpy as np

class BCF:
    def __init__(self, n_mu_trees=200, n_tau_trees=50,
                 mu_alpha=0.95, mu_beta=2.0, mu_k=2.0,
                 tau_alpha=0.25, tau_beta=3.0, tau_k=1.0,
                 ndpost=1000, nskip=100, random_state=42):
        self.ndpost = ndpost
        self.nskip = nskip
        self.random_state = random_state
        self.eps_lambda = None

        # Initialize priors
        self.prior = BCFPrior(
            n_mu_trees=n_mu_trees,
            n_tau_trees=n_tau_trees,
            mu_alpha=mu_alpha,
            mu_beta=mu_beta,
            mu_k=mu_k,
            tau_alpha=tau_alpha,
            tau_beta=tau_beta,
            tau_k=tau_k
        )
        
        # Initialize sampler
        rng = np.random.default_rng(random_state)
        self.sampler = BCFSampler(
            prior=self.prior,
            proposal_probs = {'grow':0.5, 'prune':0.5},
            generator = rng # ,
            # proposal_probs_tau={'grow':0.3, 'prune':0.3, 'change':0.4}
        )

    def fit(self, X, y, z):
        """Extend fit to handle treatment indicator z"""
        data = BCFPreprocessor.fit_transform(X, y, z)
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
    @staticmethod
    def fit_transform(X, y, z):
        dp = DefaultPreprocessor()
        dataset = dp.fit_transform(X, y)
        return BCFDataset(dataset.X, dataset.y, z, dataset.thresholds)
    