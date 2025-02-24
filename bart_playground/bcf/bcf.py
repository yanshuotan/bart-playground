
from ..bart import BART
from ..util import Dataset, DefaultPreprocessor
from .bcf_prior import BCFPrior
from .bcf_sampler import BCFSampler
from .bcf_dataset import BCFDataset

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
        rng = np.random.default_rng(random_state)
        self.prior = BCFPrior(
            n_mu_trees=n_mu_trees,
            n_tau_trees=n_tau_trees,
            mu_alpha=mu_alpha,
            mu_beta=mu_beta,
            mu_k=mu_k,
            tau_alpha=tau_alpha,
            tau_beta=tau_beta,
            tau_k=tau_k,
            generator=rng
        )
        
        # Initialize sampler
        self.sampler = BCFSampler(
            prior=self.prior,
            proposal_probs = {'grow':0.5, 'prune':0.5},
            generator = rng # ,
            # proposal_probs_tau={'grow':0.3, 'prune':0.3, 'change':0.4}
        )

    def fit(self, X, y, z):
        """Extend fit to handle treatment indicator z"""
        self.preprocessor = BCFPreprocessor()
        data = self.preprocessor.fit_transform(X, y, z)
        data.z = z  # Store treatment vector
        # self.sampler.prior.fit(data)
        self.sampler.add_data(data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.sampler.run(self.ndpost + self.nskip)

    def predict_all(self, X, Z):
        """Return all mu, tau and y predictions"""
        post_mu = np.zeros((X.shape[0], self.ndpost))
        post_tau = np.zeros_like(post_mu)
        post_y = np.zeros_like(post_mu)
        
        for k in range(self.ndpost):
            params = self.sampler.trace[self.nskip + k]
            
            # np.sum([t.evaluate(X) for t in params.mu_trees], axis=0)
            post_mu[:,k] = params.mu_view.evaluate(X)
            post_tau[:, k] = params.tau_view.evaluate(X)
            post_y[:, k] = self.preprocessor.backtransform_y(
                post_mu[:, k] + Z * post_tau[:, k]
                )
            
        return post_mu, post_tau, post_y
    
    def predict_components(self, X, Z):
        """Return separate mu, tau and y prediction means"""
        post_mu, post_tau, post_y = self.predict_all(X, Z)
        return np.mean(post_mu, axis=1), np.mean(post_tau, axis=1), np.mean(post_y, axis=1)
    
    def predict(self, X, Z):
        """Return the mean prediction of y"""
        return self.predict_components(X, Z)[2]

class BCFPreprocessor(DefaultPreprocessor):
    def fit_transform(self, X, y, z):
        dataset = super().fit_transform(X, y)
        return BCFDataset(dataset.X, dataset.y, z)
    