from bart_playground.bcf.bcf_params import BCFParams
from .bcf_prior import BCFPrior
from .bcf_sampler import BCFSampler
from .bcf_util import BCFPreprocessor

import numpy as np

class BCF:
    def __init__(self, n_treat_arms:int, n_mu_trees=200, n_tau_trees : list[int] = [],
                 mu_alpha=0.95, mu_beta=2.0, mu_k=2.0,
                 tau_alpha=0.25, tau_beta=3.0, tau_k=1.0,
                 ndpost=1000, nskip=100, random_state=42):
        self.ndpost = ndpost
        self.nskip = nskip
        self.random_state = random_state
        self.eps_lambda = None
        self.is_fitted = False  # Add this attribute for Incrementable
        self.trace = []         # Add this attribute for Incrementable
        self.data = None        # Add this attribute for Incrementable

        if n_tau_trees == []:
            n_tau_trees = [50] * n_treat_arms
        self.n_treat_arms = n_treat_arms

        # Initialize priors
        rng = np.random.default_rng(random_state)
        self.prior = BCFPrior(
            n_treat_arms=n_treat_arms,
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

    def fit(self, X, y, z, quietly = False):
        """Extend fit to handle treatment indicator z"""
        self.preprocessor = BCFPreprocessor()
        self.data = self.preprocessor.fit_transform(X, y, z)  # Store data attribute
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        self.trace = self.sampler.run(self.ndpost + self.nskip, quietly=quietly)  # Store trace
        self.is_fitted = True  # Set is_fitted flag

    def predict_all(self, X, Z):
        """Return all mu, tau and y predictions"""
        Z = Z.reshape(-1, self.n_treat_arms)
        post_mu = np.zeros((X.shape[0], self.ndpost))
        post_tau = np.zeros((X.shape[0], self.n_treat_arms, self.ndpost))
        post_y = np.zeros_like(post_mu)
        
        for k in range(self.ndpost):
            params : BCFParams = self.trace[self.nskip + k + 1]
            
            # np.sum([t.evaluate(X) for t in params.mu_trees], axis=0)
            post_mu[:,k] = params.mu_view.evaluate(X)
            for i in range(self.n_treat_arms):
                post_tau[:, i, k] = params.tau_view_list[i].evaluate(X)
            post_y[:, k] = self.preprocessor.backtransform_y(
                    post_mu[:, k] + np.sum(Z * post_tau[:, :, k], axis = 1)
                    )
            
        return post_mu, post_tau, post_y
    
    def predict_components(self, X, Z):
        """Return separate mu, tau and y prediction means"""
        post_mu, post_tau, post_y = self.predict_all(X, Z)
        return np.mean(post_mu, axis=1), np.mean(post_tau, axis=2), np.mean(post_y, axis=1)
    
    def predict(self, X, Z):
        """Return the mean prediction of y"""
        return self.predict_components(X, Z)[2]
    
    def update_fit(self, X, y, z, add_ndpost=20, add_nskip=10, quietly=False):
        """
        Update an existing fitted model with new data points.
        
        Parameters:
            X: New feature data to add
            y: New target data to add
            z: New treatment indicator data (for causal models)
            add_ndpost: Number of posterior samples to draw
            add_nskip: Number of burn-in iterations to skip
            quietly: Whether to suppress output
            
        Returns:
            self
        """
        if not self.is_fitted:
            # If not fitted yet, just do a regular fit
            self.fit(X, y, z, quietly=quietly)
            return self
            
        additional_iters = add_ndpost + add_nskip
        # Set all previous iterations + add_nskip as burn-in
        self.nskip += self.ndpost + add_nskip
        # Set new add_ndpost iterations as post-burn-in
        self.ndpost = add_ndpost
        
        # Update the dataset using the appropriate preprocessor method
        self.data = self.preprocessor.update_transform(X, y, z, self.data)
            
        # Update thresholds 
        # if needed TODO
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        
        # Run the sampler for additional iterations
        new_trace = self.sampler.continue_run(additional_iters, new_data=self.data, quietly=quietly)
        self.trace = self.trace + new_trace[1:]
        
        return self
    