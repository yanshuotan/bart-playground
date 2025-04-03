from sys import prefix
from bart_playground.bcf.bcf_params import BCFParams
from .bcf_prior import BCFPrior
from .bcf_sampler import BCFSampler
from .bcf_util import BCFDataset, BCFPreprocessor

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray

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
        self._data = None        # Add this attribute for Incrementable

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

    @property
    def data(self) -> BCFDataset:
        """Return the data attribute"""
        if self._data is None:
            raise ValueError("No data. The model has not been fitted yet. Please call fit() first.")
        return self._data

    def fit(self, X, y, Z, ps : Union[bool, NDArray] = True, quietly = False):
        """Extend fit to handle treatment indicator Z"""
        self.preprocessor = BCFPreprocessor()
        self._data = self.preprocessor.fit_transform(X, y, Z, ps)  # Store data attribute
        self.sampler.add_data(self.data)
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        full_trace = self.sampler.run(self.ndpost + self.nskip, quietly=quietly)
        # Only store post burn-in samples
        self.trace = full_trace[self.nskip+1:]  
        self.is_fitted = True  # Set is_fitted flag

    def predict_all(self, X, Z, ps=None):
        """Return all mu, tau and y predictions"""
        # if self was fitted with a propensity score, we need ps to be passed
        if self.data.has_propensity and ps is None:
            if hasattr(self.preprocessor, 'ps_models') and self.preprocessor.ps_models is not None:
                # Generate propensity scores using saved models
                ps = np.zeros((X.shape[0], self.data.Z.shape[1]))
                for i, model in enumerate(self.preprocessor.ps_models):
                    ps[:, i] = model.predict_proba(X)[:, 1]
            else:
                raise ValueError("Propensity score was used during fitting. Please pass it to predict.")
            
        if ps is not None:
            if ps.ndim == 1:
                ps = ps.reshape(-1, 1)
            if ps.shape[0] != X.shape[0]:
                raise ValueError("Shape of ps does not match shape of X.")
            if ps.shape[1] != self.data.Z.shape[1]:
                raise ValueError(f"Shape of ps columns ({ps.shape[1]}) does not match number of treatment arms ({self.data.Z.shape[1]}).")
            X = np.hstack([X, ps])

        Z = self.preprocessor.transform_z(Z)
        post_mu = np.zeros((X.shape[0], self.ndpost))
        post_tau = np.zeros((X.shape[0], self.n_treat_arms, self.ndpost))
        post_y = np.zeros_like(post_mu)
        
        # debug
        # print(len(self.trace), "trace length")
        # print(self.ndpost, "ndpost")
        # print(self.nskip, "nskip")
        for k in range(self.ndpost):
            params : BCFParams = self.trace[k-self.ndpost]
            
            post_mu[:,k] = params.mu_view.evaluate(X)
            for i in range(self.n_treat_arms):
                post_tau[:, i, k] = params.tau_view_list[i].evaluate(X)
            post_y[:, k] = self.preprocessor.backtransform_y(
                    post_mu[:, k] + np.sum(Z * post_tau[:, :, k], axis = 1)
                    )
            
        return post_mu, post_tau, post_y
    
    def predict_components(self, X, Z, ps=None):
        """Return separate mu, tau and y prediction means"""
        post_mu, post_tau, post_y = self.predict_all(X, Z, ps)
        return np.mean(post_mu, axis=1), np.mean(post_tau, axis=2), np.mean(post_y, axis=1)
    
    def predict(self, X, Z, ps=None):
        """Return the mean prediction of y"""
        return self.predict_components(X, Z, ps)[2]
    
    # Should think about add_ndskip and add_ndpost in the future
    def update_fit(self, X, y, Z, ps : Union[bool, NDArray]=True, add_ndpost=20, add_nskip=10, quietly=False):
        """
        Update an existing fitted model with new data points.
        
        Parameters:
            X: New feature data to add
            y: New target data to add
            Z: New treatment indicator data (for causal models)
            ps: Propensity scores for new data (optional)
            add_ndpost: Number of posterior samples to draw
            add_nskip: Number of burn-in iterations to skip
            quietly: Whether to suppress output
            
        Returns:
            self
        """
        # If not fitted yet, or data is empty, or not enough data, just do a regular fit
        if not self.is_fitted or self.data is None or self.data.n <= 20:
            if self.is_fitted and self.data is not None:
                # Combine existing and new data
                X_combined = np.vstack((self.data.covariates, X))
                y_combined = np.hstack((self.data.y, y))
                z_combined = np.vstack((self.data.Z, Z))
                
                if isinstance(ps, bool):
                    # ps=True (always refit propensity model) 
                    # ps=False (disable propensity scores)
                    self.fit(X_combined, y_combined, z_combined, ps=ps, quietly=quietly)
                else:
                    # ps is an array
                    if self.data.has_propensity:
                        # Combine the old and new propensity scores
                        old_ps = self.data.X[:, -self.data.Z.shape[1]:]
                        if ps.ndim == 1:
                            ps = ps.reshape(-1, 1)
                        # Ensure compatible shapes
                        if ps.shape[1] != old_ps.shape[1]:
                            raise ValueError(f"Shape of ps ({ps.shape[1]}) does not match shape of existing ps ({old_ps.shape[1]}).")
                        combined_ps = np.vstack((old_ps, ps))
                        # Fit with combined data and propensity scores
                        self.fit(X_combined, y_combined, z_combined, ps=combined_ps, quietly=quietly)
                    else:
                        raise ValueError("Cannot provide propensity scores array when the base model was fit without propensity scores.")
            else:
                self.fit(X, y, Z, ps=ps, quietly=quietly)
            return self
            
        # Set all previous iterations + add_nskip as burn-in
        self.nskip += self.ndpost + add_nskip
        # Set new add_ndpost iterations as post-burn-in
        self.ndpost = add_ndpost
        
        # Update the dataset using the appropriate preprocessor method
        self._data = self.preprocessor.update_transform(X, y, Z, self.data, ps=ps)
            
        # Update thresholds
        self.sampler.add_thresholds(self.preprocessor.thresholds)
        
        # Run the sampler for additional iterations
        new_trace = self.sampler.continue_run(add_ndpost + add_nskip, new_data=self.data, quietly=quietly)
        self.trace = self.trace + new_trace[1:]
        # self.trace = self.trace + new_trace[add_nskip+1:]  # Only keep post burn-in samples
        
        return self
