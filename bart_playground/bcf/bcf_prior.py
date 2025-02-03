
# bcf_prior.py

from ..priors import DefaultPrior
from .bcf_util import BCFDataset
from .bcf_params import BCFParams
class BCFPrior:
    """Manages separate priors for prognostic (μ) and treatment (τ) trees"""
    def __init__(self, n_mu_trees=200, n_tau_trees=50,
                 mu_alpha=0.95, mu_beta=2.0, mu_k=2.0,
                 tau_alpha=0.25, tau_beta=3.0, tau_k=1.0,
                 eps_q=0.9, eps_nu=3, specification="linear"):
        """
            Initialize the BCF prior with specified parameters.
        """
        self.eps_q = eps_q
        self.eps_nu = eps_nu
        self.specification = specification

        # Prognostic effect prior
        self.mu_prior = DefaultPrior(
            n_trees=n_mu_trees,
            tree_alpha=mu_alpha,
            tree_beta=mu_beta,
            f_k=mu_k,
            eps_q=eps_q, # These three parameters are useless
            eps_nu=eps_nu,
            specification=specification
        )
        
        # Treatment effect prior
        self.tau_prior = DefaultPrior(
            n_trees=n_tau_trees,
            tree_alpha=tau_alpha,
            tree_beta=tau_beta,
            f_k=tau_k,
            eps_q=eps_q, # These three parameters are useless
            eps_nu=eps_nu,
            specification=specification
        )
    
    def fit(self, data : BCFDataset):
        """
        Same as BART. Fits the prior's hyperparameters to the provided dataset.
        """
        self.eps_lambda = self._fit_eps_lambda(data, self.specification)

    def init_global_params(self, data : BCFDataset):
        """
        Same as BART. Initialize global parameters for the model.

        This method samples the epsilon sigma squared (eps_sigma2) parameter 
        based on the provided data and returns it in a dictionary.
        """
        self.mu_prior.fit(data)
        eps_sigma2 = self._sample_eps_sigma2(data.X.shape[1], data.y)
        return {"eps_sigma2" : eps_sigma2}

    def resample_global_params(self, bcf_params : BCFParams):
        eps_sigma2 = self._sample_eps_sigma2(bcf_params.data.n, 
                                             bcf_params.data.y - bcf_params.evaluate())
        return {"eps_sigma2" : eps_sigma2}

    def resample_leaf_vals(self, bcf_params : BCFParams, ensemble_id, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees in a specified ensemble.
        """
        if ensemble_id == 'mu':
            return self.mu_prior.resample_leaf_vals(
                bcf_params.mu_trees, tree_ids
            )
        else:
            return self.tau_prior.resample_leaf_vals(
                bcf_params.tau_trees, tree_ids
            )

    def trees_log_prior(self, bcf_params : BCFParams, ensemble_id, tree_ids):
        """Get log prior for specified trees in a specified ensemble"""
        if ensemble_id == 'mu':
            return self.mu_prior.trees_log_prior(
                bcf_params.mu_trees, tree_ids
            )
        else:
            return self.tau_prior.trees_log_prior(
                bcf_params.tau_trees, tree_ids
            )
        
    def trees_log_marginal_lkhd(self, bcf_params : BCFParams, ensemble_id, tree_ids):
        """
        Calculate the log marginal likelihood of the trees in a specified BART model.
        """
        if ensemble_id == 'mu':
            return self.mu_prior.trees_log_marginal_lkhd(
                bcf_params.mu_trees, tree_ids
            )
        else:
            return self.tau_prior.trees_log_marginal_lkhd(
                bcf_params.tau_trees, tree_ids
            )

    def _fit_eps_lambda(self, data, specification="linear"):
        # Which prior to use does not matter
        return self.mu_prior._fit_eps_lambda(data, specification)
    
    def _sample_eps_sigma2(self, n, residuals):
        # Which prior to use does not matter
        return self.mu_prior._sample_eps_sigma2(n, residuals)
    