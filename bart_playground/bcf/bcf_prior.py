
# bcf_prior.py

from ..priors import DefaultPrior
from .bcf_dataset import BCFDataset
from .bcf_params import BCFParams
from ..moves import Move
from ..params import Parameters
import numpy as np
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
      
    # TODO
    def trees_log_prior_ratio(self, move : Move, ensemble_id):
        if ensemble_id == 'mu':
            bart_prior : DefaultPrior = self.mu_prior
        else:
            bart_prior : DefaultPrior = self.tau_prior
            
        log_prior_current = bart_prior.trees_log_prior(move.current, move.trees_changed)
        log_prior_proposed = bart_prior.trees_log_prior(move.proposed, move.trees_changed)
        return log_prior_proposed - log_prior_current
    
    # TODO
    def trees_log_marginal_lkhd_ratio(self, move : Move, ensemble_id, marginalize: bool=False):
        """
        Compute the ratio of marginal likelihoods for a given move.

        Parameters:
        - move: Move
            The move to compute the marginal likelihood ratio for.
        - marginalize: bool
            Whether to marginalize over the ensemble.

        Returns:
        - float
            Marginal likelihood ratio.
        """
        if not marginalize:
            trees = move.trees_changed
        else:
            trees = np.arange(self.current.n_trees)
            
        if ensemble_id == 'mu':
            bart_prior : DefaultPrior = self.mu_prior
        else:
            bart_prior : DefaultPrior = self.tau_prior
            
        log_lkhd_current = bart_prior.trees_log_marginal_lkhd(move.current,
                                                            tree_ids= trees)
        log_lkhd_proposed = bart_prior.trees_log_marginal_lkhd(move.proposed,
                                                            tree_ids= trees)
        return log_lkhd_proposed - log_lkhd_current
    
    # copied, TODO
    def trees_log_mh_ratio(self, move : Move, ensemble_id, marginalize : bool=False):
         return self.trees_log_prior_ratio(move, ensemble_id) + \
            self.trees_log_marginal_lkhd_ratio(move, ensemble_id, marginalize)
    
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
        eps_sigma2 = self._sample_eps_sigma2(data.y) # TODO
        return {"eps_sigma2" : eps_sigma2}
    
    # use global residuals to resample global eps_sigma2
    def resample_global_params(self, bcf_params : BCFParams):
        eps_sigma2 = self._sample_eps_sigma2(bcf_params.data.n, 
                                             bcf_params.data.y - bcf_params.evaluate())
        return {"eps_sigma2" : eps_sigma2}

    def resample_leaf_vals(self, bart_params, ensemble_id, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees in a specified ensemble.
        """
        if ensemble_id == 'mu':
            bart_prior : DefaultPrior = self.mu_prior
        else:
            bart_prior : DefaultPrior = self.tau_prior
        return bart_prior.resample_leaf_vals(bart_params, tree_ids)
        
    def _fit_eps_lambda(self, data, specification="linear"):
        # Which prior to use does not matter
        return self.mu_prior._fit_eps_lambda(data, specification)
    
    def _sample_eps_sigma2(self, residuals):
        # Which prior to use does not matter
        return self.mu_prior._sample_eps_sigma2(residuals)
    