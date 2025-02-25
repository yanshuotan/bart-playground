
# bcf_prior.py

from ..priors import *
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
                 eps_q=0.9, eps_nu=3, specification="linear", generator=np.random.default_rng()):
        """
            Initialize the BCF prior with specified parameters.
        """
        # self.eps_q = eps_q
        # self.eps_nu = eps_nu
        # self.specification = specification

        # Prognostic effect prior
        self.mu_prior = TreesPrior(
            n_trees=n_mu_trees,
            tree_alpha=mu_alpha,
            tree_beta=mu_beta,
            f_k=mu_k,
            generator=generator
        )
        
        # Treatment effect prior
        self.tau_prior = TreesPrior(
            n_trees=n_tau_trees,
            tree_alpha=tau_alpha,
            tree_beta=tau_beta,
            f_k=tau_k,
            generator=generator
        )
        self.global_prior = GlobalParamPrior(eps_q, eps_nu, specification, generator=generator)

        self.mu_likelihood = BARTLikelihood(self.mu_prior.f_sigma2)
        self.tau_likelihood = BARTLikelihood(self.tau_prior.f_sigma2)
      
    def trees_log_prior_ratio(self, move : Move, ensemble_id):
        if ensemble_id == 'mu':
            bart_prior : TreesPrior = self.mu_prior
        else:
            bart_prior : TreesPrior = self.tau_prior
            
        log_prior_current = bart_prior.trees_log_prior(move.current, move.trees_changed)
        log_prior_proposed = bart_prior.trees_log_prior(move.proposed, move.trees_changed)
        return log_prior_proposed - log_prior_current
    
    def trees_log_marginal_lkhd_ratio(self, move : Move, data_y, ensemble_id, marginalize: bool=False):
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
            trees = np.arange(move.current.n_trees)
            
        if ensemble_id == 'mu':
            bart_likelihood : BARTLikelihood = self.mu_likelihood
        else:
            bart_likelihood : BARTLikelihood = self.tau_likelihood
            
        log_lkhd_current = bart_likelihood.trees_log_marginal_lkhd(move.current, data_y,
                                                            tree_ids= trees)
        log_lkhd_proposed = bart_likelihood.trees_log_marginal_lkhd(move.proposed, data_y,
                                                            tree_ids= trees)
        return log_lkhd_proposed - log_lkhd_current
    
    def trees_log_mh_ratio(self, move : Move, data_y, ensemble_id, marginalize : bool=False):
         return self.trees_log_prior_ratio(move, ensemble_id) + \
            self.trees_log_marginal_lkhd_ratio(move, data_y, ensemble_id, marginalize)
    
    def init_global_params(self, data : BCFDataset):
        """
        Same as BART. Initialize global parameters for the model.

        This method samples the epsilon sigma squared (eps_sigma2) parameter 
        based on the provided data and returns it in a dictionary.
        """
        self.global_prior.fit_hyperparameters(data)
        eps_sigma2 = self.global_prior._sample_eps_sigma2(data.y)
        return {"eps_sigma2" : eps_sigma2}
    
    # use global residuals to resample global eps_sigma2
    def resample_global_params(self, bcf_params : BCFParams, data_y, treated):
        eps_sigma2 = self.global_prior._sample_eps_sigma2(data_y - bcf_params.evaluate(z = treated))
        return {"eps_sigma2" : eps_sigma2}

    def resample_leaf_vals(self, bart_params, data_y, ensemble_id, tree_ids):
        """
        Resample the values of the leaf nodes for the specified trees in a specified ensemble.
        """
        if ensemble_id == 'mu':
            bart_prior : TreesPrior = self.mu_prior
        else:
            bart_prior : TreesPrior = self.tau_prior
        return bart_prior.resample_leaf_vals(bart_params, data_y, tree_ids)
        