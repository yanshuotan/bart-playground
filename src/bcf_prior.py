
from .priors import DefaultPrior

class BCFPrior:
    """Manages separate priors for prognostic (μ) and treatment (τ) trees"""
    def __init__(self, n_mu_trees=200, n_tau_trees=50,
                 mu_alpha=0.95, mu_beta=2.0, mu_k=2.0,
                 tau_alpha=0.25, tau_beta=3.0, tau_k=1.0,
                 eps_q=0.9, eps_nu=3, specification="linear"):
        
        # Prognostic effect prior
        self.mu_prior = DefaultPrior(
            n_trees=n_mu_trees,
            tree_alpha=mu_alpha,
            tree_beta=mu_beta,
            f_k=mu_k,
            eps_q=eps_q,
            eps_nu=eps_nu,
            specification=specification
        )
        
        # Treatment effect prior
        self.tau_prior = DefaultPrior(
            n_trees=n_tau_trees,
            tree_alpha=tau_alpha,
            tree_beta=tau_beta,
            f_k=tau_k,
            eps_q=eps_q,
            eps_nu=eps_nu,
            specification=specification
        )
    
    def fit(self, data):
        """Fit both priors to data"""
        self.mu_prior.fit(data)
        self.tau_prior.fit(data)

    def init_global_params(self, data):
        """Combine parameters from both priors"""
        return {
            **self.mu_prior.init_global_params(data),
            **self.tau_prior.init_global_params(data)
        }

    def resample_global_params(self, bcf_params):
        """Resample parameters for both components"""
        return {
            'mu_params': self.mu_prior.resample_global_params(bcf_params.mu_params),
            'tau_params': self.tau_prior.resample_global_params(bcf_params.tau_params)
        }

    def resample_leaf_vals(self, bart_params, tree_type, tree_ids):
        """Route to appropriate prior"""
        if tree_type == 'mu':
            return self.mu_prior.resample_leaf_vals(
                bart_params.mu_params, tree_ids
            )
        else:
            return self.tau_prior.resample_leaf_vals(
                bart_params.tau_params, tree_ids
            )

    def trees_log_prior(self, bart_params, tree_type, tree_ids):
        """Get log prior for specified trees"""
        if tree_type == 'mu':
            return self.mu_prior.trees_log_prior(
                bart_params.mu_params, tree_ids
            )
        else:
            return self.tau_prior.trees_log_prior(
                bart_params.tau_params, tree_ids
            )