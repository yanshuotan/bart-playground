
from .params import Parameters
import copy

class BCFParams:
    """Extended parameters for BCF with separate components"""
    def __init__(self, mu_params, tau_params, global_params, data):
        self.mu_params = mu_params  # Parameters for prognostic trees
        self.tau_params = tau_params  # Parameters for treatment effect trees
        self.global_params = global_params
        self.data = data

    def copy(self, modified_mu_ids=None, modified_tau_ids=None):
        new_mu = [t.copy() if i in modified_mu_ids else t 
                 for i, t in enumerate(self.mu_params.trees)]
        new_tau = [t.copy() if i in modified_tau_ids else t 
                  for i, t in enumerate(self.tau_params.trees)]
        return BCFParams(
            Parameters(new_mu, self.mu_params.global_params, self.data),
            Parameters(new_tau, self.tau_params.global_params, self.data),
            copy.deepcopy(self.global_params),
            self.data
        )

    def evaluate(self, X=None, z=None):
        """BCF-specific evaluation: μ(x) + z*τ(x)"""
        X = self.data.X if X is None else X
        z = self.data.z if z is None else z
        
        mu_pred = self.mu_params.evaluate(X)
        tau_pred = self.tau_params.evaluate(X)
        return mu_pred + z * tau_pred

    @property
    def all_trees(self):
        return self.mu_params.trees + self.tau_params.trees
    