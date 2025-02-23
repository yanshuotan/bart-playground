
from ..util import Dataset
import copy
import numpy as np

# Reviewed

class BCFParams:
    """Trees and parameters for BCF model"""
    from .bcf_dataset import BCFDataset
    def __init__(self, mu_trees : list, tau_trees : list, global_params, mu_cache = None, tau_cache = None):
        self.mu_trees = mu_trees  # Prognostic trees
        # self.n_mu_trees = len(self.mu_trees)
        self.tau_trees = tau_trees  # Treatment effect trees
        # self.n_tau_trees = len(self.tau_trees)
        self.global_params = global_params
        # self._data = data
        
        from .bcf_util import BCFParamView
        self.mu_view = BCFParamView(self, "mu", mu_cache)
        self.tau_view = BCFParamView(self, "tau", tau_cache)
        
    @property
    def data(self):
        raise Exception("Should never be called")
        return self._data

    def copy(self, modified_mu_ids=None, modified_tau_ids=None):
        """Create a copy of BCFParams with optionally modified trees.
        
        Args:
            modified_mu_ids: indices of mu trees to copy
            modified_tau_ids: indices of tau trees to copy
        """
        if modified_mu_ids is None:
            modified_mu_ids = range(len(self.mu_trees))
        if modified_tau_ids is None:
            modified_tau_ids = range(len(self.tau_trees))
        
        copied_mu_trees = self.mu_trees.copy()
        for tree_id in modified_mu_ids:
            copied_mu_trees[tree_id] = self.mu_trees[tree_id].copy()
        new_mu = copied_mu_trees

        copied_tau_trees = self.tau_trees.copy()
        for tree_id in modified_tau_ids:
            copied_tau_trees[tree_id] = self.tau_trees[tree_id].copy()
        new_tau = copied_tau_trees

        # new_mu = [t.copy() if i in modified_mu_ids else t 
        #          for i, t in enumerate(self.mu_trees)]
        # new_tau = [t.copy() if i in modified_tau_ids else t 
        #           for i, t in enumerate(self.tau_trees)]
        
        return BCFParams(
            new_mu, new_tau,
            copy.deepcopy(self.global_params),
            mu_cache = copy.deepcopy(self.mu_view.cache),
            tau_cache = copy.deepcopy(self.tau_view.cache)
        )

    def evaluate(self, z, X=None):
        """μ(x) + z*τ(x)
        z is the treatment indicator 0-1 np.ndarray
        """
        if z is None:
            raise ValueError("z is None")
        z = z.astype(bool)

        if X is None:
            X_treated = None
        else:
            X_treated = X[z]

        # Ensure mu_view and tau_view are not None
        if self.mu_view is None or self.tau_view is None:
            raise ValueError("mu_view or tau_view is None")

        mu_pred = self.mu_view.evaluate(X=X)
        tau_pred = np.zeros_like(mu_pred)

        # Check if z has True values to avoid index error
        if np.any(z):
            evaled = self.tau_view.evaluate(X=X_treated)
            if z.shape[0] != tau_pred.shape[0]:
                raise ValueError("z.shape[0] != tau_pred.shape[0]")
            if np.sum(z) != evaled.shape[0]:
                raise ValueError("np.sum(z) != evaled.shape[0]")
            tau_pred[z] = evaled
            
        return mu_pred + z * tau_pred
        
    def update_leaf_vals(self, ensemble_id, tree_ids, leaf_vals):
        """
        Update the leaf values of specified trees in a specified ensemble.

        Parameters:
        - ensemble_id (str): Either "mu" or "tau".
        - tree_ids (list of int): List of tree IDs whose leaf values need to be updated.
        - leaf_vals (list of float): List of new leaf values to be assigned to the trees.

        Returns:
        - None
        """
        if(ensemble_id == "mu"):
            view = self.mu_view
        elif(ensemble_id == "tau"):
            view = self.tau_view

        view.update_leaf_vals(tree_ids, leaf_vals)

        # leaf_counter = 0
        # for tree_id in tree_ids:
        #     tree = trees[tree_id]
        #     tree.leaf_vals[tree.leaves] = \
        #         leaf_vals[range(leaf_counter, leaf_counter + tree.n_leaves)]
        #     leaf_counter += tree.n_leaves    
    