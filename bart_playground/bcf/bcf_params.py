
from ..util import Dataset
import copy
import numpy as np

# Reviewed

class BCFParams:
    """Trees and parameters for BCF model"""
    def __init__(self, mu_trees : list, tau_trees : list, global_params, data : Dataset):
        self.mu_trees = mu_trees  # Prognostic trees
        self.n_mu_trees = len(self.mu_trees)
        self.tau_trees = tau_trees  # Treatment effect trees
        self.n_tau_trees = len(self.tau_trees)
        self.global_params = global_params
        self.data = data

    def copy(self, modified_mu_ids=None, modified_tau_ids=None):
        """Create a copy of BCFParams with optionally modified trees.
        
        Args:
            modified_mu_ids: indices of mu trees to copy
            modified_tau_ids: indices of tau trees to copy
        """
        if modified_mu_ids is None:
            modified_mu_ids = []
        if modified_tau_ids is None:
            modified_tau_ids = []
            
        new_mu = [t.copy() if i in modified_mu_ids else t 
                 for i, t in enumerate(self.mu_trees.trees)]
        new_tau = [t.copy() if i in modified_tau_ids else t 
                  for i, t in enumerate(self.tau_trees.trees)]
        
        return BCFParams(
            new_mu, new_tau,
            copy.deepcopy(self.global_params),
            self.data
        )

    def evaluate(self, X=None, z=None):
        """μ(x) + z*τ(x)"""
        X = self.data.X if X is None else X
        z = self.data.z if z is None else z
        
        mu_pred = np.zeros(X.shape[0])
        tau_pred = np.zeros(X.shape[0])
        for i, tree in enumerate(self.mu_trees):
            mu_pred += tree.evaluate(X)
        for i, tree in enumerate(self.tau_trees):
            tau_pred += tree.evaluate(X)
        
        return mu_pred + z * tau_pred

    def leaf_basis(self, ensemble_id, tree_ids):
        """
        Generate a horizontal stack of leaf basis arrays for the specified tree IDs in a specified ensemble.

        Parameters:
        - ensemble_id (str): Either "mu" or "tau".
        - tree_ids (list of int): List of tree IDs for which to generate the leaf basis.

        Returns:
        - numpy.ndarray: A horizontally stacked array of leaf basis arrays corresponding to the given tree IDs.
        """
        if(ensemble_id == "mu"):
            return np.hstack([self.mu_trees[tree_id].leaf_basis for tree_id in tree_ids])
        elif(ensemble_id == "tau"):
            return np.hstack([self.tau_trees[tree_id].leaf_basis for tree_id in tree_ids])
        
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
            trees = self.mu_trees
        elif(ensemble_id == "tau"):
            trees = self.tau_trees
        
        leaf_counter = 0
        for tree_id in tree_ids:
            tree = trees[tree_id]
            tree.leaf_vals[tree.leaves] = \
                leaf_vals[range(leaf_counter, leaf_counter + tree.n_leaves)]
            leaf_counter += tree.n_leaves    
    