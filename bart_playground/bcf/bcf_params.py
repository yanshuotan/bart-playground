
# BCFParams class is a container for the trees and parameters of the BCF model. It also provides a method to evaluate the model on a given input data and treatment indicators.
import copy
import numpy as np
from typing import Optional
from ..params import Tree
from .bcf_util import BCFEnsembleIndex, EnsembleName, BCFTreeIndices

class BCFParams:
    """Trees and parameters for BCF model"""
    from .bcf_dataset import BCFDataset
    def __init__(self, mu_trees : list[Tree], tau_trees_list : list[list[Tree]], global_params : dict, 
                 mu_cache : Optional[np.ndarray] = None, tau_cache_list : Optional[list[np.ndarray]] = None):
        self.mu_trees = mu_trees  # Prognostic trees
        self.tau_trees_list = tau_trees_list  # Treatment effect trees
        self.global_params = global_params
        
        from .bcf_view import BCFParamView
        self.mu_view = BCFParamView(self, BCFEnsembleIndex(EnsembleName.MU), cache = mu_cache)
        self.tau_view_list = [BCFParamView(self, BCFEnsembleIndex(EnsembleName.TAU, i),
                                            cache = tau_cache_list[i] if tau_cache_list is not None else None
                                           ) for i in range(len(tau_trees_list))]
        
    @property
    def n_treat_arms(self):
        return len(self.tau_trees_list)

    def copy(self, modified_ids_list : Optional[list[BCFTreeIndices]] = None):
        """Create a copy of BCFParams with optionally modified trees.
        
        Args:
        - modified_ids_list (list of tuple[BCFIndex, list of int]): List of tuples specifying the ensemble and tree IDs to be updated.
        """
        modified_mu_ids = list(range(len(self.mu_trees)))
        modified_tau_ids_list = [list(range(len(self.tau_trees_list[i]))) for i in range(self.n_treat_arms)]

        if modified_ids_list is not None:
            for i in range(len(modified_ids_list)):
                ensemble_id, modified_ids = modified_ids_list[i]
                if ensemble_id is not None:
                    if ensemble_id.ensemble_name == EnsembleName.MU:
                        modified_mu_ids = modified_ids
                    elif ensemble_id.ensemble_name == EnsembleName.TAU:
                        modified_tau_ids_list[ensemble_id.index] = modified_ids
                    else:
                        raise ValueError("ensemble_id must be either BCFIndex.MU or BCFIndex.TAU")

        copied_mu_trees = self.mu_trees.copy()
        for tree_id in modified_mu_ids:
            copied_mu_trees[tree_id] = self.mu_trees[tree_id].copy()

        copied_tau_trees_list = [self.tau_trees_list[i].copy() for i in range(self.n_treat_arms)]
        for i in range(self.n_treat_arms):
            for tree_id in modified_tau_ids_list[i]:
                copied_tau_trees_list[i][tree_id] = self.tau_trees_list[i][tree_id].copy()
        
        return BCFParams(
            copied_mu_trees, copied_tau_trees_list,
            copy.deepcopy(self.global_params),
            mu_cache = copy.deepcopy(self.mu_view.cache),
            tau_cache_list = [copy.deepcopy(tau_view.cache) for tau_view in self.tau_view_list] 
        )

    def evaluate(self, z, X=None):
        """μ(x) + z*τ(x)
        z is the treatment indicator boolean np.ndarray n_sample x n_treat_arms
        X is the input data np.ndarray n_sample x n_features
        """
        if z is None:
            raise ValueError("z is None")
        z = z.astype(bool)

        mu_pred = self.mu_view.evaluate(X=X)
        tau_pred = np.zeros_like(mu_pred)

        for i, tau_view in enumerate(self.tau_view_list):
        # z_i: boolean mask for samples receiving treatment arm i.
            z_i = z[:, i]
            # Check if z has True values to avoid index error
            if np.any(z_i):
                X_treated = X[z_i] if X is not None else None
                evaled = tau_view.evaluate(X=X_treated)
                if z_i.shape[0] != tau_pred.shape[0]:
                    raise ValueError("z.shape[0] != tau_pred.shape[0]")
                if np.sum(z_i) != evaled.shape[0]:
                    raise ValueError("np.sum(z) != evaled.shape[0]")
                tau_pred[z_i] = evaled
            
        return mu_pred + tau_pred
        
    def update_leaf_vals(self, tree_ids : BCFTreeIndices, leaf_vals):
        """
        Update the leaf values of specified trees in a specified ensemble.

        Parameters:
        - ensemble_id (str): Either "mu" or "tau".
        - tree_ids (BCFTreeIndices): List of tree IDs whose leaf values need to be updated.
        - leaf_vals (list of float): List of new leaf values to be assigned to the trees.

        Returns:
        - None
        """
        if tree_ids[0].ensemble_name == EnsembleName.MU:
            view = self.mu_view
        elif tree_ids[0].ensemble_name == EnsembleName.TAU:
            view = self.tau_view_list[tree_ids[0].index]

        view.update_leaf_vals(tree_ids[1], leaf_vals)
    