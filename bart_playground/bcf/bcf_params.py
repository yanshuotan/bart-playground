# BCFParams class is a container for the trees and parameters of the BCF model. It also provides a method to evaluate the model on a given input data and treatment indicators.
import copy
import numpy as np
from typing import Optional
from ..params import Tree
from .bcf_util import BCFEnsembleIndex, EnsembleName, BCFTreeIndices

class BCFParams:
    """Trees and parameters for BCF model"""
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
            If provided, shallow copy the unspecified trees.
        """
        if modified_ids_list is None:
            # Deep copy all trees if no modified_ids_list provided
            return BCFParams(
                [tree.copy() for tree in self.mu_trees],
                [[tree.copy() for tree in tau_trees] for tau_trees in self.tau_trees_list],
                self.global_params,
                mu_cache = self.mu_view.cache,
                tau_cache_list = [tau_view.cache for tau_view in self.tau_view_list] 
            )
        
        # Track which tree IDs need to be deep copied
        modified_mu_ids = []
        modified_tau_ids_list = [[] for _ in range(self.n_treat_arms)]

        # Parse the modified_ids_list to determine which trees to deep copy
        for ensemble_id, modified_ids in modified_ids_list:
            if ensemble_id is not None:
                if ensemble_id.ensemble_name == EnsembleName.MU:
                    modified_mu_ids = modified_ids
                elif ensemble_id.ensemble_name == EnsembleName.TAU:
                    modified_tau_ids_list[ensemble_id.index] = modified_ids
                else:
                    raise ValueError("ensemble_id must be either BCFIndex.MU or BCFIndex.TAU")

        # Start with shallow copies of all tree lists
        copied_mu_trees = self.mu_trees.copy()
        copied_tau_trees_list = [trees_list.copy() for trees_list in self.tau_trees_list]
        
        # Deep copy only the trees that are in the modified lists
        for tree_id in modified_mu_ids:
            copied_mu_trees[tree_id] = self.mu_trees[tree_id].copy()
        
        for i in range(self.n_treat_arms):
            for tree_id in modified_tau_ids_list[i]:
                copied_tau_trees_list[i][tree_id] = self.tau_trees_list[i][tree_id].copy()
        
        # No need to deep copy global_params and cache
        # because they only contain numerical values (which are immutable)
        return BCFParams(
            copied_mu_trees, 
            copied_tau_trees_list,
            self.global_params,
            mu_cache = self.mu_view.cache,
            tau_cache_list = [tau_view.cache for tau_view in self.tau_view_list] 
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

    def add_data_points(self, X_new, z_new):
        """
        Update the MCMC state to accommodate new data points by efficiently updating
        the existing tree structures.
        
        Parameters:
            X_new: New feature data to add (np.ndarray)
            z_new: New treatment assignments (np.ndarray) with shape [n_new, n_treat_arms]
            
        Returns:
            A new BCFParams object with updated caches for the new data
        """
        # Shallow copy the tree lists and update the data points in-place
        new_mu_trees = self.mu_trees.copy()
        for tree in new_mu_trees:
            tree.add_data_points(X_new)
            
        # Ensure z_new is properly shaped
        if z_new.ndim == 1:
            z_new = z_new.reshape(-1, 1)
        
        new_tau_trees_list = []
        # For each treatment arm, update tau trees with new data from that arm
        for ensemble_idx, tau_trees in enumerate(self.tau_trees_list):
            # Shallow copy the tree list
            new_tau_trees = tau_trees.copy()
            
            treated_indices = z_new[:, ensemble_idx] == 1
            if any(treated_indices):
                # Update trees with new treated data points
                X_new_treated = X_new[treated_indices]
                for tree in new_tau_trees:
                    tree.add_data_points(X_new_treated)
            
            new_tau_trees_list.append(new_tau_trees)
        
        # Create new BCFParams object with the updated trees
        new_params = BCFParams(
            mu_trees=new_mu_trees,
            tau_trees_list=new_tau_trees_list,
            global_params=self.global_params,  # Shallow copy the global params
            mu_cache=None,  # Let BCFParams initialize the caches
            tau_cache_list=None
        )
        
        return new_params
