
from .bcf_params import BCFParams
from .bcf_util import BCFEnsembleIndex, EnsembleName
from ..params import Parameters
import copy

from typing import Optional
        
class BCFParamView(Parameters):
    """
    An adapter making a BCFParams ensemble look like a single-ensemble
    BART 'Parameters' object.

    'ensemble_id' is either 'mu' or 'tau'.
    """

    def __init__(self, bcf_params : BCFParams, ensemble_id : BCFEnsembleIndex, cache = None):
        self.bcf_params = bcf_params
        self.ensemble_id = ensemble_id
        self.init_cache(cache)
    
    @property
    def global_params(self):
        return self.bcf_params.global_params
    @global_params.setter
    def global_params(self, new_params):
        self.bcf_params.global_params = new_params
        
    @property
    def trees(self):
        if self.ensemble_id.ensemble_name == EnsembleName.MU:
            return self.bcf_params.mu_trees
        else: # self.ensemble_id.ensemble_name == EnsembleName.TAU:
            return self.bcf_params.tau_trees_list[self.ensemble_id.index]

    @trees.setter
    def trees(self, new_trees):
        if self.ensemble_id.ensemble_name == EnsembleName.MU:
            self.bcf_params.mu_trees = new_trees
        else: # self.ensemble_id.ensemble_name == EnsembleName.TAU:
            self.bcf_params.tau_trees_list[self.ensemble_id.index] = new_trees
            
    @property
    def n_trees(self):
        return len(self.trees)
            
    def copy(self, modified_tree_ids : Optional[list[int]] = None):
        new_bcf = None
        
        new_bcf = self.bcf_params.copy(
                modified_ids_list = [(self.ensemble_id, modified_tree_ids)] 
                if modified_tree_ids is not None else None
                )
            
        copied = BCFParamView(new_bcf, self.ensemble_id, cache=copy.deepcopy(self.cache))

        if self.ensemble_id.ensemble_name == EnsembleName.MU:
            new_bcf.mu_view = copied
        else:
            new_bcf.tau_view_list[self.ensemble_id.index] = copied

        return copied
    