from abc import ABC, abstractmethod
from ..util import Dataset, DefaultPreprocessor
import numpy as np
from .bcf_params import BCFParams
from ..params import Parameters
import copy
        
class BCFParamView(Parameters):
    """
    An adapter making a BCFParams ensemble look like a single-ensemble
    BART 'Parameters' object.

    'ensemble_id' is either 'mu' or 'tau'.
    """

    def __init__(self, bcf_params : BCFParams, ensemble_id, cache=None):
        self.bcf_params = bcf_params
        self.ensemble_id = ensemble_id
        self.init_cache(cache)
        # self._internal_count = 0
        # self._init_treated_data()
    
    def _init_treated_data(self):
        raise Exception("Should never be called")
        parent_data = self.bcf_params.data
        
        self.context_treated = parent_data.X[parent_data.treated, :]
        
        prep = DefaultPreprocessor()
        self.thresholds_treated = prep.gen_thresholds(self.context_treated)
    
    @property
    def global_params(self):
        return self.bcf_params.global_params
    @global_params.setter
    def global_params(self, new_params):
        self.bcf_params.global_params = new_params
        
    @property
    def trees(self):
        if self.ensemble_id == 'mu':
            return self.bcf_params.mu_trees
        else:
            return self.bcf_params.tau_trees

    @trees.setter
    def trees(self, new_trees):
        if self.ensemble_id == 'mu':
            self.bcf_params.mu_trees = new_trees
        else:
            self.bcf_params.tau_trees = new_trees
            
    @property
    def n_trees(self):
        return len(self.trees)
            
    def copy(self, modified_tree_ids=None):
        new_bcf = None
        if(self.ensemble_id == 'mu'):
            new_bcf = self.bcf_params.copy(modified_mu_ids=modified_tree_ids, modified_tau_ids=[])
        else:
            new_bcf = self.bcf_params.copy(modified_tau_ids=modified_tree_ids, modified_mu_ids=[])
            
        copied = BCFParamView(new_bcf, self.ensemble_id, cache=copy.deepcopy(self.cache))
        if(self.ensemble_id == 'mu'):
            new_bcf.mu_view = copied
        else:
            new_bcf.tau_view = copied

        return copied
    