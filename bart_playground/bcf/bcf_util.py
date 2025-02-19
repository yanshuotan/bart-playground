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
        self._init_treated_data()
    
    def _init_treated_data(self):
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
    def data(self):
        raise Exception("Shouldn't be called now")
        return self.bart_data()
        # return self.bcf_params.data
    @data.setter
    def data(self, new_data):
        raise Exception("data.setter called, but not implemented")
        # self.bcf_params.data = new_data
        
    def bart_data(self):
        parent_data = self.bcf_params.data
        if(self.ensemble_id == "mu"):
            residuals = parent_data.y - parent_data.z * self.bcf_params.tau_view.evaluate()
            return Dataset(parent_data.X, residuals, parent_data.thresholds)
        else:
            residuals = parent_data.y - self.bcf_params.mu_view.evaluate()
            return Dataset(self.context_treated, residuals[parent_data.treated], self.thresholds_treated)
        
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
            new_bcf = self.bcf_params.copy(modified_mu_ids=modified_tree_ids)
        else:
            new_bcf = self.bcf_params.copy(modified_tau_ids=modified_tree_ids)
            
        sub_model = BCFParamView(new_bcf, self.ensemble_id, cache=copy.deepcopy(self.cache))
        return sub_model

    # def evaluate(self, X: np.ndarray=None, tree_ids=None, all_except=None) -> float:
    #     # TODO: Only update necessary parts of the ensemble 
    #     if X is None:
    #         X = self.data.X
    #     yhat = 0.
    #     for t in self.trees:
    #         yhat += t.predict(X)
    #     return yhat
    