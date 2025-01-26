from abc import ABC, abstractmethod
from ..util import Dataset
import numpy as np
from .bcf_params import BCFParams
from ..params import Parameters

class BCFDataset(Dataset):
    def __init__(self, X, y, z, thresholds):
        self.z = z
        super().__init__(X, y, thresholds)
        
class BCFParamSlice(Parameters):
    """
    An adapter making a BCFParams ensemble look like a single-ensemble
    BART 'Parameters' object.

    'ensemble_id' is either 'mu' or 'tau'.
    """

    def __init__(self, bcf_params, ensemble_id):
        self.bcf_params = bcf_params
        self.ensemble_id = ensemble_id
        self.data = bcf_params.data  

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

    def copy(self, trees_changed=None):
        # TODO: Only copy necessary parts of the ensemble 
        new_bcf = self.bcf_params.copy()
        sub_model = BCFParamSlice(new_bcf, self.ensemble_id)
        return sub_model

    def evaluate(self, X: np.ndarray=None, tree_ids=None, all_except=None) -> float:
        # TODO: Only update necessary parts of the ensemble 
        if X is None:
            X = self.data.X
        yhat = 0.
        for t in self.trees:
            yhat += t.predict(X)
        return yhat
    