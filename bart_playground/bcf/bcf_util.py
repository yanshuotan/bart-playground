
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np
from ..util import Dataset, DefaultPreprocessor

class BCFDataset(Dataset):
    def __init__(self, X, y, z):
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        self.z = z
        for i in range(z.shape[0]):
            assert np.sum(z[i, :]) <= 1 # At most one treatment arm shall be True
        super().__init__(X, y)
        
    @property
    def treated(self):
        assert self.z.shape[1] == 1, "Only one treatment arm is supported when using property treated."
        return self.treated_by(0)
    
    def treated_by(self, arm = 0):
        return self.z[:, arm] == 1
    
class BCFPreprocessor(DefaultPreprocessor):
    def fit_transform(self, X, y, z):
        dataset = super().fit_transform(X, y)
        return BCFDataset(dataset.X, dataset.y, z)
class EnsembleName(Enum):
    MU = "mu"
    TAU = "tau"
    
    def toggle(self):
        return next(member for member in type(self) if member != self)

@dataclass
class BCFEnsembleIndex:
    ensemble_name: EnsembleName
    _index: int = -1

    def __post_init__(self):
        # If ensemble_id is TAU, index must be provided.
        if self.ensemble_name == EnsembleName.TAU and self._index == -1:
            raise ValueError("Index must be provided when ensemble_id is 'tau'.")
        
    @property
    def index(self):
        if self.ensemble_name == EnsembleName.MU:
            return -1
        return self._index
    
BCFTreeIndices = tuple[BCFEnsembleIndex, list[int]]
        