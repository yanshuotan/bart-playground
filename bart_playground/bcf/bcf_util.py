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
        if z is None or len(z) == 0:
            raise ValueError("Treatment indicator z must be provided and non-empty.")
        # If z is an integer array of shape (n, ), convert it to a 2D array using one-hot encoding
        if z.ndim == 1:
            z = (z == np.arange(1, np.max(z)+1))
        # Require positivity: any X[z[:, i]] must contain at least one valid sample
        for i in range(z.shape[1]):
            if np.sum(z[:, i]) == 0:
                raise ValueError(f"Treatment arm {i+1} has no samples.")
        # and the control group (i.e. not in any treatment group) must be present as well
        if np.sum(z, axis=1).min() == 1:
                raise ValueError("Control group must be present.")
        dataset = super().fit_transform(X, y)
        return BCFDataset(dataset.X, dataset.y, z)
    
    def update_transform(self, X_new, y_new, z_new, dataset):
        """
        Update an existing dataset with new data points, including treatment indicators.
        """
        # Call the parent method first to handle X and y
        base_dataset = super().update_transform(X_new, y_new, dataset)
        
        # Now handle the treatment indicator z
        if z_new.ndim == 1:
            z_new = z_new.reshape(-1, 1)
            
        z_combined = np.vstack([dataset.z, z_new])
        
        # Create and return a BCFDataset with the updated data
        return BCFDataset(base_dataset.X, base_dataset.y, z_combined)

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
