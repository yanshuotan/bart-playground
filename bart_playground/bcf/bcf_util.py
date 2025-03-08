from calendar import c
from enum import Enum
from dataclasses import dataclass
from tkinter.tix import Tree
from typing import Optional, Union
import numpy as np
from ..util import Dataset, DefaultPreprocessor
from numpy.typing import NDArray

class BCFDataset(Dataset):
    def __init__(self, X, y, Z, has_propensity : bool = False):
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        self.Z = Z
        for i in range(Z.shape[0]):
            assert np.sum(Z[i, :]) <= 1 # At most one treatment arm shall be True
        self._propensity = has_propensity
        super().__init__(X, y)
        
    @property
    def treated(self):
        assert self.Z.shape[1] == 1, "Only one treatment arm is supported when using property treated."
        return self.treated_by(0)
    
    def treated_by(self, arm = 0):
        return self.Z[:, arm] == 1
    
    @property  
    def has_propensity(self):
        return self._propensity
    
    @property
    def covariates(self):
        if self.has_propensity:
            return self.X[:, :(-self.Z.shape[1])]
        else:
            return self.X
    
class BCFPreprocessor(DefaultPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ps_models = None  # Changed from ps_model to ps_models (will be a list)

    def transform_z(self, Z, check_positivity = True):
        """
        Transform the treatment indicator Z into a one-hot encoded array.
        """
        # If Z is an integer array of shape (n, ), convert it to a 2D array using one-hot encoding
        if Z.ndim == 1:
            Z = (Z == np.arange(1, np.max(Z)+1))
        # Ensure Z is a 2D array
        if Z.ndim != 2:
            raise ValueError("Treatment indicator Z must be a 2D array.")
        if check_positivity:
            # Require positivity: any X[Z[:, i]] must contain at least one valid sample
            for i in range(Z.shape[1]):
                if np.sum(Z[:, i]) == 0:
                    raise ValueError(f"Treatment arm {i+1} has no samples.")
            # and the control group (i.e. not in any treatment group) must be present as well
            if np.sum(Z, axis=1).min() == 1:
                    raise ValueError("Control group must be present.")
        return Z

    def fit_transform(self, X, y, Z, ps : Union[bool, NDArray] = True):
        """
        Parameters
        ----------
        X: Feature matrix.
        y: Target variable.
        Z: Treatment indicator. Shape (n, k) where n is the number of samples and k is the number of treatment arms (excluding control).
          If an integer array, it will be converted to a one-hot encoded array.
        ps (Union[bool, NDArray]): If False, do not fit a propensity score model. If True, fit a propensity score model. If an array of shape (n, k),
            use it as the propensity score. The propensity score is a probability of being assigned to a treatment group given the covariates.
        """
        if Z is None or len(Z) == 0:
            raise ValueError("Treatment indicator Z must be provided and non-empty.")
        Z = self.transform_z(Z)
        
        if ps is not False:
            if isinstance(ps, bool):
                # Fit separate propensity score models for each treatment arm
                from sklearn.linear_model import LogisticRegression
                self.ps_models = []
                ps_values = np.zeros((X.shape[0], Z.shape[1]))
                
                for i in range(Z.shape[1]):
                    model = LogisticRegression()
                    model.fit(X, Z[:, i])
                    ps_values[:, i] = model.predict_proba(X)[:, 1]  # Take probability of positive class
                    self.ps_models.append(model)
                
                X = np.hstack([X, ps_values])
            else:
                # Use provided propensity scores
                if ps.ndim == 1:
                    ps = ps.reshape(-1, 1)
                if ps.shape[0] != X.shape[0]:
                    raise ValueError("Shape of ps does not match shape of X.")
                if ps.shape[1] != Z.shape[1]:
                    raise ValueError(f"Shape of ps ({ps.shape[1]}) does not match shape of Z ({Z.shape[1]}).")
                X = np.hstack([X, ps])

        dataset = super().fit_transform(X, y)
        return BCFDataset(dataset.X, dataset.y, Z, ps is not False)
    
    def update_transform(self, X_new, y_new, z_new, dataset, ps: Union[bool, NDArray] =True):
        """
        Update an existing dataset with new data points, including treatment indicators.
        
        Parameters:
        -----------
        X_new: New feature data
        y_new: New target data
        z_new: New treatment indicator data
        dataset: Existing dataset to update
        ps (Union[bool, NDArray]): If False, do not fit a propensity score model. If True, fit a propensity score model. If an array of shape (n, k),
            use it as the propensity score. The propensity score is a probability of being assigned to a treatment group given the covariates.
        """
        # Transform z_new to proper format
        # No need to check positivity here
        z_new = self.transform_z(z_new, check_positivity=False)
        
        if dataset.has_propensity:
            if isinstance(ps, bool):
                if ps:
                    # If ps is enabled but not provided, we need to generate it using saved models
                    if hasattr(self, 'ps_models') and self.ps_models is not None:
                        ps_values = np.zeros((X_new.shape[0], z_new.shape[1]))
                        for i, model in enumerate(self.ps_models):
                            ps_values[:, i] = model.predict_proba(X_new)[:, 1]
                        ps = ps_values
                    else:
                        raise ValueError("Propensity scores enabled but no models available.")
                else:
                    raise ValueError("Cannot disable propensity scores in update when the model was fit with them.")
            
            # some checks
            if not isinstance(ps, np.ndarray):
                raise ValueError("Propensity scores must be either True, False, or a numpy array.")
            if ps.ndim == 1:
                ps = ps.reshape(-1, 1)
            if ps.shape[0] != X_new.shape[0]:
                raise ValueError("Shape of ps does not match shape of X_new.")
            if ps.shape[1] != z_new.shape[1]:
                raise ValueError(f"Shape of ps ({ps.shape[1]}) does not match shape of z_new ({z_new.shape[1]}).")
            
            X_new = np.hstack([X_new, ps])

        elif ps is not False:
            # If dataset doesn't use propensity scores, we have to ignore ps
            import warnings
            if not isinstance(ps, bool):
                warnings.warn("Propensity scores provided in updating but ignored because the model was not fit with them.")
            else:
                warnings.warn("Propensity scores enabled in updating but ignored because the model was not fit with them.")
        
        base_dataset = super().update_transform(X_new, y_new, dataset)
        z_combined = np.vstack([dataset.Z, z_new])
        return BCFDataset(base_dataset.X, base_dataset.y, z_combined, dataset.has_propensity)
    
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
