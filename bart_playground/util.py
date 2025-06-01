from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# For faster random sampling
def fast_choice(generator, array):
    """Fast random selection from an array."""
    if len(array) == 0:
        raise ValueError("Cannot choose from empty array")
    elif len(array) == 1:
        return array[0]
    return array[generator.integers(0, len(array))]

class Dataset:

    def __init__(self, X, y):
        # if X is pd.DataFrame:
            # X = X.to_numpy()
        # if X is pd.DataFrame:
            # X = X.to_numpy()
        self.X = X
        self.y = y

    @property
    def n(self):
        return self.X.shape[0]
    @property
    def p(self):
        return self.X.shape[1]

class Preprocessor(ABC):
    @property
    def thresholds(self):
        return self._thresholds
    @thresholds.setter
    def thresholds(self, value):
        self._thresholds = value

    @abstractmethod
    def gen_thresholds(self, X):
        pass
    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X, y) -> Dataset:
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    @abstractmethod
    def transform_y(self, y) -> np.ndarray:
        pass

    @abstractmethod
    def backtransform_y(self, y) -> np.ndarray:
        pass

    @abstractmethod
    def update_transform(self, X_new, y_new, dataset):
        """
        Update an existing dataset with new data points.
        
        Parameters:
            X_new: New feature data to add
            y_new: New target data to add
            dataset: Existing dataset to update
            
        Returns:
            Updated dataset
        """
        pass

class DefaultPreprocessor(Preprocessor):
    """
    Default implementation for preprocessing input data.
    """
    def __init__(self, max_bins: int=100):
        """
        Initialize the default X preprocessor.

        Parameters:
        - max_bins: int
            Maximum number of bins.
        """
        self.max_bins = max_bins
        self.splits = None

    def fit(self, X, y):
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be None")
        self.y_max = y.max()
        self.y_min = y.min()
        self._thresholds = self.gen_thresholds(X)
        
    def transform(self, X, y):
        return Dataset(X, self.transform_y(y))
    
    def gen_thresholds(self, X):
        q_vals = np.linspace(0, 1, self.max_bins, endpoint=False)
        return dict({k : np.unique(np.quantile(X[:, k], q_vals)) for k in range(X.shape[1])})
    
    @staticmethod
    def test_thresholds(X):
        return dict({k : np.unique(X[:, k]) for k in range(X.shape[1])})

    def transform_y(self, y) -> np.ndarray:
        if self.y_max == self.y_min:
            return y
        else:
            return (y - self.y_min) / (self.y_max - self.y_min) - 0.5
    
    def backtransform_y(self, y) -> np.ndarray:
        return (self.y_max - self.y_min) * (y + 0.5) + self.y_min

    def update_transform(self, X_new, y_new, dataset):
        """
        Update an existing dataset with new data points.
        """
        X_combined = np.vstack([dataset.X, X_new])
        
        y_new_transformed = self.transform_y(y_new)
        y_combined = np.vstack([dataset.y.reshape(-1, 1), 
                              y_new_transformed.reshape(-1, 1)]).flatten()
        updated_dataset = Dataset(X_combined, y_combined)
        return updated_dataset
    
class BinaryPreprocessor(DefaultPreprocessor):
    """
    Preprocessor for binary classification tasks.
    """
    def transform_y(self, y) -> np.ndarray:
        # Convert binary labels to -0.5 and 0.5
        return y
    
    def backtransform_y(self, y) -> np.ndarray:
        return y
    
from scipy.stats import geninvgauss, gamma, invgamma

def rvs_gig(eta, chi, psi, size=1, random_state=None):
    """
    Sample random variates from GIG(eta, chi, psi), handling degenerate limits.
    Supports vectorized inputs for eta, chi, and psi.
    """
    eta = np.asarray(eta)
    chi = np.asarray(chi)
    psi = np.asarray(psi)
    eta, chi, psi = np.broadcast_arrays(eta, chi, psi)
    
    # Initialize result array
    if eta.shape == ():
        # Scalar case
        if chi == 0:
            return gamma.rvs(a=eta, scale=2.0/psi, size=size, random_state=random_state)
        elif psi == 0:
            return invgamma.rvs(a=-eta, scale=chi/2.0, size=size, random_state=random_state)
        else:
            b_scipy = np.sqrt(psi * chi)
            scale_scipy = np.sqrt(chi / psi)
            return geninvgauss.rvs(eta, b_scipy, scale=scale_scipy, size=size, random_state=random_state)
    
    # Vectorized case
    result = np.zeros(eta.shape if size == 1 else eta.shape + (size,))
    
    # Case 1: chi == 0 --> Gamma
    chi_zero_mask = (chi == 0)
    if np.any(chi_zero_mask):
        result[chi_zero_mask] = gamma.rvs(
            a=eta[chi_zero_mask], scale=2.0/psi[chi_zero_mask], 
            size=(np.sum(chi_zero_mask), size) if size > 1 else np.sum(chi_zero_mask), 
            random_state=random_state
        ).reshape(result[chi_zero_mask].shape)
    
    # Case 2: psi == 0 --> InvGamma
    psi_zero_mask = (psi == 0) & ~chi_zero_mask
    if np.any(psi_zero_mask):
        result[psi_zero_mask] = invgamma.rvs(
            a=-eta[psi_zero_mask], scale=chi[psi_zero_mask]/2.0,
            size=(np.sum(psi_zero_mask), size) if size > 1 else np.sum(psi_zero_mask),
            random_state=random_state
        ).reshape(result[psi_zero_mask].shape)
    
    # Case 3: General case
    general_mask = (chi > 0) & (psi > 0)
    if np.any(general_mask):
        eta_gen = eta[general_mask]
        chi_gen = chi[general_mask]
        psi_gen = psi[general_mask]
        b_scipy = np.sqrt(psi_gen * chi_gen)
        scale_scipy = np.sqrt(chi_gen / psi_gen)
        result[general_mask] = geninvgauss.rvs(
            eta_gen, b_scipy, scale=scale_scipy,
            size=(np.sum(general_mask), size) if size > 1 else np.sum(general_mask),
            random_state=random_state
        ).reshape(result[general_mask].shape)
    
    return result

def gig_normalizing_constant(eta, chi, psi):
    """
    Compute the normalizing constant for the GIG distribution.
    Supports vectorized inputs for eta, chi, and psi.
    """
    from scipy.special import gamma, kv
    
    # Convert inputs to numpy arrays for vectorized operations
    eta = np.asarray(eta)
    chi = np.asarray(chi)
    psi = np.asarray(psi)
    
    # Ensure all arrays have the same shape
    eta, chi, psi = np.broadcast_arrays(eta, chi, psi)
    
    # Initialize result array
    result = np.zeros_like(eta, dtype=float)
    
    # Case 1: chi == 0 --> Gamma limit, requires eta > 0
    chi_zero_mask = (chi == 0)
    if np.any(chi_zero_mask):
        eta_chi_zero = eta[chi_zero_mask]
        psi_chi_zero = psi[chi_zero_mask]
        
        # Check validity
        if np.any(eta_chi_zero <= 0):
            raise ValueError("For chi=0, GIG(eta,psi,0) is Gamma only if eta>0.")
        
        # Z(eta,psi,0) = (2/psi)^eta * Gamma(eta)
        result[chi_zero_mask] = (2.0 / psi_chi_zero)**eta_chi_zero * gamma(eta_chi_zero)
    
    # Case 2: psi == 0 --> Inverse-Gamma limit, requires eta < 0
    psi_zero_mask = (psi == 0) & ~chi_zero_mask  # Exclude chi==0 case
    if np.any(psi_zero_mask):
        eta_psi_zero = eta[psi_zero_mask]
        chi_psi_zero = chi[psi_zero_mask]
        
        # Check validity
        if np.any(eta_psi_zero >= 0):
            raise ValueError("For psi=0, GIG(eta,0,chi) is Inverse-Gamma only if eta<0.")
        
        # Z(eta,0,chi) = (chi/2)^eta * Gamma(-eta)
        result[psi_zero_mask] = (chi_psi_zero / 2.0)**eta_psi_zero * gamma(-eta_psi_zero)
    
    # Case 3: General case: psi > 0 and chi > 0
    general_mask = (psi > 0) & (chi > 0)
    if np.any(general_mask):
        eta_general = eta[general_mask]
        chi_general = chi[general_mask]
        psi_general = psi[general_mask]
        
        # Z(eta,psi,chi) = 2 * K_p(sqrt(psi*chi)) * (chi/psi)^(eta/2)
        z = np.sqrt(psi_general * chi_general)
        Kp = kv(eta_general, z)  # Modified Bessel function K_p(z)
        result[general_mask] = 2.0 * Kp * (chi_general / psi_general)**(0.5 * eta_general)
    
    # Return scalar if input was scalar
    if result.shape == ():
        return result.item()
    return result
