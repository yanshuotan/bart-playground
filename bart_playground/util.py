from abc import ABC, abstractmethod
from re import U

import numpy as np
from numba import njit, objmode, f8
from scipy.special import kv
import math

# For faster random sampling
def fast_choice(generator, array):
    """Fast random selection from an array."""
    len_arr = len(array)
    if len_arr == 1:
        return array[0]
    return array[generator.integers(0, len_arr)]

def fast_choice_with_weights(generator, array, weights, cum_weights=None):
    """Fast random selection from an array with given weights.
    
    If cum_weights is provided (precomputed cumsum of weights), it will be used
    directly to avoid redundant np.cumsum calls.
    """
    if weights is None:
        return fast_choice(generator, array)
    len_arr = len(array)
    if len_arr != len(weights):
        raise ValueError("Array and weights must have the same length")
    if len_arr == 1:
        return array[0]
    # Use precomputed cumsum if available (caller responsible for correctness)
    if cum_weights is None:
        cum_weights = np.cumsum(weights)
    weight_sum = cum_weights[-1]
    U = generator.uniform(0, weight_sum)
    idx = np.searchsorted(cum_weights, U)
    return array[idx]

class Dataset:

    def __init__(self, X, y):
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
    def gen_thresholds(self, X) -> dict:
        pass
        
    def fit(self, X, y):
        pass

    def transform(self, X, y) -> Dataset:
        return Dataset(
            self.transform_X(X),
            self.transform_y(y)
        )

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
    def transform_X(self, X) -> np.ndarray:
        return X

    def transform_y(self, y) -> np.ndarray:
        return y

    def backtransform_y(self, y) -> np.ndarray:
        return y

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
        X_combined = np.vstack([dataset.X, X_new])
        
        y_new_transformed = self.transform_y(y_new)
        y_combined = np.vstack([dataset.y.reshape(-1, 1), 
                              y_new_transformed.reshape(-1, 1)]).flatten()
        updated_dataset = Dataset(X_combined, y_combined)
        return updated_dataset

class DefaultPreprocessor(Preprocessor):
    """
    Default implementation for preprocessing input data for continuous BART.
    """
    def __init__(self, max_bins: int=100):
        """
        Initialize the default preprocessor.

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
    
    def gen_thresholds(self, X):
        q_vals = np.linspace(0, 1, self.max_bins, endpoint=False)
        return dict({k : np.unique(np.quantile(X[:, k], q_vals)) for k in range(X.shape[1])})
    
    @staticmethod
    def test_thresholds(X):
        return dict({k : np.unique(X[:, k]) for k in range(X.shape[1])})
    
    def transform_X(self, X) -> np.ndarray:
        return X.astype(np.float32)
    
    def transform_y(self, y) -> np.ndarray:
        if self.y_max == self.y_min:
            y_res = y # do not transform if all values are the same
        else:
            y_res = (y - self.y_min) / (self.y_max - self.y_min) - 0.5
        return y_res.reshape(-1, ).astype(np.float32)
    
    def backtransform_y(self, y) -> np.ndarray:
        if self.y_max == self.y_min: # y not transformed
            return y
        else:
            return (self.y_max - self.y_min) * (y + 0.5) + self.y_min
    
class ClassificationPreprocessor(Preprocessor):
    """
    Preprocessor for classification tasks.
    """
    def __init__(self, max_bins: int=100):
        """
        Initialize the classification preprocessor.

        Parameters:
        - max_bins: int
            Maximum number of bins.
        """
        self.max_bins = max_bins
        self.uniq_labels = None
        
    @property
    def labels(self):
        if self.uniq_labels is None:
            raise ValueError("Preprocessor must be fitted before accessing ClassificationPreprocessor.labels")
        return self.uniq_labels
    
    def fit(self, X, y):
        if X is None or y is None or len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be None")
        self.uniq_labels, y_encoded = np.unique(y, return_inverse=True)
        self._thresholds = self.gen_thresholds(X)
        
    def transform_y(self, y):
        if y is None or len(y) == 0:
            raise ValueError("y cannot be None or empty")
        if self.uniq_labels is None:
            raise ValueError("Preprocessor must be fitted before transforming data")
        label_to_index = {label: idx for idx, label in enumerate(self.uniq_labels)}
        y_encoded = np.array([label_to_index[val] for val in y], dtype=int)
        return y_encoded
    
    def backtransform_y(self, y):
        if y is None or len(y) == 0:
            raise ValueError("y cannot be None or empty")
        if self.uniq_labels is None:
            raise ValueError("Preprocessor must be fitted before backtransforming data")
        y_decoded = self.uniq_labels[y]
        return y_decoded

    def gen_thresholds(self, X):
        """
        Use the same quantile-based binning as DefaultPreprocessor
        """
        q_vals = np.linspace(0, 1, self.max_bins, endpoint=False)
        return dict({k : np.unique(np.quantile(X[:, k], q_vals)) for k in range(X.shape[1])})
    
@njit(cache=True)
def log_K_asymptotic(v, x):
    """
    One-term asymptotic: log K_v(x) ~ 
      -v * ln(e*x/(2*v)) - 0.5*ln(v) + 0.5*ln(pi/2) + O(1/v).
    Reference: DLMF 10.40.8.
    """
    # Compute -v * ln(e*x/(2v)) = -v * (1 + ln(x/(2v)))
    term1 = -v * (1.0 + math.log(x / (2.0 * v)))
    # Prefactor: -0.5*ln(v) + 0.5*ln(pi/2)
    term2 = -0.5 * math.log(v) + 0.5 * math.log(math.pi / 2.0)
    return term1 + term2

class GIG:
    """
    Generalized Inverse Gaussian (GIG) distribution.
    """
    @staticmethod
    def rvs_gig_scalar(eta, chi, psi, generator=np.random.default_rng()):
        """
        Sample random variates from GIG(eta, chi, psi) for scalar eta, chi, psi.
        Handles degenerate Gamma (chi=0) and InvGamma (psi=0) cases.
        """
        # Case 1: chi == 0 -> Gamma(eta, scale=2/psi)
        if chi == 0:
            if eta <= 0:
                raise ValueError("For chi=0, eta must be > 0.")
            return generator.gamma(eta, scale=2.0/psi)

        # Case 2: psi == 0 -> InvGamma(-eta, scale=chi/2)
        if psi == 0:
            if eta >= 0:
                raise ValueError("For psi=0, eta must be < 0.")
            return 1.0 / generator.gamma(-eta, scale=2.0/chi)
            # return invgamma.rvs(a=-eta, scale=chi/2.0, random_state=random_state)

        # Case 3: general GIG
        return GIG._gig_devroye(eta, chi, psi, generator=generator)
        # b = math.sqrt(psi * chi)
        # scale_param = math.sqrt(chi / psi)
        # return geninvgauss.rvs(eta, b, scale=scale_param, size=size, random_state=random_state)
        
    @staticmethod
    def _gig_devroye(eta, chi, psi, generator=np.random.default_rng()):
        """
        Devroye (2014) logistic-transform sampler for GIG(eta, chi, psi).
        Parametrized as GIG(eta, chi, psi) where:
        p \\propto x^{eta-1} exp(-1/2 * (chi/x + psi*x)) for x > 0.
        Equivalent to geninvgauss(eta, b, scale) where:
        b = sqrt(chi * psi) and scale = sqrt(chi / psi).
        """
        omega = math.sqrt(chi * psi)
        swap  = eta < 0.0
        eta   = abs(eta)

        # Devroye alpha
        alpha = math.sqrt(omega**2 + eta**2) - eta

        # Functions needed
        def _psi(x):
            return -alpha*(math.cosh(x) - 1.0) - eta*(math.exp(x) - x - 1.0)
        def _psi_prime(x):
            return -alpha*math.sinh(x) - eta*(math.exp(x) - 1.0)

        # Set t
        t = 0.0
        x = -_psi(1.0)
        if 0.5 <= x <= 2.0:
            t = 1.0
        elif x > 2.0:
            t = 1.0 if (alpha == 0 and eta == 0) else math.sqrt(2.0/(alpha+eta))
        else:  # x < 0.5
            t = 1.0 if (alpha == 0 and eta == 0) else math.log(4.0/(alpha+2.0*eta))

        # Set s
        s = 0.0
        x = -_psi(-1.0)
        if 0.5 <= x <= 2.0:
            s = 1.0
        elif x > 2.0:
            s = 1.0 if (alpha == 0 and eta == 0) else math.sqrt(4.0/(alpha*math.cosh(1.0)+eta))
        else:  # x < 0.5
            if alpha == 0 and eta == 0:
                s = 1.0
            elif alpha == 0:
                s = 1.0 / eta
            elif eta == 0:
                s = math.log(1.0 + 1.0/alpha + math.sqrt(1.0/alpha**2 + 2.0/alpha))
            else:
                s = min(1.0/eta,
                        math.log(1.0 + 1.0/alpha + math.sqrt(1.0/alpha**2 + 2.0/alpha)))

        # Compute the parameters for the rejection sampling
        param_eta   = -_psi(t)
        param_zeta  = -_psi_prime(t)
        param_theta = -_psi(-s)
        param_xi    =  _psi_prime(-s)

        p  = 1.0 / param_xi
        r  = 1.0 / param_zeta
        t_prime = t - r * param_eta
        s_prime = s - p * param_theta
        q  = t_prime + s_prime

        # Generation
        while True:
            U = generator.random()
            V = generator.random()
            W = generator.random()

            if U < q / (p + q + r):
                X = -s_prime + q * V
            elif U < (q + r) / (p + q + r):
                X = t_prime - r * math.log(V)
            else:                           
                X = -s_prime + p * math.log(V)

            f1 = math.exp(-param_eta   - param_zeta * (X - t))
            f2 = math.exp(-param_theta + param_xi   * (X + s))
            chi_X = 1.0 if -s_prime <= X <= t_prime else (f1 if X > t_prime else f2) # This chi(X) is the chi function in the Devroye paper, not the chi parameter

            if W * chi_X <= math.exp(_psi(X)):
                break

        # Result computation
        x = math.exp(X) * (eta/omega + math.sqrt(1.0 + (eta/omega)**2))
        if swap:
            x = 1.0 / x

        # Transform to GIG(eta, chi, psi)
        return x * math.sqrt(chi / psi)

    @staticmethod
    @njit(cache=True)
    def log_gig_normalizing_constant_numba(eta: float, chi: float, psi: float) -> float:
        """
        Compute the GIG normalizing constant for scalar inputs (eta, chi, psi).
        Numba-accelerated version where only the Bessel K call uses object mode.
        """
        # Case 1: chi == 0 --> Gamma limit (eta > 0)
        if chi == 0:
            if eta <= 0:
                raise ValueError("For chi=0, GIG(eta, psi, 0) requires eta > 0.")
            # Z = (2/psi)^eta * Gamma(eta)
            log_val = math.log(2.0 / psi) * eta + math.lgamma(eta)
            return log_val

        # Case 2: psi == 0 --> Inverse-Gamma limit (eta < 0)
        if psi == 0:
            if eta >= 0:
                raise ValueError("For psi=0, GIG(eta, 0, chi) requires eta < 0.")
            # Z = (chi/2)^eta * Gamma(-eta)
            log_val = math.log(chi / 2.0) * eta + math.lgamma(-eta)
            return log_val

        # Case 3: General case chi > 0 and psi > 0
        # Z = 2 * K_eta(sqrt(psi*chi)) * (chi/psi)^(eta/2)
        z = math.sqrt(psi * chi)
        # Temporarily switch to object mode to call scipy.special.kv
        with objmode(K_val=f8):
            K_val = kv(eta, z)
            
        if math.isinf(K_val):
            log_K_val = log_K_asymptotic(eta, z)
        else:
            log_K_val = math.log(K_val)

        return math.log(2.0) + log_K_val + math.log(chi / psi) * (0.5 * eta)
