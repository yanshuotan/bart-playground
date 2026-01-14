import numpy as np
from typing import Optional, Any, Dict, List

try:
    from bartz.BART import gbart as Bartz
    BARTZ_AVAILABLE = True
except ImportError:
    BARTZ_AVAILABLE = False
    Bartz = None

try:
    from stochtree import BARTModel
    STOCHTREE_AVAILABLE = True
except ImportError:
    STOCHTREE_AVAILABLE = False
    BARTModel = None

class BartzWrapper:
    """
    Wrapper for bartz library to be used with BARTTSAgent.
    """
    def __init__(self, ndpost: int = 1000, nskip: int = 100, **kwargs):
        if not BARTZ_AVAILABLE:
            raise ImportError("bartz library is not installed.")
        self.ndpost = int(ndpost)
        self.nskip = int(nskip)
        self.kwargs = kwargs
        self.model = None
        self._range_post = range(self.ndpost)

    def fit(self, X: np.ndarray, y: np.ndarray, quietly: bool = True):
        # bartz expects X to be (n_features, n_samples)
        # y should be float32
        X_T = X.T
        y_float = y.astype(np.float32)
        
        actual_kwargs = dict(self.kwargs)
        if "random_state" in actual_kwargs:
            actual_kwargs["seed"] = actual_kwargs.pop("random_state")

        # type='wbart' is used in the example
        self.model = Bartz(X_T, y_float, ndpost=self.ndpost, nskip=self.nskip, 
                           type='wbart', printevery=None if quietly else 100, **actual_kwargs)
        self.is_fitted = True
        self.trace = [] # Dummy for diagnostics_mixin
        self.sampler = type('DummySampler', (), {
            'move_selected_counts': {}, 
            'move_success_counts': {}, 
            'move_accepted_counts': {}
        })()
        return self

    @property
    def range_post(self):
        return self._range_post

    def predict_trace(self, k: int, X: np.ndarray, backtransform: bool = True) -> np.ndarray:
        """
        Predict using the k-th posterior sample.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        
        # bartz predict expects (n_features, n_samples)
        # predict returns (ndpost, n_samples)
        preds = self.model.predict(X.T)
        return preds[k]

    def posterior_f(self, X: np.ndarray, backtransform: bool = True) -> np.ndarray:
        """
        Return predictions for all posterior samples.
        Returns: (n_samples, ndpost)
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        
        preds = self.model.predict(X.T) # (ndpost, n_samples)
        return preds.T # (n_samples, ndpost)

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "BartzWrapper",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            **self.kwargs
        }

    def feature_inclusion_frequency(self, normalize: str = "split"):
        raise NotImplementedError("feature_inclusion_frequency not supported for BartzWrapper")


class StochTreeWrapper:
    """
    Wrapper for stochtree library to be used with BARTTSAgent.
    Supports toggling GFR (Gradient Forest Restoration/Regression).
    """
    def __init__(self, ndpost: int = 500, nskip: int = 500, use_gfr: bool = True, **kwargs):
        if not STOCHTREE_AVAILABLE:
            raise ImportError("stochtree library is not installed.")
        self.ndpost = int(ndpost)
        self.nskip = int(nskip)
        self.use_gfr = use_gfr
        self.kwargs = kwargs
        self.model = None
        self._constant_y: Optional[float] = None
        self._range_post = range(self.ndpost)

    def fit(self, X: np.ndarray, y: np.ndarray, quietly: bool = True):
        # Short-circuit constant outcomes (stochtree can fail / waste work here)
        if np.unique(y).size == 1:
            self._constant_y = float(np.ravel(y)[0])
            self.model = None
            self.is_fitted = True
            self.trace = [] # Dummy for diagnostics_mixin
            self.sampler = type('DummySampler', (), {
                'move_selected_counts': {}, 
                'move_success_counts': {}, 
                'move_accepted_counts': {}
            })()
            return self
        
        self._constant_y = None
        self.model = BARTModel()
        
        # Mapping: ndpost -> num_mcmc, nskip -> num_burnin
        sample_kwargs = {
            "num_mcmc": self.ndpost,
            "num_burnin": self.nskip
        }
        
        # If GFR is disabled, set num_gfr=0 (based on external_example.py)
        if not self.use_gfr:
            sample_kwargs["num_gfr"] = 0
            
        # Standardize parameter names if they come from _prepare_bart_kwargs
        actual_kwargs = dict(self.kwargs)
        if "n_trees" in actual_kwargs:
            n_trees = int(actual_kwargs.pop("n_trees"))
            mean_forest_params = dict(actual_kwargs.get("mean_forest_params") or {})
            mean_forest_params["num_trees"] = n_trees
            actual_kwargs["mean_forest_params"] = mean_forest_params
            
        if "random_state" in actual_kwargs:
            val = actual_kwargs.pop("random_state")
            # Ensure seed is within 32-bit signed int range (Pybind11 int)
            safe_seed = int(val % 2147483647)
            if "general_params" not in actual_kwargs:
                actual_kwargs["general_params"] = {}
            actual_kwargs["general_params"]["random_seed"] = safe_seed
            
        sample_kwargs.update(actual_kwargs)
        
        # stochtree.sample(X, y, ...)
        # Assuming X is (n_samples, n_features) which is standard
        self.model.sample(X, y, **sample_kwargs)
        self.is_fitted = True
        self.trace = [] # Dummy for diagnostics_mixin
        self.sampler = type('DummySampler', (), {
            'move_selected_counts': {}, 
            'move_success_counts': {}, 
            'move_accepted_counts': {}
        })()
        return self

    @property
    def range_post(self):
        return self._range_post

    def predict_trace(self, k: int, X: np.ndarray, backtransform: bool = True) -> np.ndarray:
        if self._constant_y is not None:
            return np.full((X.shape[0],), self._constant_y, dtype=float)
        
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        
        # Assuming stochtree has a predict method for new data
        if hasattr(self.model, "predict"):
            # predict likely returns (n_samples, ndpost) or (ndpost, n_samples)
            # Some versions might return a dict.
            preds = self.model.predict(X)
            if isinstance(preds, dict):
                preds = preds.get('y_hat', preds.get('mean', None))
            
            preds = np.asarray(preds)
            
            # Check shape logic: if we pass 1 sample, we expect (1, ndpost)
            if preds.shape[0] == X.shape[0]:
                 return preds[:, k]
            elif preds.shape[1] == X.shape[0]:
                 return preds[k, :]
            else:
                 # Fallback: assume (n_samples, ndpost)
                 return preds[:, k]
        else:
             # Fallback if stochtree only supports test set during sampling
             # This would be problematic for Bandit usage unless we retrain or stochtree provides another way
             raise NotImplementedError("stochtree.BARTModel.predict method required for Bandit usage.")

    def posterior_f(self, X: np.ndarray, backtransform: bool = True) -> np.ndarray:
        if self._constant_y is not None:
            return np.full((X.shape[0], self.ndpost), self._constant_y, dtype=float)
        
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
            
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X)
            if isinstance(preds, dict):
                preds = preds.get('y_hat', preds.get('mean', None))
            
            preds = np.asarray(preds)

            # Ensure shape is (n_samples, ndpost)
            if preds.shape[0] != X.shape[0] and preds.shape[1] == X.shape[0]:
                preds = preds.T
            return preds
        else:
            raise NotImplementedError("stochtree.BARTModel.predict method required for Bandit usage.")

    def get_params(self) -> Dict[str, Any]:
        return {
            "model_type": "StochTreeWrapper",
            "ndpost": self.ndpost,
            "nskip": self.nskip,
            "use_gfr": self.use_gfr,
            **self.kwargs
        }

    def feature_inclusion_frequency(self, normalize: str = "split"):
        raise NotImplementedError("feature_inclusion_frequency not supported for StochTreeWrapper")
