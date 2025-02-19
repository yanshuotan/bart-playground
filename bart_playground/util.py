from abc import ABC, abstractmethod

import numpy as np

class Dataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n, self.p = X.shape

class Preprocessor(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X, y):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

    @abstractmethod
    def transform_y(self, y):
        pass

    @abstractmethod
    def backtransform_y(self, y):
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
        self.y_max = y.max()
        self.y_min = y.min()
        self._thresholds = self.gen_thresholds(X)
        
    def transform(self, X, y):
        return Dataset(X, self.transform_y(y))
    
    def gen_thresholds(self, X):
        q_vals = np.linspace(0, 1, self.max_bins, endpoint=False)
        return dict({k : np.unique(np.quantile(X[:, k], q_vals)) for k in range(X.shape[1])})
    
    @property
    def thresholds(self):
        return self._thresholds
    @thresholds.setter
    def thresholds(self, value):
        self._thresholds = value

    def transform_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min) - 0.5
    
    def backtransform_y(self, y):
        return (self.y_max - self.y_min) * (y + 0.5) + self.y_min