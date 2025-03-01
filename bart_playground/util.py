from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pandas as pd

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
    