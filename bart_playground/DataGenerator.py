import os
import pickle
import logging

import numpy as np
from sklearn.tree import DecisionTreeRegressor

LOGGER = logging.getLogger("DataGenerator")
logging.basicConfig(level=logging.INFO)


class DataGenerator:
    """
    Unified data generator for testing decision trees and similar models.

    Provides configurable scenarios that simulate different data challenges.
    """

    def __init__(self, n_samples=100, n_features=1, noise=0.1, random_seed=None):
        """
        Initialize the generator with default parameters.

        Args:
            n_samples (int): Number of data points.
            n_features (int): Number of features.
            noise (float): Standard deviation of noise.
            random_seed (int): Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def _add_noise(self, y):
        """Add Gaussian noise to target values."""
        return y + self.rng.normal(0, self.noise, size=len(y))

    def generate(self, scenario:str="linear", **kwargs) -> tuple:
        """
        Generate data for a specific scenario.

        Args:
            scenario (str): Name of the scenario.

        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target array.
        """
        func = getattr(self, scenario, None)
        if callable(func):
            result = func(**kwargs)
            if not isinstance(result, tuple):
                raise TypeError(f"Expected tuple return from {scenario}, got {type(result)}")
            return result
        else:
            raise NotImplementedError(f"No such a scenario supported: {scenario}")

    def linear(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y = X @ weights
        y = self._add_noise(y)
        return X, y

    def piecewise_flat(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        piecewise_conditions = [
            (X @ np.ones(self.n_features) < 0.3, lambda x: -0.5),
            ((X @ np.ones(self.n_features) >= 0.3) & (X @ np.ones(self.n_features) < 0.7), lambda x: 0),
            (X @ np.ones(self.n_features) >= 0.7, lambda x: 0.5)
        ]
        y = np.zeros(self.n_samples)
        for condition, func in piecewise_conditions:
            y[condition] = func((X[condition] @ np.ones(self.n_features)))
        y = self._add_noise(y)
        return X, y

    def heteroscedastic(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y = X @ weights
        noise = np.linalg.norm(X, axis=1) ** 2
        y += self.rng.normal(0, noise, size=len(y))
        return X, y

    def nonoverlapping_vs_overlapping(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        boundary = 0.5 * np.ones(self.n_features)
        distances = np.linalg.norm(X - boundary, axis=1)
        y_nonoverlapping = np.where(distances < 0.3, 2, 8) + self.rng.normal(0, self.noise, self.n_samples)
        y_overlapping = np.where(distances < 0.3, 2, 8) + self.rng.normal(0, 3 * self.noise, self.n_samples)
        return {"nonoverlapping": (X, y_nonoverlapping), "overlapping": (X, y_overlapping)}

    def spiky_vs_smooth(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        y_spiky = np.zeros(self.n_samples)
        y_spiky[self.rng.choice(self.n_samples, size=self.n_samples // 10, replace=False)] = 10
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y_smooth = np.exp(-(X @ weights - 2) ** 2 / 0.05)
        return {"spiky": (X, y_spiky), "smooth": (X, y_smooth)}

    def cyclic(self):
        X = self.rng.uniform(0, 10, (self.n_samples, self.n_features))
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y = np.sin(X @ weights)
        y = self._add_noise(y)
        return X, y

    def distribution_shift(self):
        X_train = self.rng.uniform(0, 0.5, (self.n_samples // 2, self.n_features))
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y_train = X_train @ weights + self._add_noise(X_train @ weights)
        X_test = self.rng.uniform(0.5, 1.0, (self.n_samples // 2, self.n_features))
        y_test = X_test @ weights + self._add_noise(X_test @ weights)
        return {"train": (X_train, y_train), "test": (X_test, y_test)}

    def multimodal(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        decision_boundary = 0.5 * np.ones(self.n_features)
        distances = np.linalg.norm(X - decision_boundary, axis=1)
        y = np.where(distances < 0.3,
                     self.rng.normal(2, 0.1, self.n_samples),
                     self.rng.normal(8, 0.1, self.n_samples))
        return X, y

    def imbalanced(self):
        X_majority = self.rng.uniform(0, 1, (int(self.n_samples * 0.9), self.n_features))
        X_minority = self.rng.uniform(2, 3, (int(self.n_samples * 0.1), self.n_features))
        X = np.vstack([X_majority, X_minority])
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y_majority = X_majority @ weights + self.rng.normal(0, self.noise, len(X_majority))
        y_minority = X_minority @ weights + self.rng.normal(0, self.noise, len(X_minority))
        y = np.concatenate([y_majority, y_minority])
        return X, y

    def piecewise_linear(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        piecewise_conditions = [
            (X @ np.ones(self.n_features) < 0.3, lambda x: 2 * x),
            ((X @ np.ones(self.n_features) >= 0.3) & (X @ np.ones(self.n_features) < 0.7), lambda x: 3 * x + 1),
            (X @ np.ones(self.n_features) >= 0.7, lambda x: -x + 3)
        ]
        y = np.zeros(self.n_samples)
        for condition, func in piecewise_conditions:
            y[condition] = func((X[condition] @ np.ones(self.n_features)))
        y = self._add_noise(y)
        return X, y

    def tied_x(self, tie_percentage=0.5):
        X = np.zeros((self.n_samples, self.n_features))
        for i in range(self.n_features):
            n_tied = int(self.n_samples * tie_percentage)
            unique_values = self.rng.uniform(0, 1, n_tied)
            X[:, i] = self.rng.choice(unique_values, self.n_samples)
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y = X @ weights
        y = self._add_noise(y)
        return X, y

    def tied_y(self):
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        weights = self.rng.uniform(1, 5, size=self.n_features)
        y = X @ weights
        y = np.round(y, decimals=1)
        y = self._add_noise(y)
        return X, y

    # --- New Methods for Friedman data ---

    def friedman1(self):
        """
        Friedman #1:
        y = 10*sin(pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4] + noise
        Requires at least 5 features.
        """
        if self.n_features < 5:
            raise ValueError("Friedman1 requires at least 5 features.")
        X = self.rng.uniform(0, 1, (self.n_samples, self.n_features))
        y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
             20 * (X[:, 2] - 0.5) ** 2 +
             10 * X[:, 3] +
             5 * X[:, 4])
        y = self._add_noise(y)
        return X, y

    def friedman2(self):
        """
        Friedman #2:
        Generate the Friedman #2 regression problem.

        Inputs X are 4 independent features uniformly distributed on the intervals:
          - X[:, 0] ~ Uniform(0, 100)
          - X[:, 1] ~ Uniform(40*pi, 560*pi)
          - X[:, 2] ~ Uniform(0, 1)
          - X[:, 3] ~ Uniform(1, 11)

        The output is computed as:
          y = sqrt( X[:,0]^2 + (X[:,1]*X[:,2] - 1/(X[:,1]*X[:,3]) )^2 ) + noise * N(0,1)

        Requires at least 4 features.
        """
        if self.n_features < 4:
            raise ValueError("Friedman2 requires at least 4 features.")
        X = np.empty((self.n_samples, self.n_features))
        X[:, 0] = self.rng.uniform(0, 100, self.n_samples)
        X[:, 1] = self.rng.uniform(40 * np.pi, 560 * np.pi, self.n_samples)
        X[:, 2] = self.rng.uniform(0, 1, self.n_samples)
        X[:, 3] = self.rng.uniform(1, 11, self.n_samples)
        if self.n_features > 4:
            X[:, 4:] = self.rng.uniform(0, 1, (self.n_samples, self.n_features - 4))
        frac = X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])
        y = np.sqrt(X[:, 0] ** 2 + frac ** 2)
        y = self._add_noise(y)
        return X, y

    def friedman3(self):
        """
        Friedman #3:
        Generate data as in sklearn.datasets.make_friedman3.

        Inputs X are 4 independent features uniformly distributed on the intervals:
          - X[:, 0] ~ Uniform(0, 100)
          - X[:, 1] ~ Uniform(40*pi, 560*pi)
          - X[:, 2] ~ Uniform(0, 1)
          - X[:, 3] ~ Uniform(1, 11)

        y = arctan((X[:,1]*X[:,2] - 1/(X[:,1]*X[:,3])) / X[:,0]) + noise * N(0,1)

        Requires at least 4 features.
        """
        if self.n_features < 4:
            raise ValueError("Friedman3 requires at least 4 features.")
        X = np.empty((self.n_samples, self.n_features))
        X[:, 0] = self.rng.uniform(0, 100, self.n_samples)
        X[:, 1] = self.rng.uniform(40 * np.pi, 560 * np.pi, self.n_samples)
        X[:, 2] = self.rng.uniform(0, 1, self.n_samples)
        X[:, 3] = self.rng.uniform(1, 11, self.n_samples)
        if self.n_features > 4:
            X[:, 4:] = self.rng.uniform(0, 1, (self.n_samples, self.n_features - 4))
        frac = (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]
        y = np.arctan(frac)
        y = self._add_noise(y)
        return X, y
    
    def linear_additive(self):
        X = self.rng.normal(0, 1, (self.n_samples, self.n_features))
        weights = np.array([0.2, -1, 0.6, -0.9, 0.85, 0, 0, 0, 0, 0])
        y = X @ weights
        y = self._add_noise(y)
        return X, y
    
    def smooth_additive(self):
        X = self.rng.normal(0, 1, (self.n_samples, self.n_features))
        X[:, 0] = X[:, 0] ** 2
        X[:, 2] = np.cos(X[:, 2])
        X[:, 3] = np.sqrt(np.abs(X[:, 3]))
        X[:, 4] = np.sin(X[:, 4])
        weights = np.array([0.2, -1, 0.6, -0.9, 0.85, 0, 0, 0, 0, 0])
        y = X @ weights
        y = self._add_noise(y)
        return X, y
    
    def dgp_1(self):
        """
        Generates data according to the DGP 1 decision tree in your image.
        x1, x2, x3 correspond to X[:, 0], X[:, 1], X[:, 2], respectively.
        Leaves: 6, 3, 9, 12
        """
        X = self.rng.normal(0, 1, (self.n_samples, self.n_features))
        y = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            x1, x2, x3 = X[i, 0], X[i, 1], X[i, 2]
            
            if x1 <= 0.1:
                if x3 <= -0.2:
                    y[i] = 6
                else:
                    y[i] = 3
            else:  # x1 > 0.1
                if x2 <= 0.7:
                    y[i] = 9
                else:
                    y[i] = 12
        
        # Optionally add noise
        y = self._add_noise(y)
        return X, y

    def dgp_2(self):
        """
        Generates data according to the DGP 2 decision tree in your image.
        x1, x2, x3, x4, x5, x6, x7 correspond to X[:, 0..6].
        Leaves: 1, 2, 4, 3, 4, 3, 9, 12
        """
        if not self.n_features == 7:
            LOGGER.debug("n_features is not 7, using 7 for dgp_2")
            self.n_features = 7
        X = self.rng.normal(0, 1, (self.n_samples, self.n_features))
        y = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            x1 = X[i, 0]
            x2 = X[i, 1]
            x3 = X[i, 2]
            x4 = X[i, 3]
            x5 = X[i, 4]
            x6 = X[i, 5]
            x7 = X[i, 6]

            if x1 <= 0.1:
                if x3 <= -0.2:
                    if x7 <= 0.15:
                        y[i] = 1
                    else:
                        y[i] = 2
                else:
                    if x6 <= 0.2:
                        y[i] = 4
                    else:
                        y[i] = 3
            else:  # x1 > 0.1
                if x2 <= 0.7:
                    if x5 <= 0.1:
                        y[i] = 4
                    else:
                        y[i] = 3
                else:  # x2 > 0.7
                    if x4 <= 0.1:
                        y[i] = 9
                    else:
                        y[i] = 12
        
        # Optionally add noise
        y = self._add_noise(y)
        return X, y

    def low_lei_candes(self):
        # we generate data with multinormal covariates with mean zero and covariance matrix with1 on and diag and 0.01
        # the response is equal to g(x_0) * g(x_1) + noise. g(x) = 2/(1+exp(-12(x-0.5)))
        if not self.n_features == 10:
            LOGGER.debug("n_features is not 10, using 10 for low_lei_candes")
            self.n_features = 10
        X = self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=np.eye(self.n_features) + 0.01 - 0.01 * np.eye(self.n_features), size=self.n_samples)
        x_0, x_1 = X[:, 0], X[:, 1]
        def g(x):
            return 2 / (1 + np.exp(-12 * (x - 0.5)))
        y = g(x_0) * g(x_1) + self.rng.normal(0, self.noise, size=self.n_samples)
        return X, y
    
    def high_lei_candes(self):
        # same as low_lei_candes but with 100 features
        if not self.n_features == 100:
            LOGGER.debug("n_features is not 100, using 100 for high_lei_candes")
            self.n_features = 100
        np.random.seed(self.random_seed)
        X = self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=np.eye(self.n_features) + 0.01 - 0.01 * np.eye(self.n_features), size=self.n_samples)
        x_0, x_1 = X[:, 0], X[:, 1]
        def g(x):
            return 2 / (1 + np.exp(-12 * (x - 0.5)))
        np.random.seed(self.random_seed)
        y = g(x_0) * g(x_1) + self.rng.normal(0, self.noise, size=self.n_samples)
        return X, y

    def lss(self):
        # local sparse spiky model (behr et al 2021)
        # the covaiares are generated similar to low_lei_candes
        # the response is generated as follows: f(x) = 2*1_{x_0<0 & x_2>0} -3 * 1_{x_4> 0 & x_5 > 1} + 0.8 * 1_{x_2<1.5 & x_4<1}
        if not self.n_features == 10:
            LOGGER.debug("n_features is not 10, using 10 for lss")
            self.n_features = 10
        X = self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=np.eye(self.n_features) + 0.01 - 0.01 * np.eye(self.n_features), size=self.n_samples)
        x_0, x_2, x_4, x_5 = X[:, 0], X[:, 2], X[:, 4], X[:, 5]
        y = 2 * (x_0 < 0) * (x_2 > 0) - 3 * (x_4 > 0) * (x_5 > 1) + 0.8 * (x_2 < 1.5) * (x_4 < 1)
        return X, y

    def piecewise_linear(self):
        # piecese wise linear function from Kunzel et al 2019
        # the covariates are 20 dimentional with the same mean and variance as in low_lei_candes
        # the is linear with the three pieces x_19 > 4 x_19 < -4 and 4 <= x_19 <= -4
        # the coeffcients anre sampled uniformly from [-15, 15]
        if not self.n_features == 20:
            LOGGER.debug("n_features is not 20, using 20 for piecewise_linear")
            self.n_features = 20
        X = self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=np.eye(self.n_features) + 0.01 - 0.01 * np.eye(self.n_features), size=self.n_samples)
        x_19 = X[:, 19]
        y = np.zeros(self.n_samples)
        idx_1 = x_19 > 4
        idx_2 = x_19 < -4
        idx_3 = (x_19 >= -4) & (x_19 <= 4)
        # set the seed for reproducibility
        # set the seed to 1, 2, 3 for the three pieces
        np.random.seed(1)
        coefs1 = np.random.uniform(-15, 15, size=self.n_features)
        np.random.seed(2)
        coefs2 = np.random.uniform(-15, 15, size=self.n_features)
        np.random.seed(3)
        coefs3 = np.random.uniform(-15, 15, size=self.n_features)
        y[idx_1] = X[idx_1] @ coefs1
        y[idx_2] = X[idx_2] @ coefs2
        y[idx_3] = X[idx_3] @ coefs3
        y += self.rng.normal(0, self.noise, size=self.n_samples)
        return X, y
    
    def sum(self):
        # the covariates are 10 dimentional  with mean 0 variance 1 and covariance 0.01 if i = j+10 otherwise 0
        cov_matrix = np.eye(self.n_features)
        for i in range(self.n_features):
            for j in range(self.n_features):
                if i == j + 10:
                    cov_matrix[i, j] = 0.01
        X = self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=cov_matrix, size=self.n_samples)
        y = np.sum(X, axis=1)
        y+= self.rng.normal(0, self.noise, size=self.n_samples)
        return X, y

    def tree(self):
        # the covariates are 10 dimentional  with mean 0 variance 1 and covariance 0.01 like in low_lei_candes
        # the response is generated as follows: f(x) = 2*1_{x_0<0 & x_2>0} -3 * 1_{x_4> 0 & x_5 > 1} + 0.8 * 1_{x_2<1.5 & x_4<1}
        if not self.n_features == 10:
            LOGGER.debug("n_features is not 10, using 10 for tree")
            self.n_features = 10
        np.random.seed(self.random_seed)
        X = self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=np.eye(self.n_features) + 0.01 - 0.01 * np.eye(self.n_features), size=self.n_samples)
        # fit a cart tree on standard normal data with max depth 7
        np.random.seed(self.random_seed)
        y_cart = self.rng.normal(0, 1, size=self.n_samples)
        tree_file = os.path.join( "models", "tree_model.pkl")
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(tree_file), exist_ok=True)
        if os.path.exists(tree_file):
            with open(tree_file, 'rb') as file:
                tree = pickle.load(file)
        else:
            tree = DecisionTreeRegressor(max_depth=7, random_state=self.random_seed)

            tree.fit(X, y_cart)
            # Save the model using pickle
            with open(tree_file, 'wb') as file:
                pickle.dump(tree, file)


        # print the tree splits and feature indices
        LOGGER.info(f"Tree splits: {tree.tree_.threshold[0:10]}")
        LOGGER.info(f"Tree feature indices: {tree.tree_.feature[0:10]}")
        # set the seed 
        np.random.seed(self.random_seed)
        y = tree.predict(X) + self.rng.normal(0, self.noise, size=self.n_samples)
        return X, y

    def self_test(self):
        """
        Test the generator with various combinations of n_samples, n_features, and noise.
        Ensures that all scenarios produce correctly shaped data.
        """
        test_params = [
            {"n_samples": 50, "n_features": 1, "noise": 0.1},
            {"n_samples": 100, "n_features": 2, "noise": 0.2},
            {"n_samples": 200, "n_features": 5, "noise": 0.5},
        ]

        scenarios = [
            "linear", "heteroscedastic", "nonoverlapping_vs_overlapping",
            "spiky_vs_smooth", "cyclic", "distribution_shift", "multimodal",
            "imbalanced", "piecewise_linear", "tied_y", "tied_x",
            "friedman1", "friedman2", "friedman3"
            "imbalanced", "piecewise_linear", "tied_y", "tied_x", "tree", "sum", "piecewise_linear", "low_lei_candes", "high_lei_candes", "lss"
        ]

        for params in test_params:
            self.n_samples = params["n_samples"]
            self.n_features = params["n_features"]
            self.noise = params["noise"]

            for scenario in scenarios:
                data : tuple = self.generate(scenario)
                if isinstance(data, dict):  # Handle scenarios with multiple cases
                    for key, (X, y) in data.items():
                        if scenario == "distribution_shift":  # Special handling for train/test split
                            expected_shape = (self.n_samples // 2, self.n_features)
                            expected_len = self.n_samples // 2
                        else:
                            expected_shape = (self.n_samples, self.n_features)
                            expected_len = self.n_samples

                        assert X.shape == expected_shape, f"{scenario} ({key}): X shape mismatch"
                        assert len(y) == expected_len, f"{scenario} ({key}): y length mismatch"
                else:
                    X, y = data
                    assert X.shape == (self.n_samples, self.n_features), f"{scenario}: X shape mismatch"
                    assert len(y) == self.n_samples, f"{scenario}: y length mismatch"
