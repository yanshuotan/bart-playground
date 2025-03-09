import numpy as np


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
