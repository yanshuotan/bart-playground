import numpy as np
from typing import Union, List


class BanditEncoder:
    """
    A utility class for encoding features for multi-armed bandit problems.
    This class handles different encoding strategies for the arms and features.
    """
    def __init__(self, n_arms: int, n_features: int, encoding: str) -> None:
        self.n_arms = n_arms
        self.n_features = n_features
        self.encoding = encoding

        if encoding == 'one-hot':
            self.combined_dim = n_features + n_arms
        elif encoding == 'multi':
            self.combined_dim = n_features * n_arms
        elif encoding == 'separate':
            self.combined_dim = n_features
            # "separate" encoding means that we will use different models with the feature vector as is
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    def encode(self, x: Union[np.ndarray, List[float]], arm: int) -> np.ndarray:
        """
        Encode the feature vector x for a specific arm using the specified encoding strategy.

        Parameters:
            x (array-like): Feature vector
            arm (int): Index of the arm to encode for; if arm == -1, then encode all arms

        Returns:
            np.ndarray: Encoded feature vector
        """
        x = np.array(x).reshape(1, -1)

        if arm == -1:
            range_arms = range(self.n_arms)
        else:
            range_arms = [arm]

        total_arms = len(range_arms)
        x_combined = np.zeros((total_arms, self.combined_dim))

        if self.encoding == 'one-hot':
            # One-hot encoded treatment options
            for row_idx, arm in enumerate(range_arms):
                x_combined[row_idx, :self.n_features] = x
                x_combined[row_idx, self.n_features + arm] = 1
        elif self.encoding == 'multi':
            # Block structure approach (data_multi style)
            for row_idx, arm in enumerate(range_arms):
                start_idx = arm * self.n_features
                end_idx = start_idx + self.n_features
                x_combined[row_idx, start_idx:end_idx] = x
        elif self.encoding == 'separate':
            x_combined = x
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        return x_combined


