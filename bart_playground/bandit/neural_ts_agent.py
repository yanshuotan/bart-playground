import numpy as np
import torch
from .agent import BanditAgent
from .NeuralTS.learner_diag import NeuralTSDiag # Assuming learner_diag.py contains NeuralTSDiag

class NeuralTSDiagAgent(BanditAgent):
    """
    Bandit agent implementing NeuralTSDiag. Assumes CUDA is available.
    It uses a single NeuralTSDiag model. The input to NeuralTSDiag is an augmented
    context, created by combining the original context with a one-hot encoding of each arm.
    The NeuralTSDiag model then selects the best (context, arm) combination.
    
    Note: The underlying NeuralTSDiag model from NeuralTS-main.learner_diag.py
    hardcodes CUDA usage.
    """
    def __init__(self,
                 n_arms: int,
                 n_features: int,
                 lambda_param: float = 0.001,
                 nu_param: float = 1.0,
                 hidden_size: int = 100,
                 style: str = 'ts',
                 delay: int = 1,  # Added delay parameter with default 1
                 nn_lr: float = 0.01): # nn_lr is noted but not directly used as NeuralTSDiag manages LR
        """
        Initialize the NeuralTSDiagAgent. Assumes CUDA is available.

        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features in the original context.
            lambda_param (float): Lambda regularization parameter for NeuralTSDiag (passed as 'lamdba').
            nu_param (float): Nu exploration parameter for NeuralTSDiag.
            hidden_size (int): Hidden layer size for the neural network in NeuralTSDiag.
            style (str): Exploration style for NeuralTSDiag, 'ts' or 'ucb'.
            delay (int): Delay for model updates in NeuralTSDiag.
            nn_lr (float): Learning rate (Note: NeuralTSDiag manages its LR internally).
        """
        self.augmented_dim = n_features * n_arms
        super().__init__(n_arms, n_features)

        # NeuralTSDiag class in learner_diag.py uses a typo 'lamdba' for its parameter
        self.model = NeuralTSDiag(dim=self.augmented_dim,
                                  lamdba=lambda_param, # Typo 'lamdba' is intentional
                                  nu=nu_param,
                                  hidden=hidden_size,
                                  style=style)
        self.model.delay = delay

    def _create_augmented_context_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Creates a batch of augmented contexts. Each row corresponds to an arm.
        Calls _create_single_augmented_context for each arm and stacks the results.
        """
        batch = np.zeros((self.n_arms, self.augmented_dim))
        
        for arm in range(self.n_arms):
            # Get single augmented context and add to batch
            single_context = self._create_single_augmented_context(x, arm)
            batch[arm] = single_context
        
        return batch

    def _create_single_augmented_context(self, x: np.ndarray, arm: int) -> np.ndarray:
        """
        Creates a single augmented context for a given original context and arm.
        Augmented context = [original_context, one_hot_encoded_arm].
        x is original context, arm is the chosen arm index.
        Returns shape (1, augmented_dim)
        """
        if x.ndim == 1:
            x_orig = x.reshape(1, -1)
        elif x.ndim == 2 and x.shape[0] == 1:
            x_orig = x
        else:
            raise ValueError(f"Input context x must be 1D or 2D with 1 row. Got shape {x.shape}")

        # Create vector with all zeros
        augmented_context = np.zeros((1, self.augmented_dim))
        
        # Place features at arm-specific position
        start_idx = arm * self.n_features
        end_idx = start_idx + self.n_features
        augmented_context[0, start_idx:end_idx] = x_orig
        
        return augmented_context

    def choose_arm(self, x: np.ndarray, **kwargs) -> int:
        """
        Choose an arm.
        x is the original context vector (n_features,).
        """
        augmented_context_batch_np = self._create_augmented_context_batch(x)

        # NeuralTSDiag.select expects a numpy array and returns the index of the best row.
        # This index directly corresponds to our arm index.
        # The select method in NeuralTSDiag handles moving data to CUDA internally.
        chosen_arm_index, _, _, _ = self.model.select(augmented_context_batch_np)
        
        if isinstance(chosen_arm_index, torch.Tensor):
            chosen_arm_index = chosen_arm_index.item()
            
        return int(chosen_arm_index)

    def update_state(self, arm: int, x: np.ndarray, y: float) -> "BanditAgent":
        """
        Update the agent's state.
        arm: index of the chosen arm.
        x: original context vector.
        y: observed reward.
        """
        augmented_x_chosen_arm_np = self._create_single_augmented_context(x, arm)
        
        self.model.train(augmented_x_chosen_arm_np, float(y))
        return self

# To make it directly usable if someone imports *, though explicit import is better.
__all__ = ['NeuralTSDiagAgent']
