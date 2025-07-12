import numpy as np
import torch
from .agent import BanditAgent
from .baselines.NeuralTS.learner_diag import NeuralTSDiag
from .baselines.NeuralTS.learner_diag_linear import LinearTSDiag
from .baselines.NeuralTS.learner_diag_kernel import KernelTSDiag
from .baselines.NeuralTS.learner_neural import NeuralTS
from .baselines.NeuralTS.learner_kernel import KernelTS
from .baselines.NeuralTS.learner_linear import LinearTS

class NeuralPaperAgent(BanditAgent):
    def __init__(self,
                 n_arms: int,
                 n_features: int):
        self.augmented_dim = n_features * n_arms
        self.model.delay = 1
        self.t = 0
        # self.model should have `select` and `train` methods
        super().__init__(n_arms, n_features)
        
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
        chosen_arm_index, nrm, sig, ave_rwd = self.model.select(augmented_context_batch_np)
        
        # self.t += 1
        # if self.t % 50 == 0:
        #     print('nrm, sig, ave_rwd: {:.3f}, {:.3e}, {:.3e}'.format(nrm, sig, ave_rwd))

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
        augmented_x_chosen_arm_np = self._create_single_augmented_context(x, arm).flatten()
        
        self.model.train(augmented_x_chosen_arm_np, float(y))
        return self


class NeuralTSDiagAgent(NeuralPaperAgent):
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
                 lamdba_param: float = 0.00001,
                 nu_param: float = 0.00001,
                 style: str = 'ts'):
        """
        Initialize the NeuralTSDiagAgent. Assumes CUDA is available.

        Parameters:
            n_arms (int): Number of arms.
            n_features (int): Number of features in the original context.
            lamdba_param (float): Lambda regularization parameter for NeuralTSDiag (passed as 'lamdba').
            nu_param (float): Nu exploration parameter for NeuralTSDiag.
            style (str): Exploration style for NeuralTSDiag, 'ts' or 'ucb'.
        """
        self.augmented_dim = n_features * n_arms

        self.model = NeuralTSDiag(dim=self.augmented_dim,
                                  lamdba=lamdba_param, # lamdba = 1/lambda
                                  nu=nu_param,
                                  style=style)
        super().__init__(n_arms, n_features)

class NeuralTSAgent(NeuralPaperAgent):
    def __init__(self,
                 n_arms: int,
                 n_features: int,
                 lamdba_param: float = 0.00001,
                 nu_param: float = 0.00001,
                 style: str = 'ts'):
        self.augmented_dim = n_features * n_arms

        self.model = NeuralTS(dim=self.augmented_dim,
                              lamdba=lamdba_param, # lamdba = 1/lambda
                              nu=nu_param,
                              style=style)
        super().__init__(n_arms, n_features)

class LinearTSDiagAgent(NeuralPaperAgent):
    """
    The underlying LinearTSDiag model from NeuralTS-main, using a diagonal kernel for faster computation.
    """
    def __init__(self,
                 n_arms: int,
                 n_features: int,
                 lamdba_param: float = 1,
                 nu_param: float = 0.3, # 0.1 or 1
                 style: str = 'ts'):
        self.augmented_dim = n_features * n_arms

        self.model = LinearTSDiag(dim=self.augmented_dim,
                                  lamdba=lamdba_param, # lamdba = 1/lambda
                                  nu=nu_param,
                                  style=style)
        super().__init__(n_arms, n_features)
        
class NLinearTSAgent(NeuralPaperAgent):
    """
    The underlying LinearTS model from NeuralTS-main.
    """
    def __init__(self,
                 n_arms: int,
                 n_features: int,
                 lamdba_param: float = 1,
                 nu_param: float = 0.3, # 0.1 or 1
                 style: str = 'ts'):
        self.augmented_dim = n_features * n_arms

        self.model = LinearTS(dim=self.augmented_dim,
                              lamdba=lamdba_param, # lamdba = 1/lambda
                              nu=nu_param,
                              style=style)
        super().__init__(n_arms, n_features)
        
class KernelTSDiagAgent(NeuralPaperAgent):
    """
    The underlying KernelTSDiag model from NeuralTS-main, using a diagonal kernel for faster computation.
    """
    def __init__(self,
                 n_arms: int,
                 n_features: int,
                 lamdba_param: float = 1,
                 nu_param: float = 0.01,
                 style: str = 'ts'):
        self.augmented_dim = n_features * n_arms

        self.model = KernelTSDiag(dim=self.augmented_dim,
                                  lamdba=lamdba_param, # lamdba = 1/lambda
                                  nu=nu_param,
                                  style=style)
        super().__init__(n_arms, n_features)
    
class NKernelTSAgent(NeuralPaperAgent):
    """
    The underlying KernelTS model from NeuralTS-main.
    """
    def __init__(self,
                 n_arms: int,
                 n_features: int,
                 lamdba_param: float = 1,
                 nu_param: float = 0.01,
                 style: str = 'ts'):
        self.augmented_dim = n_features * n_arms

        self.model = KernelTS(dim=self.augmented_dim,
                              lamdba=lamdba_param, # lamdba = 1/lambda
                              nu=nu_param,
                              style=style)
        super().__init__(n_arms, n_features)