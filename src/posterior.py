import numpy as np
from scipy.linalg import svd

from params import BARTParams, TreeParams
from moves import Move
from priors import BARTPrior

class BARTPosterior:
    """
    Represents the posterior for the BART model.
    """
    def __init__(self, params : BARTParams, prior: BARTPrior, X : np.ndarray, y : np.ndarray):
        """
        Initialize the BART posterior.

        Parameters:
        - params: BARTParams
            Parameters of the BART model.
        - prior: BARTPrior
            Priors for the BART model.
        """
        self.params = params
        self.prior = prior
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.noise_ratio = None

    def get_leaf_indicators(self, tree_ids):
        ordinal_encoding = np.zeros((self.n, len(tree_ids)), dtype=int)
        for col, tree_id in enumerate(tree_ids):
            ordinal_encoding[:, col] = self.params[tree_id].traverse_tree(self.X)
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        leaf_indicators = one_hot_encoder.fit_transform(ordinal_encoding)
        return leaf_indicators

    def get_log_prior_ratio(self, move : Move):
        """
        Compute the ratio of priors for a given move.

        Parameters:
        - move: Move
            The move to compute the prior ratio for.

        Returns:
        - float
            Prior ratio.
        """
        self._validate_move(move)
        current_prior = np.sum([self.prior.tree_log_prior(np.self.params.trees[tree_id]) 
                                 for tree_id in move.trees_changed])
        proposed_prior = np.sum([self.prior.tree_log_prior(move.proposed.trees[tree_id]) 
                                  for tree_id in move.trees_changed])
        return proposed_prior - current_prior

    def get_log_marginal_likelihood_ratio(self, move, marginalize: bool = False):
        """
        Compute the ratio of marginal likelihoods for a given move.

        Parameters:
        - move: Move
            The move to compute the marginal likelihood ratio for.
        - marginalize: bool
            Whether to marginalize over the ensemble.

        Returns:
        - float
            Marginal likelihood ratio.
        """
        self._validate_move(move)
        if not marginalize:
            resids = self.y - self.params.evaluate(holdout = move.trees_changed)
        leaf_indicators_current = self.get_leaf_indicators(move.trees_changed)
        leaf_indicators_proposed = self.get_leaf_indicators
        
        current_marginal_lkhd = self.get_log_marginal_lkhd(leaf_indicators_current)
        proposed_marginal_lkhd = self.get_log_marginal_lkhd(leaf_indicators_proposed)
        return proposed_marginal_lkhd - current_marginal_lkhd

    def get_log_marginal_lkhd(leaf_indicators):
        U, S, _ = svd(leaf_indicators)
        logdet = np.sum(np.log(S ** 2 / self.noise_ratio + 1))
        r_U_coefs = U.T @ resids
        r_U = U @ y_U_coefs
        ls_resids = np.sum((resids - r_U) ** 2)
        ridge_bias = np.sum(r_U_coefs ** 2 / (S ** 2 / self.noise_ratio + 1))
        return - (logdet + (ls_resids + ridge_bias) / self.params.sigma2) / 2

    def _validate_move(self, move):
        if move.proposed is None:
            raise ValueError("Move has not been proposed yet.")
        if move.current is not self.params:
            raise ValueError("Current state of move must be equal to self.params")

    def sample_sigma2(self):
        """
        Sample the noise variance.

        Returns:
        - float
            Sampled noise variance.
        """
        pass

    def get_leaf_posterior_mean(self, tree_index: int):
        """
        Compute the posterior mean for the leaf parameters of a given tree.

        Parameters:
        - tree_index: int
            Index of the tree.

        Returns:
        - float
            Posterior mean.
        """
        pass

    def sample_leaf_params(self, tree_index: int):
        """
        Sample the leaf parameters for a given tree.

        Parameters:
        - tree_index: int
            Index of the tree.

        Returns:
        - np.ndarray
            Sampled leaf parameters.
        """
        pass