import numpy as np

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

    def get_leaf_indicators(self, tree_ids):
        for tree_id in tree_ids:
            pass


    def get_log_marginal_likelihood(self, tree_ids, ):
        """
        Compute the log marginal likelihood for the tree structures with indices in tree_ids, 
        conditioned on the tree parameters with indices not in tree_ids
        """
        residuals = self.y - self.params.evaluate([id for id in range(self.params.ntrees) if id not in tree_ids])


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
        current_marginal_lkhd = None
        proposed_marginal_lkhd = None
        return proposed_marginal_lkhd - current_marginal_lkhd

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