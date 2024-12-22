import numpy as np

from params import BARTParams

class Move:
    """
    Base class for moves in the BART sampler.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray):
        """
        Initialize the move.

        Parameters:
        - random_state: int
            Random state for reproducibility.
        - current: BARTParams
            Current state of the BART model.
        - trees_changed: np.ndarray
            Indices of trees that were changed.
        """
        self.current = current
        self.proposed = None
        self.trees_changed = trees_changed

    def propose(self, generator):
        """
        Propose a new state.
        """
        self.proposed = self.current.copy()
        return self.proposed

    def get_log_prior_ratio(self):
        log_prior_current = self.current.get_log_prior(self.trees_changed)
        log_prior_proposed = self.proposed.get_log_prior(self.trees_changed)
        return log_prior_proposed - log_prior_current

    def get_log_marginal_lkhd_ratio(self, marginalize: bool=False):
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
        if not marginalize:
            marginal_lkhd_current = self.get_log_marginal_lkhd(self.trees_changed)
            marginal_lkhd_proposed = self.get_log_marginal_lkhd(self.trees_changed)
        else:
            marginal_lkhd_current = self.get_log_marginal_lkhd(np.arange(self.current.n_trees))
            marginal_lkhd_proposed = self.get_log_marginal_lkhd(np.arange(self.current.n_trees))
        return marginal_lkhd_proposed - marginal_lkhd_current
    
    def get_log_MH_ratio(self, marginalize : bool=False):
         return self.get_log_prior_ratio() + self.get_log_marginal_lkhd_ratio(marginalize)


class Grow(Move):
    """
    Move to grow a tree.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, generator):
        self.proposed = self.current.copy()
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = tree.get_random_leaf(generator)
        tree.split_leaf(node_id, generator)
        return self.proposed


class Prune(Move):
    """
    Move to prune a tree.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, generator):
        self.proposed = self.curent.copy()
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = tree.get_random_terminal_split(generator)
        tree.prune_split(node_id)
        return self.proposed

class Change(Move):
    """
    Move to change a tree.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, generator):
        self.proposed = self.curent.copy()
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = tree.get_random_split(generator)
        tree.vars[node_id] = var
        tree.thresholds[node_id] = threshold
        return self.proposed


class Swap(Move):
    """
    Move to swap two trees.
    """
    pass

all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap}