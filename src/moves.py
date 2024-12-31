import numpy as np
from abc import ABC, abstractmethod

from params import BARTParams

class Move(ABC):
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

    @abstractmethod
    def propose(self, generator):
        """
        Propose a new state.
        """
        pass

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
    def __init__(self, current : BARTParams, trees_changed: np.ndarray, tol=1000):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def propose(self, generator):
        for _ in range(self.tol):
            self.proposed = self.current.copy(self.trees_changed)
            tree = self.proposed.trees[self.trees_changed[0]]
            node_id = generator.choice(tree.leaves)
            var = generator.integers(tree.data.p)
            threshold = generator.choice(tree.data.thresholds[var])
            if tree.split_leaf(node_id, var, threshold): # If no empty leaves are created
                return self.proposed
        self.proposed = self.current # Exceeded tol tries without finding a valid proposal. Stay at current state
        return self.proposed

class Prune(Move):
    """
    Move to prune a tree.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1

    def propose(self, generator):
        self.proposed = self.curent.copy(self.trees_changed)
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = generator.choice(tree.terminal_split_nodes)
        tree.prune_split(node_id)
        return self.proposed

class Change(Move):
    """
    Move to change a tree.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray, tol=1000):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def propose(self, generator):
        for _ in range(self.tol):
            self.proposed = self.curent.copy(self.trees_changed)
            tree = self.proposed.trees[self.trees_changed[0]]
            node_id = generator.choice(tree.split_nodes)
            var = generator.integers(tree.data.p)
            threshold = generator.choice(tree.data.thresholds[var])
            tree.vars[node_id] = var
            tree.thresholds[node_id] = threshold
            if tree.update_n(node_id): # If no empty leaves are created
                return self.proposed
        self.proposed = self.current # Exceeded tol tries without finding a valid proposal. Stay at current state
        return self.proposed

class Swap(Move):
    """
    Move to swap two trees.
    """
    def __init__(self, current : BARTParams, trees_changed: np.ndarray, tol=1000):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def propose(self, generator):
        for _ in range(self.tol):
            self.proposed = self.curent.copy(self.trees_changed)
            tree = self.proposed.trees[self.trees_changed[0]]
            parent_id = generator.choice(tree.nonterminal_split_nodes)
            child_id = 2 * parent_id + generator.integers(1, 3)
            parent_var = tree.vars[parent_id]
            child_var = tree.vars[child_id]
            parent_threshold = tree.thresholds[parent_id]
            child_threshold = tree.thresholds[child_id]
            tree.vars[parent_id] = child_var
            tree.vars[child_id] = parent_var
            tree.thresholds[parent_id] = child_threshold
            tree.thresholds[child_id] = parent_threshold
            if tree.update_n(parent_id): # If no empty leaves are created
                return self.proposed
        self.proposed = self.current # Exceeded tol tries without finding a valid proposal. Stay at current state
        return self.proposed

all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap}