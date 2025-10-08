import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional
from .params import Parameters
from .util import fast_choice, fast_choice_with_weights


class Move(ABC):
    """
    Base class for moves in the BART sampler.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, 
                 possible_thresholds : Optional[dict] = None, tol : int = 100):
        """
        Initialize the move.

        Parameters:
        - current: BARTParams
            Current state of the BART model.
        - trees_changed: np.ndarray
            Indices of trees that were changed.
        """
        self.current = current
        # self.proposed = None
        self.trees_changed = trees_changed
        self._possible_thresholds = possible_thresholds
        self.s = current.global_params.get("s", None)
        self.tol = tol
        self.log_tran_ratio = 0 # The log of remaining transition ratio after cancellations in the MH acceptance probability. 

    @property
    def possible_thresholds(self):
        assert self._possible_thresholds, "possible_thresholds must be initialized"
        return self._possible_thresholds
    @property
    def _num_possible_proposals(self):
        return self.tol

    def propose(self, generator):
        """
        Propose a new state.
        """
        if self.is_feasible():
            for _ in range(self._num_possible_proposals):
                proposed = self.current.copy(self.trees_changed)
                success = self.try_propose(proposed, generator)
                if success:
                    self.proposed = proposed
                    return True
            # If exit loop without returning, have exceeded tol tries without 
            # finding a valid proposal.
        return False

    @abstractmethod
    def is_feasible(self) -> bool:
        """
        Check whether move is feasible.
        """
        pass

    @abstractmethod
    def try_propose(self, proposed, generator) -> bool:
        """
        Try to propose a new state.
        """
        pass

class Grow(Move):
    """
    Move to grow a new split.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray,
                 possible_thresholds : dict, tol : int = 100):
        if not possible_thresholds:
            raise ValueError("Possible thresholds must be provided for grow move.")
        super().__init__(current, trees_changed, possible_thresholds, tol)
        assert len(trees_changed) == 1

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        self.cur_leaves = tree.leaves
        self.cur_n_terminal_splits = len(tree.terminal_split_nodes)
        return True
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, self.cur_leaves)
        var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
        threshold = fast_choice(generator, self.possible_thresholds[var])
        n_leaves = len(self.cur_leaves)
        
        success = tree.split_leaf(node_id, var, threshold)
        if node_id % 2:
            neighbor = node_id + 1
        else:
            neighbor = node_id - 1
        # Update the number of non-terminal splits
        # + 1 only if parent is a non-terminal split
        n_splits = self.cur_n_terminal_splits + 1 - tree.is_leaf(neighbor)
        self.log_tran_ratio = math.log(n_leaves) - math.log(n_splits)
        return success

class Prune(Move):
    """
    Move to prune a terminal split.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray,
                 possible_thresholds = None, tol : int = 100):
        super().__init__(current, trees_changed, tol = tol)
        assert len(trees_changed) == 1

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        self.cur_terminal_split_nodes = tree.terminal_split_nodes
        return len(self.cur_terminal_split_nodes) > 0

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, self.cur_terminal_split_nodes)
        n_splits = len(self.cur_terminal_split_nodes)
        
        tree.prune_split(node_id)
        n_leaves = tree.n_leaves
        self.log_tran_ratio = math.log(n_splits) - math.log(n_leaves)
        return True

class Change(Move):
    """
    Move to change the split variable and threshold for an internal node.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray,
                 possible_thresholds : dict, tol : int = 100):
        if not possible_thresholds:
            raise ValueError("Possible thresholds must be provided for change move.")
        super().__init__(current, trees_changed, possible_thresholds, tol)
        assert len(trees_changed) == 1

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        return len(tree.split_nodes) > 0
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, tree.split_nodes)
        var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
        threshold = fast_choice(generator, self.possible_thresholds[var])
        
        success = tree.change_split(node_id, var, threshold)
        return success

class Swap(Move):
    """
    Move to swap the split variables and thresholds for a pair of parent-child nodes.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray,
                 possible_thresholds = None, tol : int = 100):
        super().__init__(current, trees_changed, tol = tol)
        assert len(trees_changed) == 1
        self.swappable_pairs = []
        self.idx = 0
        
    @property
    def _num_possible_proposals(self):
        return min(self.tol, len(self.swappable_pairs))
    
    def _ini_swappable_pairs(self):
        tree = self.current.trees[self.trees_changed[0]]
        nonterminal_split_nodes = tree.nonterminal_split_nodes

        # Collect all valid parent-child pairs where the child is also a split node.
        self.swappable_pairs = [
            (parent_id, 2 * parent_id + lr)
            for parent_id in nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * parent_id + lr] != -1
        ]
        self.idx = 0

    def is_feasible(self):
        '''
        Note that this method has a side effect of initializing the swappable_pairs.
        '''
        self._ini_swappable_pairs()
        return self._num_possible_proposals > 0

    def try_propose(self, proposed, generator):
        if self.idx == 0: # Shuffle the pairs once at the start
            generator.shuffle(self.swappable_pairs)
            
        parent_id, child_id = self.swappable_pairs[self.idx]
        tree = proposed.trees[self.trees_changed[0]]
        success = tree.swap_split(parent_id, child_id)  # If no empty leaves are created
        self.idx += 1
        return success
    
 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap}

# Mapping of each move to its contrary move used in MH ratio adjustments
contrary_moves = {
    "grow": "prune",
    "prune": "grow",
    "change": "change",
    "swap": "swap",
}
