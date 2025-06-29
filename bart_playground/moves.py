import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional
from .params import Parameters
from .util import fast_choice
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
        self.tol = tol
        self.log_tran_ratio = 0 # The log of remaining transition ratio after cancellations in the MH acceptance probability. 

    @property
    def possible_thresholds(self):
        assert self._possible_thresholds, "possible_thresholds must be initialized"
        return self._possible_thresholds

    def propose(self, generator):
        """
        Propose a new state.
        """
        if self.is_feasible():
            for _ in range(self.tol):
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
        var = generator.integers(tree.dataX.shape[1])
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
        var = generator.integers(tree.dataX.shape[1])
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

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        self.cur_nonterminal_split_nodes = tree.nonterminal_split_nodes
        return len(self.cur_nonterminal_split_nodes) > 0
    
    def propose(self, generator):
        """
        Try all possible parent-child pairs, return True if any swap succeeds.
        """
        if not self.is_feasible():
            return False
        tree = self.current.trees[self.trees_changed[0]]
        # Collect all possible parent-child pairs
        pairs = [
            (parent_id, 2 * parent_id + lr)
            for parent_id in self.cur_nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * parent_id + lr] != -1
        ]

        generator.shuffle(pairs)
        tries = 0
        for parent_id, child_id in pairs:
            if tries >= self.tol:
                break
            proposed = self.current.copy(self.trees_changed)
            success = self.try_propose(proposed, generator, parent_id, child_id)
            tries += 1
            if success:
                self.proposed = proposed
                return True
        return False
    
    def try_propose(self, proposed, generator, parent_id, child_id):
        tree = proposed.trees[self.trees_changed[0]]   
        success = tree.swap_split(parent_id, child_id) # If no empty leaves are created
        return success
    
 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap}
