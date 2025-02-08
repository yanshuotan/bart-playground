import numpy as np
from abc import ABC, abstractmethod

from .params import Parameters


class Move(ABC):
    """
    Base class for moves in the BART sampler.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray):
        """
        Initialize the move.

        Parameters:
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
    def is_feasible(self):
        """
        Check whether move is feasible.
        """
        pass

    @abstractmethod
    def try_propose(self, proposed, generator):
        """
        Try to propose a new state.
        """
        pass

class Grow(Move):
    """
    Move to grow a new split.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def is_feasible(self):
        return True
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = generator.choice(tree.leaves)
        var = generator.integers(tree.data.p)
        threshold = generator.choice(tree.data.thresholds[var])
        success = tree.split_leaf(node_id, var, threshold)
        return success

class Prune(Move):
    """
    Move to prune a terminal split.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        return len(tree.terminal_split_nodes) > 0

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = generator.choice(tree.terminal_split_nodes)
        tree.prune_split(node_id)
        return True

class Change(Move):
    """
    Move to change the split variable and threshold for an internal node.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        return len(tree.split_nodes) > 0
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = generator.choice(tree.split_nodes)
        var = generator.integers(tree.data.p)
        threshold = generator.choice(tree.data.thresholds[var])
        success = tree.change_split(node_id, var, threshold)
        return success

class Swap(Move):
    """
    Move to swap the split variables and thresholds for a pair of parent-child nodes.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        return len(tree.nonterminal_split_nodes) > 0

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        parent_id = generator.choice(tree.nonterminal_split_nodes)
        lr = generator.integers(1, 3) # Choice of left/right child
        child_id = 2 * parent_id + lr
        if tree.vars[child_id] == -1: # Change to the other child if this is a leaf
            child_id = 2 * parent_id + 3 - lr
        success = tree.swap_split(parent_id, child_id) # If no empty leaves are created
        return success
    
 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap}
