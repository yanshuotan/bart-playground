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

    @abstractmethod
    def propose(self, generator):
        """
        Propose a new state.
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
    Move to prune a terminal split.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def propose(self, generator):
        self.proposed = self.current.copy(self.trees_changed)
        tree = self.proposed.trees[self.trees_changed[0]]
        # If there are no terminal splits, can not prune
        if not tree.terminal_split_nodes:
            return self.proposed 
        node_id = generator.choice(tree.terminal_split_nodes)
        tree.prune_split(node_id)
        return self.proposed

class Change(Move):
    """
    Move to change the split variable and threshold for an internal node.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def propose(self, generator):
        for _ in range(self.tol):
            self.proposed = self.current.copy(self.trees_changed)
            tree = self.proposed.trees[self.trees_changed[0]]
            # If there are no splits, can not change
            if not tree.split_nodes:
                continue
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
    Move to swap the split variables and thresholds for a pair of parent-child nodes.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def propose(self, generator):
        for _ in range(self.tol):
            self.proposed = self.current.copy(self.trees_changed)
            tree = self.proposed.trees[self.trees_changed[0]]
            if not tree.nonterminal_split_nodes:
                return self.proposed 
            parent_id = generator.choice(tree.nonterminal_split_nodes)
            lr = generator.integers(1, 3) # Choice of left/right child
            child_id = 2 * parent_id + lr
            if tree.vars[child_id] == -1: # Change to the other child if this is a leaf
                child_id = 2 * parent_id + 3 - lr
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
