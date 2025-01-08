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
    
class Split(Move):
    """
    Move to Split a tree into two trees
    """
    # Assume that the input is defined in TreeParams Class
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol

    def collect_values(tree, start_index):
        # Tree here refers to class Tree
        if start_index >= len(tree.thresholds):
            return []
        thresholds = []
        vars = []
        queue = [start_index]  
        while queue:
            current_index = queue.pop(0)  
            if current_index < len(tree.thresholds): 
                thresholds.append(tree.thresholds[current_index]) 
                vars.append(tree.vars[current_index]) 
            
                left_index = 2 * current_index + 1
                right_index = 2 * current_index + 2
                if left_index < len(tree.thresholds):
                    queue.append(left_index)
                if right_index < len(tree.thresholds):
                    queue.append(right_index)
        return thresholds, vars
    
    def set_subtree_zero(tree, index):
        # Split_tree here refers to np.ndarray
        local_tree = tree.copy()
        if index >= local_tree.thresholds.size:
            return local_tree
        local_tree.vars[index] = -2
        local_tree.threshold[index] = np.nan
    
        left_index = 2 * index + 1
        local_tree = Split.set_subtree_zero(local_tree, left_index)
    
        right_index = 2 * index + 2
        local_tree = Split.set_subtree_zero(local_tree, right_index)
        return local_tree

    def propose(self, generator):
        # find node and collect thresholds
        self.proposed = self.current.copy(self.trees_changed)
        tree = self.proposed.trees[self.trees_changed[0]]
        node_id = generator.choice(tree.split_nodes)
        thersholds,vars = self.collect_values(tree,node_id)
        # update thresholds and vars
        tree = self.set_subtree_zero(tree,node_id)
        tree.vars[node_id] = -1
        # generate new tree 
        tree_new = Tree(data=tree.data)        
        # Here I assume the length of our original tree is larger than the subtree we split
        # give the corresponding value to new tree
        tree_new.thresholds[:len(thersholds)]=thersholds
        tree_new.vars[:len(vars)]=vars
        
        self.proposed.trees.append(tree_new)

        return self.proposed
 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "split": Split}