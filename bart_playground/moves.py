import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .params import Parameters, Tree
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
    def __init__(self, current : Parameters, trees_changed: np.ndarray,
                 possible_thresholds : dict, tol : int = 100):
        if not possible_thresholds:
            raise ValueError("Possible thresholds must be provided for grow move.")
        super().__init__(current, trees_changed, possible_thresholds, tol)
        assert len(trees_changed) == 1

    def is_feasible(self):
        return True
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, tree.leaves)
        var = generator.integers(tree.dataX.shape[1])
        threshold = fast_choice(generator, self.possible_thresholds[var])
        n_leaves = tree.n_leaves
        success = tree.split_leaf(node_id, var, threshold)
        n_splits = len(tree.terminal_split_nodes)
        self.log_tran_ratio = np.log(n_leaves) - np.log(n_splits)
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
        return len(tree.terminal_split_nodes) > 0

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, tree.terminal_split_nodes)
        n_splits = len(tree.terminal_split_nodes)
        tree.prune_split(node_id)
        n_leaves = tree.n_leaves
        self.log_tran_ratio = np.log(n_splits) - np.log(n_leaves)
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
        return len(tree.nonterminal_split_nodes) > 0

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        parent_id = fast_choice(generator, tree.nonterminal_split_nodes)
        lr = generator.integers(1, 3) # Choice of left/right child
        child_id = 2 * parent_id + lr
        if tree.vars[child_id] == -1: # Change to the other child if this is a leaf
            child_id = 2 * parent_id + 3 - lr
        success = tree.swap_split(parent_id, child_id) # If no empty leaves are created
        return success

class Break(Move):
    """
    Move to Split a tree into two trees
    """
    # Assume that the input is defined in TreeParams Class
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol
    
    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[0]]
        return len(tree.split_nodes) > 1

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        valid_split_nodes = [node for node in tree.split_nodes if node != 0]
        node_id = generator.choice(valid_split_nodes)
        n_splits = len(tree.split_nodes)-1
        tree_new = tree.break_new(node_id)
        tree.prune_split(node_id, recursive= True)
        n_leaves = tree.n_leaves
        proposed.trees.append(tree_new)
        proposed.update_tree_num()
        proposed.update_cache(add_ids = [proposed.n_trees-1])
        self.log_tran_ratio = np.log(n_splits) - np.log(n_leaves)
        return True

class Combine(Move):
    """
    Move to combine two trees into one tree
    """
    # Assume that the input is defined in TreeParams Class
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 2
        self.tol = tol

    def is_feasible(self):
        # Check if both trees have split nodes
        tree1 = self.current.trees[self.trees_changed[0]]
        tree2 = self.current.trees[self.trees_changed[1]]
        return len(tree1.split_nodes) > 0 and len(tree2.split_nodes) > 0


    def try_propose(self, proposed, generator):
        tree1 = proposed.trees[self.trees_changed[0]]
        tree2 = proposed.trees[self.trees_changed[1]]
        node_id = generator.choice(tree1.leaves)
        n_leaves = tree1.n_leaves
        success = tree1.combine_two(node_id, tree2)
        if success: 
            n_splits = len(tree1.split_nodes)
            self.log_tran_ratio =  np.log(n_leaves) - np.log(n_splits-1)
            proposed.update_cache(delete_ids = [self.trees_changed[1]])
            proposed.trees.remove(tree2)
            proposed.update_tree_num()
        return success    
    
class Birth(Move):
    """
    Add a new root to the ensemble. Use the same logic as the Break move for easier implementation.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 1
        self.tol = tol
    
    def is_feasible(self):
        return True

    def try_propose(self, proposed, generator):
        m = proposed.n_trees
        # Create a new root tree
        tree = Tree.new(dataX = proposed.trees[0].dataX)
        proposed.trees.append(tree)
        proposed.update_tree_num()
        proposed.update_cache(add_ids = [proposed.n_trees-1])
        num_root = sum(1 for tree in proposed.trees if tree.only_root)
        self.log_tran_ratio = 0 #np.log(m+1) - np.log(num_root)
        return True

class Death(Move):
    """
    Remove a root tree from the ensemble. Use the same logic as the Combine move for easier implementation.
    """
    def __init__(self, current: Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 2
        self.tol = tol

    def is_feasible(self):
        tree = self.current.trees[self.trees_changed[1]]
        return tree.only_root

    def try_propose(self, proposed, generator):
        num_root = sum(1 for tree in proposed.trees if tree.only_root)
        # Remove the selected root tree
        tree = proposed.trees[self.trees_changed[1]] # The root tree
        proposed.update_cache(delete_ids = [self.trees_changed[1]])
        proposed.trees.remove(tree)
        proposed.update_tree_num()
        m = proposed.n_trees
        # Update log transition ratio
        self.log_tran_ratio = 0 #np.log(num_root) - np.log(m+1)
        return True
 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "break": Break,
            "combine": Combine,
            "birth": Birth,
            "death": Death}
