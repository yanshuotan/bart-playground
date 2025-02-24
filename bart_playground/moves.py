import numpy as np
from abc import ABC, abstractmethod
import math

from .params import Parameters, Tree

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
        return tree.n_leaves > 1
    
    def collect_values(self, tree, start_index):
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
        # To make sure the length is 2^d
        if len(thresholds) & (len(thresholds) - 1) != 0:
            thresholds.append(np.nan)
            vars.append(-2)

        tree_new = Tree(data=tree.data, thresholds=thresholds, vars=vars,
                        leaf_vals=np.full(len(vars),np.nan)) 
        tree_new.node_indicators = np.full((tree.data.X.shape[0], len(vars)), 0, dtype=bool)
        tree_new.node_indicators[:, 0] = True
        tree_new.n = np.full(len(vars), -2, dtype=int)
        tree_new.n[0] = tree.data.X.shape[0]
        tree_new.evals = np.zeros(tree_new.n[0])

        return tree_new
    
    def set_subtree_zero(self, tree, index,toleaf = True):
        # Split_tree here refers to np.ndarray
        local_tree = tree.copy()
        if index >= len(local_tree.thresholds):
            return local_tree
        local_tree.vars[index] = -1 if toleaf else -2
        local_tree.thresholds[index] = np.nan
        local_tree.leaf_vals[index] = np.nan
    
        left_index = 2 * index + 1
        local_tree = self.set_subtree_zero(local_tree, left_index,toleaf=False)
    
        right_index = 2 * index + 2
        local_tree = self.set_subtree_zero(local_tree, right_index,toleaf=False)
        return local_tree

    def try_propose(self, proposed, generator):
        # find node and collect thresholds
        tree = proposed.trees[self.trees_changed[0]]
        node_id = generator.choice(tree.split_nodes)
        tree_new = self.collect_values(tree,node_id)
        tree_remain = self.set_subtree_zero(tree,node_id)
        
        success = tree_remain.update_n() & tree_new.update_n()
        if success:
            proposed.trees.remove(tree)
            proposed.trees.append(tree_remain)
            proposed.trees.append(tree_new)
        return success
    
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
        return True

    def create_combined(self, tree1, tree2):
        l1 = []
        for i in range(len(tree1.thresholds)):
            if tree1.vars[i] == -1:
                l1.append(i)
        d1 = int(np.ceil(np.log2(l1[-1] + 2)))
        l2 = []
        for i in range(len(tree2.thresholds)):
            if tree2.vars[i] == -1:
                l2.append(i)
        d2 = int(np.ceil(np.log2(l2[-1] + 2)))
        length = pow(2,d1+d2-1)
        if length < len(tree1.vars):
            length = len(tree1.vars)
        
        # d1 = int(np.ceil(np.log2(len(tree1.vars))))
        # d2 = int(np.ceil(np.log2(len(tree2.vars))))
        # length = pow(2,d1+d2-1)
        new_thresholds = np.full(length, np.nan)
        new_vars = np.full(length, -2)
        new_leaf_vals = np.full(length,np.nan)
        new_n = np.full(length,-2)
        new_node_indicators = np.full((tree1.data.X.shape[0],length), 0 ,dtype=bool)
        # give the corresponding value to new tree
        new_thresholds[:len(tree1.thresholds)]=tree1.thresholds
        new_vars[:len(tree1.vars)]=tree1.vars
        new_leaf_vals[:len(tree1.leaf_vals)] = tree1.leaf_vals
        new_n[:len(tree1.n)]=tree1.n
        new_node_indicators[:, :tree1.node_indicators.shape[1]] = tree1.node_indicators
        new_evals = tree1.evals
        tree_new = Tree(data=tree1.data, thresholds=new_thresholds, vars=new_vars,
                        leaf_vals=new_leaf_vals, n=new_n,
                        node_indicators=new_node_indicators,evals=new_evals)
        
        return tree_new  

    def update_tree(self, tree, start_index, new_vars, new_thresholds):
        #set the corresponding value of tree A to tree B
        queue = [start_index]  
        var_index = 0 

        while queue and var_index < len(new_vars):
            current_index = queue.pop(0)  
            if current_index < len(tree.thresholds):  
            
                tree.vars[current_index] = new_vars[var_index]
                tree.thresholds[current_index] = new_thresholds[var_index]

                left_index = 2 * current_index + 1
                right_index = 2 * current_index + 2
                # if left_index >= len(tree.vars) or right_index >= len(tree.vars):
                #     tree._resize_arrays()
                queue.append(left_index)
                queue.append(right_index)

                var_index += 1
        
        return tree
    

    def try_propose(self, proposed, generator):
        # find node and collect thresholds
        tree1 = proposed.trees[self.trees_changed[0]]
        tree2 = proposed.trees[self.trees_changed[1]]
        node_id = generator.choice(tree1.leaves)
        combined_tree = self.create_combined(tree1,tree2)
        updated_tree = self.update_tree(combined_tree,node_id,tree2.vars,tree2.thresholds)
        
        success = updated_tree.update_n(node_id)
        if success:
            #proposed.trees.insert(self.trees_changed[0],updated_tree)
            proposed.trees.append(updated_tree)
            proposed.trees.remove(tree1)
            proposed.trees.remove(tree2)
     
        return success
 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap}
