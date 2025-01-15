import numpy as np
from abc import ABC, abstractmethod
import math

from src.params import Parameters
from src.params import Tree

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
    
class Break(Move):
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
        local_tree.thresholds[index] = np.nan
    
        left_index = 2 * index + 1
        local_tree = Break.set_subtree_zero(local_tree, left_index)
    
        right_index = 2 * index + 2
        local_tree = Break.set_subtree_zero(local_tree, right_index)
        return local_tree

    def propose(self, generator):
        for _ in range(self.tol):
            # find node and collect thresholds
            self.proposed = self.current.copy(self.trees_changed)
            tree = self.proposed.trees[self.trees_changed[0]]
            node_id = generator.choice(tree.split_nodes)
            thresholds,vars = Break.collect_values(tree,node_id)
            # update thresholds and vars
            tree = Break.set_subtree_zero(tree,node_id)
            tree.vars[node_id] = -1
        
            new_thresholds = np.full(len(thresholds), np.nan)
            new_vars = np.full(len(vars), -2)
            new_leaf_vals = np.fall(len(vars),np.nan)
            # give the corresponding value to new tree
            new_thresholds[:len(thresholds)]=thresholds
            new_vars[:len(vars)]=vars
            # generate new tree 
            tree_new = Tree(data=tree.data, thresholds=new_thresholds, vars=new_vars,
                        leaf_vals=new_leaf_vals, n=np.fall(len(new_thresholds),-2),
                        node_indicators=np.full((tree.data.X.shape[0], len(new_thresholds)), 0, dtype=bool))  
            self.proposed.trees.append(tree_new)
            if tree_new.split_leaf(node_id, vars, thresholds): # If no empty leaves are created
                return self.proposed
        self.proposed = self.current

        return self.proposed

class Combine(Move):
    """
    Move to combine two trees into one tree
    """
    # Assume that the input is defined in TreeParams Class
    def __init__(self, current : Parameters, trees_changed: np.ndarray, tol=100):
        super().__init__(current, trees_changed)
        assert len(trees_changed) == 2
        self.tol = tol

    def create_combined(tree1,tree2):
        d2 = round(math.log2(len(tree2.thresholds)+1))
        d1 = round(math.log2(len(tree1.thresholds)+1))
        new_thresholds = np.full(2**(d1+d2)-1, np.nan)
        new_vars = np.full(2**(d1+d2)-1, -2)
        new_leaf_vals = np.full(2**(d1+d2)-1,np.nan)
        new_n = np.full(2**(d1+d2)-1,-2)
        new_node_indicators = np.full((tree1.data.X.shape[0],2**(d1+d2)-1),0,dtype=bool)
        # give the corresponding value to new tree
        new_thresholds[:len(tree1.thresholds)]=tree1.thresholds
        new_vars[:len(tree1.vars)]=tree1.vars
        new_leaf_vals[:len(tree1.leaf_vals)] = tree1.leaf_vals
        new_n[:len(tree1.n)]=tree1.n
        new_node_indicators=tree1.node_indicators
        tree_new = Tree(data=tree1.data, thresholds=new_thresholds, vars=new_vars,
                        leaf_vals=new_leaf_vals, n=new_n,
                        node_indicators=new_node_indicators)
        return tree_new  

    
    def update_tree(tree, start_index, new_vars, new_thresholds):
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
                queue.append(left_index)
                queue.append(right_index)

                var_index += 1

        return tree

    def propose(self, generator):
        for _ in range(self.tol):
            # find node and collect thresholds
            self.proposed = self.current.copy(self.trees_changed)
            tree1 = self.proposed.trees[self.trees_changed[0]]
            tree2 = self.proposed.trees[self.trees_changed[1]]
            node_id = generator.choice(tree1.leaves)
            combined_tree = Combine.create_combined(tree1,tree2)
            updated_tree = Combine.update_tree(combined_tree,node_id,tree2.vars,tree2.thresholds)
            if updated_tree.update_n(node_id): 
                self.proposed.trees.remove(tree1)
                self.proposed.trees.remove(tree2)
                self.proposed.trees.append(updated_tree)
                return self.proposed
        self.proposed = self.current
        return self.proposed

all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "break": Break,
            "combine": Combine}