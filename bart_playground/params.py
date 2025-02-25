import numpy as np
import copy
from typing import Optional
from numpy.typing import NDArray
from .util import Dataset

class Tree:
    """
    Represents the parameters of a single tree in the BART model, combining both
    the tree structure and leaf values into a single object.
    """
    def __init__(self, dataX: Optional[np.ndarray], vars, thresholds, leaf_vals, n, node_indicators, evals):
        """
        Initialize the tree parameters.

        Parameters:
        - dataX : np.ndarray, optional
            The np.ndarray object containing the X (covariates) of data.
        - vars : np.ndarray, optional
            Array of variables used for splitting at each node. Default is None.
        - thresholds : np.ndarray, optional
            Array of split values at each node. Default is None.
        - leaf_vals : np.ndarray, optional
            Values at the leaf nodes. Default is None.
        - n : np.ndarray, optional
            Array representing the number of data points at each node. Default is None.
        - node_indicators : np.ndarray, optional
            Boolean array indicating which data examples lie within each node. Default is None.
        """
        self.dataX = dataX
        self.vars = vars
        self.thresholds = thresholds
        self.leaf_vals : NDArray[np.float_] = leaf_vals

        self.n = n
        self.node_indicators = node_indicators
        self.evals = evals

    @classmethod
    def new(cls, dataX=None):
        # Define the basic tree parameters.
        vars = np.full(8, -2, dtype=int)  # -2 represents an inexistent node
        vars[0] = -1                      # -1 represents a leaf node
        thresholds = np.full(8, np.nan, dtype=float)
        leaf_vals = np.full(8, np.nan, dtype=float)
        leaf_vals[0] = 0                   # Initialize the leaf value

        if dataX is not None:
            # If dataX is provided, initialize caching arrays.
            node_indicators = np.full((dataX.shape[0], 8), False, dtype=bool)
            node_indicators[:, 0] = True   # All observations go to the root
            n = np.full(8, -2, dtype=int)
            n[0] = dataX.shape[0]
            evals = np.zeros(dataX.shape[0])
        else:
            # Otherwise, skip caching.
            node_indicators = None
            n = None
            evals = None

        return cls(dataX, vars, thresholds, leaf_vals, n, node_indicators, evals)

    @classmethod
    def from_existing(cls, other: "Tree"):
        return cls(
            other.dataX,
            copy.deepcopy(other.vars),
            copy.deepcopy(other.thresholds),
            copy.deepcopy(other.leaf_vals),
            copy.deepcopy(other.n),
            copy.deepcopy(other.node_indicators),
            copy.deepcopy(other.evals)
        )

    def copy(self):
        return Tree.from_existing(self)

    def traverse_tree(self, X: np.ndarray) -> np.ndarray:
        """
        Traverse the tree to find the leaf nodes for a given input data matrix.

        Parameters:
        - X: np.ndarray
            Input data (2D array).

        Returns:
        - int
            Index of the leaf node.
        """
        node_ids = np.full(X.shape[0], 0, dtype=int)
        if not self.split_nodes: # Tree has no splits
            return node_ids
        routing = X[:, self.vars[self.split_nodes]] > self.thresholds[self.split_nodes]
        split_node_counter = 0
        for k in range(len(self.vars)):
            if self.is_split_node(k):
                node_ids[node_ids == k] = node_ids[node_ids == k] * 2 + \
                    1 + routing[node_ids == k, split_node_counter]
                split_node_counter += 1
        return node_ids

    def evaluate(self, X: Optional[np.ndarray]=None) -> NDArray[np.float_]:
        """
        Evaluate the tree for a given input data matrix.


        Parameters:
        - x: np.ndarray
            Input data (2D array).

        Returns:
        - float
            Output values of the tree.
        """
        if X is None:
            if self.evals is not None:
                return self.evals
            else:
                raise ValueError("No cached data available for evaluation.")
        else:
            leaf_ids = self.traverse_tree(X)  # Find the leaf node for the input
            return self.leaf_vals[leaf_ids]  # Return the value at the leaf node

    def _resize_arrays(self):
        """
        Resize the internal arrays of the class by doubling their length.

        This method resizes the following arrays:
        - `vars`: An array of integers, resized to double its current length, with new elements initialized to -2.
        - `thresholds`: An array of floats, resized to double its current length, with new elements initialized to NaN.
        - `leaf_vals`: An array of floats, resized to double its current length, with new elements initialized to NaN.
        - `n`: An array of integers, resized to double its current length, with new elements initialized to -2.
        - `node_indicators`: A 2D boolean array, resized to double its current length along the second dimension, with new elements initialized to 0.

        The existing elements of each array are preserved, and the new elements are initialized as specified.
        """
        old_length = len(self.vars)
        new_length = old_length * 2

        # Resize vars array
        new_vars = np.full(new_length, -2, dtype=int)
        new_vars[:len(self.vars)] = self.vars
        self.vars = new_vars

        # Resize split array
        new_thresholds = np.full(new_length, np.nan, dtype=float)
        new_thresholds[:len(self.thresholds)] = self.thresholds
        self.thresholds = new_thresholds

        # Resize leaf_vals array
        new_leaf_vals = np.full(new_length, np.nan, dtype=float)
        new_leaf_vals[:len(self.leaf_vals)] = self.leaf_vals
        self.leaf_vals = new_leaf_vals

        if self.dataX is not None:
            # Resize n_vals array
            new_n = np.full(new_length, -2, dtype=int)
            new_n[:len(self.n)] = self.n
            self.n = new_n

            # Resize node_indicators array
            new_node_indicators = np.full((self.node_indicators.shape[0], new_length), 0, dtype=bool)
            # Copy the existing node_indicators into the first part of new_node_indicators
            new_node_indicators[:, :old_length] = self.node_indicators
            self.node_indicators = new_node_indicators

    def split_leaf(self, node_id: int, var: int, threshold: float, left_val: float=np.nan, 
                   right_val: float=np.nan):
        """
        Split a leaf node into two child nodes.

        Parameters:
        - node_id: int
        - threshold: float
        - left_val: float, optional
            Value to assign to the left child node (default is np.nan).
        - right_val: float, optional
            Value to assign to the right child node (default is np.nan).
        Returns:
        - bool
            True if both child nodes have more than 0 samples, False otherwise.
        Raises:
        - ValueError
            If the node is not a leaf and cannot be split.
        """
        # Check if the node is already a leaf
        if self.vars[node_id] != -1:
            raise ValueError("Node is not a leaf and cannot be split.")

        # Check if the index overflows and resize arrays if necessary
        left_child = node_id * 2 + 1
        right_child = node_id * 2 + 2

        if left_child >= len(self.vars) or right_child >= len(self.vars):
            self._resize_arrays()

        # Assign the split variable and threshold to the leaf node
        self.vars[node_id] = var
        self.thresholds[node_id] = threshold

        # Initialize the new leaf nodes
        self.vars[left_child] = -1
        self.vars[right_child] = -1

        # Assign the provided values to the new leaf nodes

        self.leaf_vals[node_id] = np.nan
        self.leaf_vals[left_child] = left_val
        self.leaf_vals[right_child] = right_val

        if self.dataX is not None:
            # Update the node indicators and counts
            x_bigger = self.dataX[:, var] > threshold
            self.node_indicators[:, left_child] = self.node_indicators[:, node_id] & ~x_bigger
            self.node_indicators[:, right_child] = self.node_indicators[:, node_id] & x_bigger
            # self.node_indicators[:, left_child] = self.node_indicators[:, node_id] & self.data.X[:, var] <= threshold
            # self.node_indicators[:, right_child] = self.node_indicators[:, node_id] & self.data.X[:, var] > threshold
            self.n[left_child] = np.sum(self.node_indicators[:, left_child])
            self.n[right_child] = np.sum(self.node_indicators[:, right_child])
            is_valid = self.n[left_child] > 0 and self.n[right_child] > 0
        else:
            is_valid = True

        return is_valid

    def prune_split(self, node_id: int):
        """
        Prune a terminal split node, turning it back into a leaf.

        Parameters:
        - node_id: int
            Index of the split node to prune.
        """
        # Check if the node is a split node
        if not self.is_terminal_split_node(node_id):
            raise ValueError("Node is not a terminal split node and cannot be pruned.")

        # Turn the split node into a leaf
        self.vars[node_id] = -1
        self.thresholds[node_id] = np.nan

        # Delete previous leaf nodes
        left_child = node_id * 2 + 1
        right_child = node_id * 2 + 2
        self.vars[left_child] = -2
        self.vars[right_child] = -2
        self.leaf_vals[left_child] = np.nan
        self.leaf_vals[right_child] = np.nan

        if self.dataX is not None:
            self.n[left_child] = -2
            self.n[right_child] = -2

    def change_split(self, node_id, var, threshold, update_n=True):
        self.vars[node_id] = var
        self.thresholds[node_id] = threshold
        if update_n:
            is_valid = self.update_n(node_id)
            return is_valid
    
    def swap_split(self, parent_id, child_id):
        parent_var, parent_threshold = self.vars[parent_id], self.thresholds[parent_id]
        child_var, child_threshold = self.vars[child_id], self.thresholds[child_id]
        self.change_split(child_id, parent_var, parent_threshold, update_n=False)
        is_valid = self.change_split(parent_id, child_var, child_threshold, update_n=True)
        return is_valid
    
    def update_n(self, node_id=0):
        """
        Updates the counts of samples reaching each node in the decision tree.

        This method recursively updates the counts of samples (`self.n`) for each node in the decision tree,
        starting from the specified `node_id`. If the node is a leaf, it checks if the count of samples
        reaching that node is greater than 0. If the node is not a leaf, it updates the counts for its
        left and right children based on the splitting criterion defined by the variable and threshold
        at the current node.

        Parameters:
        - node_id (int, optional): The ID of the node to start updating from. Defaults to 0 (the root node).

        Returns:
        - bool: True if the counts of samples reaching all nodes are greater than 0, False otherwise.
        """
        if self.dataX is None:
            raise ValueError("Data matrix is not provided.")
        
        if self.is_leaf(node_id):
            return self.n[node_id] > 0
        else:
            var = self.vars[node_id]
            threshold = self.thresholds[node_id]
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            self.node_indicators[:, left_child] = self.node_indicators[:, node_id] & (self.dataX[:, var] <= threshold)
            self.node_indicators[:, right_child] = self.node_indicators[:, node_id] & (self.dataX[:, var] > threshold)
            self.n[left_child] = np.sum(self.node_indicators[:, left_child])
            self.n[right_child] = np.sum(self.node_indicators[:, right_child])
            is_valid = self.update_n(left_child) and self.update_n(right_child)
            return is_valid
        
    def update_outputs(self):
        self.evals = self.leaf_basis @ self.leaf_vals[self.leaves]

    def set_leaf_value(self, node_id, leaf_val):
        if self.is_leaf(node_id):
            self.leaf_vals[node_id] = leaf_val
        else:
            raise ValueError("Not a leaf")

    def is_leaf(self, node_id):
        return self.vars[node_id] == -1

    def is_split_node(self, node_id):
        return self.vars[node_id] not in [-1, -2]
    
    def is_terminal_split_node(self, node_id):
        return self.is_split_node(node_id) \
            and self.is_leaf(node_id * 2 + 1) \
            and self.is_leaf(node_id * 2 + 2)

    @property
    def leaves(self):
        return [i for i in range(len(self.vars)) if self.is_leaf(i)]

    @property
    def n_leaves(self):
        return len(self.leaves)
    
    @property
    def split_nodes(self):
        return [i for i in range(len(self.vars)) if self.is_split_node(i)]
    
    @property
    def terminal_split_nodes(self):
        return [i for i in range(len(self.vars)) if self.is_terminal_split_node(i)]
    
    @property
    def nonterminal_split_nodes(self):
        return [i for i in range(len(self.vars)) if self.is_split_node(i) 
                and not self.is_terminal_split_node(i)]

    @property
    def leaf_basis(self) -> NDArray[np.bool_]:
        if self.dataX is None:
            raise ValueError("Data matrix is not provided.")
        
        return self.node_indicators[:, self.leaves]
    
    def __str__(self):
        """
        Return a string representation of the TreeParams object.
        """
        return self._print_tree()
        
    def __repr__(self):
        if self.dataX is None:
            return f"Tree(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals})"
        else:
            return f"Tree(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals}, n_vals={self.n})"

    def _print_tree(self, node_id=0, prefix=""):
        pprefix = prefix + "\t"
        if self.vars[node_id] == -1: # Leaf node
            return prefix + self._print_node(node_id)
        else:
            left_idx = node_id * 2 + 1
            right_idx = node_id * 2 + 2
            return (
                prefix
                + self._print_node(node_id)
                + "\n"
                + self._print_tree(left_idx, pprefix)
                + "\n"
                + self._print_tree(right_idx, pprefix)
            )
        
    def _print_node(self, node_id):
        if self.dataX is not None:
            n_output = self.n[node_id]
        else:
            n_output = "NA"
        if self.vars[node_id] == -1:
            return f"Val: {self.leaf_vals[node_id]:0.3f} (leaf, n = {n_output})"
        else:
            return f"X_{self.vars[node_id]} <= {self.thresholds[node_id]:0.3f}" + \
                f" (split, n = {n_output})"

class Parameters:
    """
    Represents the parameters of the BART model.
    """
    def __init__(self, trees: list, global_params, cache=None):
        """
        Initializes the parameters for the model.

        Parameters:
        - trees (list): A list of trees used in the model.
        - global_params (dict): Global parameters for the model.
        - data (Dataset): The dataset to be used.

        Attributes:
        - data (Dataset): The dataset to be used.
        - trees (list): A list of trees used in the model.
        - n_trees (int): The number of trees in the model.
        - global_params (dict): Global parameters for the model.
        """
        self.trees = trees
        self.n_trees = len(self.trees)
        self.global_params = global_params
        self.init_cache(cache)
            
    def init_cache(self, cache):
        if cache is None:
            self.cache = np.sum([tree.evaluate() for tree in self.trees], axis=0)
        else:
            self.cache = cache
            
    def copy(self, modified_tree_ids=None):
        if modified_tree_ids is None:
            modified_tree_ids = range(self.n_trees)
        copied_trees = self.trees.copy() # Shallow copy
        for tree_id in modified_tree_ids:
            copied_trees[tree_id] = self.trees[tree_id].copy()
        return Parameters(trees=copied_trees, global_params=copy.deepcopy(self.global_params), cache=copy.deepcopy(self.cache))

    # def copy(self, modified_tree_ids):
        # return copy.deepcopy(self)

    def evaluate(self, X: Optional[np.ndarray]=None, tree_ids:Optional[list[int]]=None, all_except:Optional[list[int]]=None) -> NDArray[np.float_]:
        """
        Evaluate the model on the given data.

        Parameters:
        -----------
        X : np.ndarray, optional
            The input data to evaluate. If None, the model's internal data will be used.
        tree_ids : list of int, optional
            Specific tree indices to evaluate. If provided, only these trees will be used.
        all_except : list of int, optional
            Tree indices to exclude from evaluation. If provided, all trees except these will be used.

        Returns:
        --------
        np.ndarray
            The total output of the evaluated trees on the input data.
        """

        # Trees to evaluate on
        if tree_ids is not None:
            all_except = [i for i in np.arange(self.n_trees) if i not in tree_ids]
        elif all_except is not None:
            tree_ids = [i for i in np.arange(self.n_trees) if i not in all_except]
        else:
            tree_ids = list(np.arange(self.n_trees))
            all_except = []

        if X is None:
            total_output = self.cache.copy()
            for i in all_except:
                total_output -= self.trees[i].evals
        else:
            total_output = np.zeros(X.shape[0])
            # Iterate over all trees
            for i in tree_ids:
                total_output += self.trees[i].evaluate(X)  # Add the tree's output to the total
        return total_output

    def leaf_basis(self, tree_ids):
        """
        Generate a horizontal stack of leaf basis arrays for the specified tree IDs.

        Parameters:
        - tree_ids (list of int): List of tree IDs for which to generate the leaf basis.

        Returns:
        - numpy.ndarray: A horizontally stacked array of leaf basis arrays corresponding to the given tree IDs.
        """
        return np.hstack([self.trees[tree_id].leaf_basis for tree_id in tree_ids])

    def update_leaf_vals(self, tree_ids : list[int], leaf_vals : NDArray[np.float_]):
        """
        Update the leaf values of specified trees.

        Parameters:
        - tree_ids (list of int): List of tree IDs whose leaf values need to be updated.
        - leaf_vals (list of float): List of new leaf values to be assigned to the trees.

        Returns:
        - None
        """
        leaf_counter = 0
        for tree_id in tree_ids:
            tree = self.trees[tree_id]
            tree_evals_old = tree.evals
            tree.leaf_vals[tree.leaves] = \
                leaf_vals[range(leaf_counter, leaf_counter + tree.n_leaves)]
            tree.update_outputs()
            self.cache = self.cache + tree.evals - tree_evals_old
            leaf_counter += tree.n_leaves
