import numpy as np
import copy

from .util import Dataset

class Tree:
    """
    Represents the parameters of a single tree in the BART model, combining both
    the tree structure and leaf values into a single object.
    """
    def __init__(self, data : Dataset, vars=None, thresholds=None, leaf_vals=None, n=None, 
                 node_indicators=None):
        """
        Initialize the tree parameters.

        Parameters:
        - data : Dataset
            The dataset object containing the data.
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
        self.data = data
        if vars is None:
            self.vars = np.full(8, -2, dtype=int) # -2 represents an inexistant node
            self.vars[0] = -1 # -1 represents a leaf node
            self.thresholds = np.full(8, np.nan, dtype=float)
            self.leaf_vals = np.full(8, np.nan, dtype=float)
            self.node_indicators = np.full((data.X.shape[0], 8), 0, dtype=bool)
            self.node_indicators[:, 0] = True
            self.n = np.full(8, -2, dtype=int)
            self.n[0] = data.X.shape[0]
        else:
            self.vars = copy.deepcopy(vars)
            self.thresholds = copy.deepcopy(thresholds)
            self.leaf_vals = copy.deepcopy(leaf_vals)
            self.n = copy.deepcopy(n)
            self.node_indicators = copy.deepcopy(node_indicators)

    def copy(self):
        return Tree(self.data, self.vars, self.thresholds, self.leaf_vals, self.n, 
                    self.node_indicators)

    def traverse_tree(self, X: np.ndarray) -> int:
        """
        Traverse the tree to find the leaf nodes for a given input data matrix.

        Parameters:
        - X: np.ndarray
            Input data (2D array).

        Returns:
        - int
            Index of the leaf node.
        """
        n = X.shape[0]
        node_ids = np.zeros(n, dtype=int)
        for i in range(n):
            node_ids[i] = self._traverse_tree(X[i,:])
        return node_ids

    def _traverse_tree(self, x: np.ndarray) -> int:
        """
        Traverse the tree to find the leaf node for a given input.

        Parameters:
        - x: np.ndarray
            Input data point (1D array).

        Returns:
        - int
            Index of the leaf node.
        """
        node_id = 0  # Start at the root node

        while True:
            var = self.vars[node_id]
            threshold = self.thresholds[node_id]

            if var == -1:  # Check if the node is a leaf
                return node_id  # Return the leaf node index

            # Determine the next node based on the split condition
            if x[var] <= threshold:
                node_id = node_id * 2 + 1  # Move to the left child
            else:
                node_id = node_id * 2 + 2  # Move to the right child

    def evaluate(self, X: np.ndarray) -> float:
        """
        Evaluate the tree for a given input data matrix.


        Parameters:
        - x: np.ndarray
            Input data (2D array).

        Returns:
        - float
            Output values of the tree.
        """
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
        new_length = len(self.vars) * 2

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

        # Resize n_vals array
        new_n = np.full(new_length, -2, dtype=int)
        new_n[:len(self.n)] = self.n
        self.n = new_n

        # Resize node_indicators array
        new_node_indicators = np.full((self.node_indicators.shape[0], new_length), 0, dtype=bool)
        new_node_indicators[:len(self.node_indicators.shape[1])] = self.n
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
        x_bigger = self.data.X[:, var] > threshold
        self.node_indicators[:, left_child] = self.node_indicators[:, node_id] & ~x_bigger
        self.node_indicators[:, right_child] = self.node_indicators[:, node_id] & x_bigger
        # self.node_indicators[:, left_child] = self.node_indicators[:, node_id] & self.data.X[:, var] <= threshold
        # self.node_indicators[:, right_child] = self.node_indicators[:, node_id] & self.data.X[:, var] > threshold
        self.n[left_child] = np.sum(self.node_indicators[:, left_child])
        self.n[right_child] = np.sum(self.node_indicators[:, right_child])

        # Assign the provided values to the new leaf nodes
        self.leaf_vals[left_child] = left_val
        self.leaf_vals[right_child] = right_val
        
        return self.n[left_child] > 0 and self.n[right_child] > 0
    
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
        if self.is_leaf(node_id):
            return self.n[node_id] > 0
        else:
            var = self.vars[node_id]
            threshold = self.thresholds[node_id]
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            self.node_indicators[:, left_child] = self.node_indicators[:, node_id] & self.data.X[:, var] <= threshold
            self.node_indicators[:, right_child] = self.node_indicators[:, node_id] & self.data.X[:, var] > threshold
            self.n[left_child] = np.sum(self.node_indicators[:, left_child])
            self.n[right_child] = np.sum(self.node_indicators[:, right_child])
            return self.update_n(left_child) and self.update_n(right_child)

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
    def leaf_basis(self):
        return self.node_indicators[:, self.leaves]
    
    def __str__(self):
        """
        Return a string representation of the TreeParams object.
        """
        return self._print_tree()
        
    def __repr__(self):
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
        if self.vars[node_id] == -1:
            return f"Val: {self.leaf_vals[node_id]:0.3f} (leaf, n = {self.n[node_id]})"
        else:
            return f"X_{self.vars[node_id]} <= {self.thresholds[node_id]:0.3f}" + \
                f" (split, n = {self.n[node_id]})"

class Parameters:
    """
    Represents the parameters of the BART model.
    """
    def __init__(self, trees: list, global_params, data : Dataset):
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
        self.data = data
        self.trees = trees
        self.n_trees = len(self.trees)
        self.global_params = global_params

    def copy(self, modified_tree_ids):
        copied_trees = self.trees
        for tree_id in modified_tree_ids:
            copied_trees[tree_id] = self.trees[tree_id].copy()
        return Parameters(trees=copied_trees, global_params=copy.deepcopy(self.global_params), 
                          data=self.data)

    def evaluate(self, X: np.ndarray=None, tree_ids=None, all_except=None) -> float:
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
        float
            The total output of the evaluated trees on the input data.
        """

        if X is None:
            X = self.data.X
        if tree_ids is not None:
            pass
        elif all_except is not None:
            tree_ids = [i for i in np.arange(self.n_trees) if i not in tree_ids]
        else:
            tree_ids = np.arange(self.n_trees)
        total_output = np.zeros(X.shape[0])
        # Iterate over all trees
        for i, tree in enumerate(self.trees):
            if i in tree_ids:  # Skip trees in the holdout list
                total_output += tree.evaluate(X)  # Add the tree's output to the total
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

    def update_leaf_vals(self, tree_ids, leaf_vals):
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
            tree.leaf_vals[tree.leaves] = \
                leaf_vals[range(leaf_counter, leaf_counter + tree.n_leaves)]
            leaf_counter += tree.n_leaves
