
import numpy as np
from typing import Optional
from numpy.typing import NDArray
from numba import njit

@njit
def _compute_leaf_basis(node_indicators, vars):
    """
    Numba-optimized function to compute leaf basis matrix.
    """
    # Extract columns corresponding to leaf nodes
    return node_indicators[:, (vars == -1)]

@njit
def _update_n_and_indicators_numba(starting_node, dataX, append: bool, vars, thresholds, prev_n, prev_node_indicators):
    """
    Numba-optimized function to update all node counts and indicators using DFS.
    """
    n_nodes = len(vars)

    # Modify in place to avoid copying
    n = prev_n
    node_indicators = prev_node_indicators
    # If appending, we need to take the offset
    #   into account when accessing node_indicators
    offset = prev_node_indicators.shape[0] - dataX.shape[0] if append else 0

    # Use a simple array as a stack for depth-first search
    # Avoid recursion to prevent stack overflow on large trees
    stack = np.zeros(n_nodes, dtype=np.int64)
    top = 0
    stack[top] = starting_node
    top += 1
    
    success = True

    while top > 0:
        # Pop the top node
        top -= 1
        node_id = stack[top]

        # If it is a split node, propagate indicators and counts to children
        if vars[node_id] > -1:
            current_var = vars[node_id]
            current_threshold = thresholds[node_id]
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2

            parent_indicators = node_indicators[:, node_id]
            left_count = 0
            right_count = 0
            
            for i in range(dataX.shape[0]):
                if parent_indicators[offset + i]:
                    go_left = (dataX[i, current_var] <= current_threshold)
                    node_indicators[offset + i, left_child] = go_left
                    left_count += go_left
                    node_indicators[offset + i, right_child] = not go_left
                    right_count += not go_left
                else:
                    node_indicators[offset + i, left_child] = False
                    node_indicators[offset + i, right_child] = False
                    
            if append:
                n[left_child] += left_count
                n[right_child] += right_count
            else:
                n[left_child] = left_count
                n[right_child] = right_count

            # Push children onto stack
            stack[top] = right_child
            top += 1
            stack[top] = left_child
            top += 1
        else:
            # If it is a leaf node, set success to False if it has no samples
            if n[node_id] == 0:
                success = False
    
    # success = True if all updated n are > 0 else False
    return success

class Tree:
    """
    Represents the parameters of a single tree in the BART model, combining both
    the tree structure and leaf values into a single object.
    """
    def __init__(self, dataX: Optional[np.ndarray], vars : np.ndarray, thresholds : np.ndarray, leaf_vals : np.ndarray,
                  n, node_indicators, evals):
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
        self.dataX: Optional[NDArray[np.float32]] = dataX
        self.vars: NDArray[np.int32] = vars
        self.thresholds: NDArray[np.float32] = thresholds
        self.leaf_vals: NDArray[np.float32] = leaf_vals

        self.n: NDArray[np.int32] = n
        self.node_indicators: NDArray[np.bool_] = node_indicators
        self.evals: NDArray[np.float32] = evals

    default_size: int = 8  # Default size for the tree arrays
    
    def _init_caching_arrays(self):
        """
        Initialize caching arrays for the tree.
        """
        assert self.dataX is not None, "Data matrix is not provided."
        self.node_indicators = np.full((self.dataX.shape[0], Tree.default_size), False, dtype=bool)
        self.node_indicators[:, 0] = True
        self.n = np.full(Tree.default_size, -2, dtype=int)
        self.n[0] = self.dataX.shape[0]
        self.evals = np.zeros(self.dataX.shape[0], dtype=np.float32)
        
    @property
    def cache_exists(self):
        return self.evals is not None
    
    @classmethod
    def new(cls, dataX=None):
        # Define the basic tree parameters.
        vars = np.full(Tree.default_size, -2, dtype=int)  # -2 represents an inexistent node
        vars[0] = -1                      # -1 represents a leaf node
        thresholds = np.full(Tree.default_size, np.nan, dtype=np.float32)
        leaf_vals = np.full(Tree.default_size, np.nan, dtype=np.float32)
        leaf_vals[0] = 0                   # Initialize the leaf value

        new_tree = cls(
            dataX.astype(np.float32, copy=False) if dataX is not None else None,
            vars, thresholds, leaf_vals, n=None, node_indicators=None, evals=None
        )
        if dataX is not None:
            # If dataX is provided, initialize caching arrays.
            new_tree._init_caching_arrays()
        
        return new_tree

    @classmethod
    def from_existing(cls, other: "Tree", copy_cache: bool = True):
        """
        Create a new Tree object by copying data from an existing Tree.
        """
        # Copy all arrays
        if other.n is None:
            copy_cache = False  # If n is None, we cannot copy cache arrays
        
        return cls(
            other.dataX,  # dataX is not copied since it's shared across trees
            other.vars.copy(),
            other.thresholds.copy(), 
            other.leaf_vals.copy(),
            other.n.copy() if copy_cache else None,
            other.node_indicators.copy() if copy_cache else None,
            other.evals.copy() if copy_cache else None
        )

    def copy(self, copy_cache: bool = True) -> "Tree":
        return Tree.from_existing(self, copy_cache=copy_cache)

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
        if len(self.split_nodes) == 0: # Tree has no splits
            return node_ids
        routing = X[:, self.vars[self.split_nodes]] > self.thresholds[self.split_nodes]
        split_node_counter = 0
        for k in range(len(self.vars)):
            if self.is_split_node(k):
                node_ids[node_ids == k] = node_ids[node_ids == k] * 2 + \
                    1 + routing[node_ids == k, split_node_counter]
                split_node_counter += 1
        return node_ids

    def evaluate(self, X: Optional[np.ndarray]=None) -> NDArray[np.float32]:
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
        old_size = len(self.vars)
        new_size = old_size * 2

        # -------- vars (int) --------
        # Alloc uninitialized
        a = np.empty(new_size, dtype=self.vars.dtype)
        # copy old data
        a[:old_size] = self.vars
        # init only the new part
        a[old_size:] = -2
        self.vars = a

        # ------- thresholds (float) -------
        b = np.empty(new_size, dtype=self.thresholds.dtype)
        b[:old_size] = self.thresholds
        b[old_size:] = np.nan
        self.thresholds = b

        # ------- leaf_vals (float) -------
        c = np.empty(new_size, dtype=self.leaf_vals.dtype)
        c[:old_size] = self.leaf_vals
        c[old_size:] = np.nan
        self.leaf_vals = c

        if self.cache_exists:
            # ------- n (int) -------
            d = np.empty(new_size, dtype=self.n.dtype)
            d[:old_size] = self.n
            d[old_size:] = -2
            self.n = d

            # -- node_indicators (bool 2-D) --
            rows = self.node_indicators.shape[0]
            e = np.empty((rows, new_size), dtype=bool)
            e[:, :old_size] = self.node_indicators
            e[:, old_size:] = False
            self.node_indicators = e
    
    def _truncate_tree_arrays(self):
        """
        Recursively trims the tree arrays to remove unnecessary space.

        This method ensures that if the last occurrence of -1 in `self.vars` 
        appears too early (less than half of the array length), the arrays 
        are truncated to half their size. The process continues until the 
        last -1 index is at least in the second half of the array. 

        A minimum size of Tree.default_size is enforced to prevent excessive truncation 
        that could lead to an invalid tree structure.
        """
        last_active_node = np.where(self.vars == -1)[0].max()
        new_length = len(self.vars)
        while last_active_node < (new_length // 2) and new_length > Tree.default_size:
            new_length //= 2
            
        if new_length < len(self.vars):    
            self.thresholds = self.thresholds[:new_length]
            self.vars = self.vars[:new_length]
            self.leaf_vals = self.leaf_vals[:new_length]
            self.n = self.n[:new_length]
            self.node_indicators = self.node_indicators[:, :new_length]

    def split_leaf(self, node_id: int, var: int, threshold: np.float32, left_val: np.float32=np.float32(np.nan), 
                   right_val: np.float32 = np.float32(np.nan)):
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
        
        # Assert cache arrays exist
        is_valid = _update_n_and_indicators_numba(
            node_id, self.dataX, False, self.vars, self.thresholds, self.n, self.node_indicators
        )

        return is_valid

    def prune_split(self, node_id: int, recursive = False):
        """
        Prune a terminal split node, turning it back into a leaf.

        Parameters:
        - node_id: int
            Index of the split node to prune.
        - recursive: bool, default=False
        If True, recursively prune all descendant split nodes.
        """

        # If recursive is False, ensure the node is a terminal split node
        if not recursive and not self.is_terminal_split_node(node_id):
            raise ValueError("Node is not a terminal split node and cannot be pruned (recursive=False).")

        # Check if the node is a split node
        if not self.is_split_node(node_id):
            raise ValueError("Node is not a split node and cannot be pruned.")

        # Use a stack to manage nodes to prune
        stack = [node_id]

        # Store the original node .n
        ori_n = self.n[node_id]

        while stack:
            current_node = stack.pop()

            # Check if the current node is a terminal split node
            if self.is_terminal_split_node(current_node):
                # If terminal, directly prune its children
                left_child = current_node * 2 + 1
                right_child = current_node * 2 + 2
                self.vars[left_child] = -2
                self.vars[right_child] = -2
                self.leaf_vals[left_child] = np.nan
                self.leaf_vals[right_child] = np.nan

                if self.cache_exists:
                    self.n[left_child] = -2
                    self.n[right_child] = -2
            elif not self.is_leaf(current_node):
                # If not terminal or leaf, add children to the stack for further pruning
                left_child = current_node * 2 + 1
                right_child = current_node * 2 + 2

                if left_child < len(self.vars):
                    stack.append(left_child)
                if right_child < len(self.vars):
                    stack.append(right_child)

            # After processing children, mark the current node as -2
            self.vars[current_node] = -2
            self.thresholds[current_node] = np.nan
            self.leaf_vals[current_node] = np.nan
            self.n[current_node] = -2

        # Finally, turn the original node into a leaf and set the n correctly
        self.vars[node_id] = -1
        self.n[node_id] = ori_n

        # Truncate unnecessary space in the tree arrays
        self._truncate_tree_arrays()

    def change_split(self, node_id, var, threshold, update_n=True):
        self.vars[node_id] = var
        self.thresholds[node_id] = threshold
        if update_n:
            return self.update_n(node_id)
        return True
    
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
        is_valid = _update_n_and_indicators_numba(
            node_id, self.dataX, False, self.vars, self.thresholds, self.n, self.node_indicators
        )

        return is_valid
    
    def update_n_append(self, X_new):
        """
        Update the counts of samples reaching each node in the decision tree when appending new data.

        Parameters:
        - X_new (np.ndarray): The new data to append to the existing dataset.
        Returns:
        """
        _update_n_and_indicators_numba(
            0, X_new, True, self.vars, self.thresholds, self.n, self.node_indicators
        )

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
        return np.where(self.vars == -1)[0]

    @property
    def n_leaves(self):
        return np.count_nonzero(self.vars == -1)
    
    @property
    def split_nodes(self):
        return np.where((self.vars != -1) & (self.vars != -2))[0]
    
    @property
    def terminal_split_nodes(self):
        return [i for i in range(len(self.vars)) if self.is_terminal_split_node(i)]
    
    @property
    def nonterminal_split_nodes(self):
        return [i for i in range(len(self.vars)) if self.is_split_node(i) 
                and not self.is_terminal_split_node(i)]

    @property
    def leaf_basis(self) -> NDArray[np.bool_]:
        return self.node_indicators[:, self.vars == -1]
        # return _compute_leaf_basis(self.node_indicators, self.vars)

    def __str__(self):
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
        if self.cache_exists:
            n_output = self.n[node_id]
        else:
            n_output = "NA"
        if self.vars[node_id] == -1:
            return f"Val: {self.leaf_vals[node_id]:0.9f} (leaf, n = {n_output})"
        else:
            return f"X_{self.vars[node_id]} <= {self.thresholds[node_id]:0.9f}" + \
                f" (split, n = {n_output})"

    def update_data(self, dataX: np.ndarray) -> None:
        """
        Replace the stored feature matrix, extend node-indicator arrays for any newly
        appended rows, and refresh per-node counts & cached outputs.

        Parameters
        ----------
        dataX : np.ndarray, shape (n_samples, n_features)
            The complete feature matrix (old + new samples).
        """
        old_n = 0 if self.dataX is None else self.dataX.shape[0]
        new_n = dataX.shape[0]
        n_new = new_n - old_n

        self.dataX = dataX

        # Update node-indicators
        if old_n == 0:
            # first time: init
            self._init_caching_arrays() 
        else:
            # extending existing indicators
            cols = self.node_indicators.shape[1]
            new_inds = np.zeros((n_new, cols), dtype=bool)
            new_inds[:, 0] = True         # i.e. all new samples start at root
            self.node_indicators = np.vstack([self.node_indicators, new_inds])
            self.n[0] += n_new  # Update the count of samples at the root node
            
        # Refresh counts & outputs only for the affected rows
        update_range = np.arange(old_n, new_n)
        self.update_n_append(dataX[update_range])
        self.update_outputs()
        
class Parameters:
    """
    Represents the parameters of the BART model.
    """
    def __init__(self, trees: list, global_params, cache : Optional[float]=None):
        """
        Initializes the parameters for the model.

        Parameters:
        - trees (list): A list of trees used in the model.
        - global_params (dict): Global parameters for the model.
        - cache (float, optional): Cached evaluation results for the trees.

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
    
    def clear_cache(self):
        self.cache = None
        for tree in self.trees:
            tree.evals = None
            tree.node_indicators = None
            tree.n = None
            
    def copy(self, modified_tree_ids=None, copy_cache=True):
        if modified_tree_ids is None:
            modified_tree_ids = range(self.n_trees)
        copied_trees = self.trees.copy() # Shallow copy
        for tree_id in modified_tree_ids:
            copied_trees[tree_id] = self.trees[tree_id].copy(copy_cache)
        # No need to deep copy global_params and cache
        # because they only contain numerical values (which are immutable)
        return Parameters(trees=copied_trees, 
                          global_params=self.global_params.copy(), # shallow copy suffices
                          cache=self.cache)
    
    def update_data(self, X_new):
        """
        Sets new data points for the model, replacing the existing data in all trees and re-calculating the cache.

        Parameters:
            X_new: New feature data to set (np.ndarray), including both old and new samples.
        """
        new_trees = self.trees.copy()  # Shallow copy the tree list
        for tree in new_trees:
            # Efficiently add new data points
            tree.update_data(X_new)
        
        # Create new parameters object with the updated trees and same global parameters
        new_state = Parameters(
            trees=new_trees, 
            global_params=self.global_params.copy(),
            cache=None  # Let Parameters initialize the cache, could be improved
        )
        
        return new_state

    def evaluate(self, X: Optional[np.ndarray]=None, tree_ids:Optional[list[int]]=None, all_except:Optional[list[int]]=None) -> NDArray[np.float32]:
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
        if tree_ids is None and all_except is None:
            tree_ids = list(np.arange(self.n_trees))
            all_except = []

        if X is None:
            total_output = self.cache.copy()
            if all_except is None:
                all_except = [i for i in np.arange(self.n_trees) if i not in tree_ids]
            for i in all_except:
                total_output -= self.trees[i].evals
        else:
            total_output = np.zeros(X.shape[0])
            if tree_ids is None:
                tree_ids = [i for i in np.arange(self.n_trees) if i not in all_except]
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
        if len(tree_ids) == 1:
            return self.trees[tree_ids[0]].leaf_basis
        return np.hstack([self.trees[tree_id].leaf_basis for tree_id in tree_ids])

    def update_leaf_vals(self, tree_ids : list[int], leaf_vals : NDArray[np.float32]):
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
            leaves = tree.leaves
            n_leaves = len(leaves)
            
            tree.leaf_vals[leaves] = \
                leaf_vals[range(leaf_counter, leaf_counter + n_leaves)]
            tree.update_outputs()
            self.cache = self.cache + tree.evals - tree_evals_old
            leaf_counter += n_leaves
