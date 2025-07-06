import numpy as np
from typing import Optional
from numpy.typing import NDArray
from numba import njit

@njit
def _update_n_and_leaf_id_numba(starting_node, dataX, append: bool, vars, thresholds, prev_n, prev_leaf_id):
    """
    Numba-optimized function to update all node counts and leaf_ids.
    If append is True, dataX should be appended to the existing data.
    """
    n_nodes = len(vars)

    # Modify in place to avoid copying
    n = prev_n
    leaf_ids = prev_leaf_id
    # If appending, we need to take the offset into account when accessing leaf_ids
    offset = prev_leaf_id.shape[0] - dataX.shape[0] if append else 0
    
    subtree_nodes = np.empty(n_nodes, np.bool_)
    for j in range(n_nodes):
        if _is_in_subtree(j, starting_node):
            subtree_nodes[j] = True
            if not append:
                n[j] = 0 # starting node is not updated
        else:
            subtree_nodes[j] = False
    
    for i in range(dataX.shape[0]):
        current_node = leaf_ids[offset + i]
        # Need update
        if append or current_node == starting_node or subtree_nodes[current_node]:
            leaf_ids[offset + i] = _traverse_tree_single(
                dataX[i], vars, thresholds, starting_node, n
            )
            
    # success = True if all updated n are > 0 else False
    success = True
    for j in range(n_nodes):
        if vars[j] == -1 and subtree_nodes[j] and n[j] <= 0:
            success = False
            break

    return success

@njit(inline='always')
def _is_in_subtree(node_id, ancestor_id):
    # A node cannot be in the subtree of a node with a larger index
    # in this binary tree representation.
    # The ancestor node is not considered to be in its own subtree.
    if node_id <= ancestor_id:
        return False
    
    current_node = node_id
    while current_node > ancestor_id:
        current_node = (current_node - 1) // 2
    
    return current_node == ancestor_id

@njit
def _traverse_tree_numba(X: np.ndarray, vars: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Numba-optimized function to traverse the tree for a given input data matrix.

    Parameters:
    - X: np.ndarray
        Input data (2D array).
    - vars: np.ndarray
        Array of variables used for splitting at each node.
    - thresholds: np.ndarray
        Array of split values at each node.

    Returns:
    - np.ndarray
        An array of leaf node indices for each row in X.
    """
    n_samples = X.shape[0]
    node_ids = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        node_ids[i] = _traverse_tree_single(X[i], vars, thresholds, 0, None)
    
    return node_ids

@njit
def _traverse_tree_single(X: np.ndarray, vars: np.ndarray, thresholds: np.ndarray, starting_node, n_to_update):
    """
    Numba-optimized function to traverse the tree for a given input data array (1D).
    """
    current_node = starting_node
    while vars[current_node] >= 0:  # While it's a split node
        split_var = vars[current_node]
        threshold = thresholds[current_node]
        if X[split_var] <= threshold:
            current_node = 2 * current_node + 1
        else:
            current_node = 2 * current_node + 2
        if n_to_update is not None:
            n_to_update[current_node] += 1 # Increment count for the subtree node
    return current_node

class Tree:
    """
    Represents the parameters of a single tree in the BART model, combining both
    the tree structure and leaf values into a single object.
    """
    def __init__(self, dataX: Optional[np.ndarray], vars : np.ndarray, thresholds : np.ndarray, leaf_vals : np.ndarray,
                  n, leaf_ids, evals):
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
        - leaf_ids : np.ndarray, optional
            Array indicating which leaf each data example belongs to. Default is None.
        """
        self.dataX: Optional[NDArray[np.float32]] = dataX
        self.vars: NDArray[np.int32] = vars
        self.thresholds: NDArray[np.float32] = thresholds
        self.leaf_vals: NDArray[np.float32] = leaf_vals

        self.n: NDArray[np.int32] = n
        self.leaf_ids: NDArray[np.int16] = leaf_ids
        self.evals: NDArray[np.float32] = evals

    default_size: int = 8  # Default size for the tree arrays
    
    def _init_caching_arrays(self):
        """
        Initialize caching arrays for the tree.
        """
        assert self.dataX is not None, "Data matrix is not provided."
        self.leaf_ids = np.zeros(self.dataX.shape[0], dtype=np.int16)
        self.n = np.zeros(Tree.default_size, dtype=np.int32)
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
            vars, thresholds, leaf_vals, n=None, leaf_ids=None, evals=None
        )
        if dataX is not None:
            # If dataX is provided, initialize caching arrays.
            new_tree._init_caching_arrays()
        
        return new_tree

    @classmethod
    def from_existing(cls, other: "Tree"):
        """
        Create a new Tree object by copying data from an existing Tree.
        """
        # Copy all arrays
        
        return cls(
            other.dataX,  # dataX is not copied since it's shared across trees
            other.vars.copy(),
            other.thresholds.copy(), 
            other.leaf_vals.copy(),
            other.n.copy() if other.n is not None else None,
            other.leaf_ids.copy() if other.leaf_ids is not None else None,
            other.evals.copy() if other.evals is not None else None
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
        return _traverse_tree_numba(X, self.vars, self.thresholds)

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
            d[old_size:] = 0
            self.n = d

    
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
        is_valid = _update_n_and_leaf_id_numba(
            node_id, self.dataX, False, self.vars, self.thresholds, self.n, self.leaf_ids
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
                    self.n[left_child] = 0
                    self.n[right_child] = 0
                    # Update leaf_ids for samples that were in the pruned children
                    self.leaf_ids[self.leaf_ids == left_child] = current_node
                    self.leaf_ids[self.leaf_ids == right_child] = current_node
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
            self.n[current_node] = 0

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
        is_valid = _update_n_and_leaf_id_numba(
            node_id, self.dataX, False, self.vars, self.thresholds, self.n, self.leaf_ids
        )
        return is_valid
    
    def update_n_append(self, X_new):
        """
        Update the counts of samples reaching each node in the decision tree when appending new data.

        Parameters:
        - X_new (np.ndarray): The new data to append to the existing dataset.
        Returns:
        """
        _update_n_and_leaf_id_numba(
            0, X_new, True, self.vars, self.thresholds, self.n, self.leaf_ids
        )

    def update_outputs(self):
        self.evals = self.leaf_vals[self.leaf_ids]

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
        vs = self.vars
        tree_size = len(vs)
        result = []
        for i in range(tree_size):
            left, right = 2*i + 1, 2*i + 2
            # it suffices to check right child not overflowing
            if right < tree_size and vs[left] == -1 and vs[right] == -1:
                result.append(i)
        return result
    
    @property
    def nonterminal_split_nodes(self):
        return [i for i in range(len(self.vars)) if self.is_split_node(i) 
                and not self.is_terminal_split_node(i)]

    @property
    def leaf_basis(self) -> NDArray[np.float32]:
        leaves = self.leaves
        basis = np.zeros((len(self.leaf_ids), len(leaves)), dtype=np.float32)
        for i, leaf in enumerate(leaves):
            basis[:, i] = (self.leaf_ids == leaf).astype(np.float32)
        return basis

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
        Replace the stored feature matrix, extend leaf_ids array for any newly
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

        # Update leaf_ids
        if old_n == 0:
            # first time: init
            self._init_caching_arrays() 
        else:
            # extending existing leaf_ids
            new_leaf_ids = np.zeros(n_new, dtype=np.int32)  # all new samples start at root
            self.leaf_ids = np.concatenate([self.leaf_ids, new_leaf_ids])
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
            tree.leaf_ids = None
            tree.n = None
            
    def copy(self, modified_tree_ids=None):
        if modified_tree_ids is None:
            modified_tree_ids = range(self.n_trees)
        copied_trees = self.trees.copy() # Shallow copy
        for tree_id in modified_tree_ids:
            copied_trees[tree_id] = self.trees[tree_id].copy()
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
