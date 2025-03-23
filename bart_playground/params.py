from functools import cache
from math import e
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from .util import Dataset
from numba import njit


@njit
def _compute_leaf_basis(node_indicators, vars):
    """
    Numba-optimized function to compute leaf basis matrix.
    """
    leaves = np.where(vars == -1)[0]
    return node_indicators[:, leaves]


@njit
def _update_split_leaf_indicators(dataX, var, threshold, node_indicators, node_id, left_child, right_child):
    n_samples = dataX.shape[0]
    left_count = 0
    right_count = 0
    for i in range(n_samples):
        if node_indicators[i, node_id]:
            if dataX[i, var] > threshold:
                node_indicators[i, right_child] = True
                node_indicators[i, left_child] = False
                right_count += 1
            else:
                node_indicators[i, right_child] = False
                node_indicators[i, left_child] = True
                left_count += 1
        else:
            node_indicators[i, left_child] = False
            node_indicators[i, right_child] = False
    return left_count, right_count


@njit
def _copy_all_tree_arrays(vars, thresholds, leaf_vals, n, node_indicators, evals):
    vars_copy = vars.copy()
    thresholds_copy = thresholds.copy()
    leaf_vals_copy = leaf_vals.copy()
    n_copy = n.copy() if n is not None else None
    evals_copy = evals.copy() if evals is not None else None
    if node_indicators is not None:
        node_indicators_copy = np.empty_like(node_indicators)
        for i in range(node_indicators.shape[0]):
            for j in range(node_indicators.shape[1]):
                node_indicators_copy[i, j] = node_indicators[i, j]
    else:
        node_indicators_copy = None
    return vars_copy, thresholds_copy, leaf_vals_copy, n_copy, node_indicators_copy, evals_copy


class Tree:
    """
    Represents the parameters of a single tree in the BART model.
    """

    def __init__(self, dataX: Optional[np.ndarray], vars: np.ndarray, thresholds: np.ndarray,
                 leaf_vals: np.ndarray, n, node_indicators, evals):
        self.dataX = dataX
        self.vars = vars
        self.thresholds = thresholds
        self.leaf_vals: NDArray[np.float_] = leaf_vals
        self.n = n
        self.node_indicators = node_indicators
        self.evals = evals

    def _init_caching_arrays(self):
        assert self.dataX is not None, "Data matrix is not provided."
        self.node_indicators = np.full((self.dataX.shape[0], 8), False, dtype=bool)
        self.node_indicators[:, 0] = True
        self.n = np.full(8, -2, dtype=int)
        self.n[0] = self.dataX.shape[0]
        self.evals = np.zeros(self.dataX.shape[0])

    @property
    def cache_exists(self):
        return self.evals is not None

    @classmethod
    def new(cls, dataX=None):
        vars = np.full(8, -2, dtype=int)
        vars[0] = -1
        thresholds = np.full(8, np.nan, dtype=float)
        leaf_vals = np.full(8, np.nan, dtype=float)
        leaf_vals[0] = 0
        new_tree = cls(dataX, vars, thresholds, leaf_vals, n=None, node_indicators=None, evals=None)
        if dataX is not None:
            new_tree._init_caching_arrays()
        return new_tree

    @classmethod
    def from_existing(cls, other: "Tree"):
        vars_copy, thresholds_copy, leaf_vals_copy, n_copy, node_indicators_copy, evals_copy = _copy_all_tree_arrays(
            other.vars, other.thresholds, other.leaf_vals, other.n, other.node_indicators, other.evals
        )
        return cls(other.dataX, vars_copy, thresholds_copy, leaf_vals_copy, n_copy, node_indicators_copy, evals_copy)

    def copy(self):
        return Tree.from_existing(self)

    def traverse_tree(self, X: np.ndarray) -> np.ndarray:
        node_ids = np.full(X.shape[0], 0, dtype=int)
        if not self.split_nodes:
            return node_ids
        routing = X[:, self.vars[self.split_nodes]] > self.thresholds[self.split_nodes]
        split_node_counter = 0
        for k in range(len(self.vars)):
            if self.is_split_node(k):
                node_ids[node_ids == k] = node_ids[node_ids == k] * 2 + 1 + routing[node_ids == k, split_node_counter]
                split_node_counter += 1
        return node_ids

    def evaluate(self, X: Optional[np.ndarray] = None) -> NDArray[np.float_]:
        if X is None:
            if self.evals is not None:
                return self.evals
            else:
                raise ValueError("No cached data available for evaluation.")
        else:
            leaf_ids = self.traverse_tree(X)
            return self.leaf_vals[leaf_ids]

    def _resize_arrays(self):
        old_length = len(self.vars)
        new_length = old_length * 2
        new_vars = np.full(new_length, -2, dtype=int)
        new_vars[:old_length] = self.vars
        self.vars = new_vars
        new_thresholds = np.full(new_length, np.nan, dtype=float)
        new_thresholds[:old_length] = self.thresholds
        self.thresholds = new_thresholds
        new_leaf_vals = np.full(new_length, np.nan, dtype=float)
        new_leaf_vals[:old_length] = self.leaf_vals
        self.leaf_vals = new_leaf_vals
        if self.cache_exists:
            new_n = np.full(new_length, -2, dtype=int)
            new_n[:old_length] = self.n
            self.n = new_n
            new_node_indicators = np.full((self.node_indicators.shape[0], new_length), False, dtype=bool)
            new_node_indicators[:, :old_length] = self.node_indicators
            self.node_indicators = new_node_indicators

    def split_leaf(self, node_id: int, var: int, threshold: float, left_val: float = np.nan, right_val: float = np.nan):
        if self.vars[node_id] != -1:
            raise ValueError("Node is not a leaf and cannot be split.")
        left_child = node_id * 2 + 1
        right_child = node_id * 2 + 2
        if left_child >= len(self.vars) or right_child >= len(self.vars):
            self._resize_arrays()
        self.vars[node_id] = var
        self.thresholds[node_id] = threshold
        self.vars[left_child] = -1
        self.vars[right_child] = -1
        self.leaf_vals[node_id] = np.nan
        self.leaf_vals[left_child] = left_val
        self.leaf_vals[right_child] = right_val
        if self.cache_exists:
            left_count, right_count = _update_split_leaf_indicators(
                self.dataX, var, threshold, self.node_indicators, node_id, left_child, right_child)
            self.n[left_child] = left_count
            self.n[right_child] = right_count
            is_valid = left_count > 0 and right_count > 0
        else:
            is_valid = True
        return is_valid

    def prune_split(self, node_id: int, recursive=False):
        if not self.is_split_node(node_id):
            raise ValueError("Node is not a split node and cannot be pruned.")
        if not recursive and not self.is_terminal_split_node(node_id):
            raise ValueError("Node is not a terminal split node and cannot be pruned (recursive=False).")
        self.vars[node_id] = -1
        self.thresholds[node_id] = np.nan
        left_child = node_id * 2 + 1
        right_child = node_id * 2 + 2
        self.vars[left_child] = -2
        self.vars[right_child] = -2
        self.leaf_vals[left_child] = np.nan
        self.leaf_vals[right_child] = np.nan
        if self.cache_exists:
            self.n[left_child] = -2
            self.n[right_child] = -2
        if recursive:
            self._prune_descendants(left_child)
            self._prune_descendants(right_child)

    def _prune_descendants(self, node_id: int):
        if node_id >= len(self.vars):
            return
        self.vars[node_id] = -2
        self.thresholds[node_id] = np.nan
        self.leaf_vals[node_id] = np.nan
        self.n[node_id] = -2
        left_child = node_id * 2 + 1
        right_child = node_id * 2 + 2
        self._prune_descendants(left_child)
        self._prune_descendants(right_child)

    def change_split(self, node_id, var, threshold, update_n=True):
        self.vars[node_id] = var
        self.thresholds[node_id] = threshold
        if update_n:
            is_valid = self.update_n(node_id)
        return is_valid if update_n else True

    def swap_split(self, parent_id, child_id):
        parent_var, parent_threshold = self.vars[parent_id], self.thresholds[parent_id]
        child_var, child_threshold = self.vars[child_id], self.thresholds[child_id]
        self.change_split(child_id, parent_var, parent_threshold, update_n=False)
        is_valid = self.change_split(parent_id, child_var, child_threshold, update_n=True)
        return is_valid

    def update_n(self, node_id=0, X_range=None):
        if self.dataX is None:
            raise ValueError("Data matrix is not provided.")
        if X_range is None:
            X_range = np.arange(self.dataX.shape[0])
        if self.is_leaf(node_id):
            return self.n[node_id] > 0
        else:
            var = self.vars[node_id]
            threshold = self.thresholds[node_id]
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            self.node_indicators[X_range, left_child] = self.node_indicators[X_range, node_id] & (
                        self.dataX[X_range, var] <= threshold)
            self.node_indicators[X_range, right_child] = self.node_indicators[X_range, node_id] & (
                        self.dataX[X_range, var] > threshold)
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
        return self.is_split_node(node_id) and self.is_leaf(node_id * 2 + 1) and self.is_leaf(node_id * 2 + 2)

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
        return [i for i in range(len(self.vars)) if self.is_split_node(i) and not self.is_terminal_split_node(i)]

    @property
    def leaf_basis(self) -> NDArray[np.bool_]:
        if self.dataX is None:
            raise ValueError("Data matrix is not provided.")
        return _compute_leaf_basis(self.node_indicators, self.vars)

    def __str__(self):
        return self._print_tree()

    def __repr__(self):
        if self.dataX is None:
            return f"Tree(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals})"
        else:
            return f"Tree(vars={self.vars}, thresholds={self.thresholds}, leaf_vals={self.leaf_vals}, n_vals={self.n})"

    def _print_tree(self, node_id=0, prefix=""):
        pprefix = prefix + "\t"
        if self.vars[node_id] == -1:
            return prefix + self._print_node(node_id)
        else:
            left_idx = node_id * 2 + 1
            right_idx = node_id * 2 + 2
            return (
                    prefix + self._print_node(node_id) + "\n" +
                    self._print_tree(left_idx, pprefix) + "\n" +
                    self._print_tree(right_idx, pprefix)
            )

    def _print_node(self, node_id):
        n_output = self.n[node_id] if self.cache_exists else "NA"
        if self.vars[node_id] == -1:
            return f"Val: {self.leaf_vals[node_id]:0.3f} (leaf, n = {n_output})"
        else:
            return f"X_{self.vars[node_id]} <= {self.thresholds[node_id]:0.3f} (split, n = {n_output})"

    def add_data_points(self, new_dataX):
        if self.dataX is None:
            self.dataX = new_dataX
            self._init_caching_arrays()
            self.update_n()
            self.update_outputs()
            return
        n_new = new_dataX.shape[0]
        extended_indicators = np.full((n_new, len(self.vars)), False, dtype=bool)
        extended_indicators[:, 0] = True
        self.dataX = np.vstack([self.dataX, new_dataX])
        self.node_indicators = np.vstack([self.node_indicators, extended_indicators])
        self.update_n()
        self.update_outputs()


class Parameters:
    """
    Represents the parameters of the BART model.
    """

    def __init__(self, trees: list, global_params, cache: Optional[NDArray[np.float_]] = None):
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

    def copy(self, modified_tree_ids=None):
        """
        Returns a new Parameters object.
        Only trees in modified_tree_ids are deep-copied;
        the other trees are re-used by reference.
        """
        if modified_tree_ids is None:
            modified_tree_ids = range(self.n_trees)
        copied_trees = self.trees.copy()  # shallow copy of the list
        for tree_id in modified_tree_ids:
            copied_trees[tree_id] = self.trees[tree_id].copy()
        return Parameters(
            trees=copied_trees,
            global_params=self.global_params.copy(),
            cache=self.cache
        )

    def add_data_points(self, X_new):
        new_trees = self.trees.copy()
        for tree in new_trees:
            tree.add_data_points(X_new)
        return Parameters(
            trees=new_trees,
            global_params=self.global_params.copy(),
            cache=None
        )

    def evaluate(self, X: Optional[np.ndarray] = None, tree_ids: Optional[list[int]] = None,
                 all_except: Optional[list[int]] = None) -> NDArray[np.float_]:
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
                total_output += self.trees[i].evaluate(X)
        return total_output

    def leaf_basis(self, tree_ids):
        if len(tree_ids) == 1:
            return self.trees[tree_ids[0]].leaf_basis
        return np.hstack([self.trees[tree_id].leaf_basis for tree_id in tree_ids])

    def update_leaf_vals(self, tree_ids: list[int], leaf_vals: NDArray[np.float_]):
        leaf_counter = 0
        for tree_id in tree_ids:
            tree = self.trees[tree_id]
            tree_evals_old = tree.evals.copy()
            tree.leaf_vals[tree.leaves] = leaf_vals[range(leaf_counter, leaf_counter + tree.n_leaves)]
            tree.update_outputs()
            self.cache = self.cache + tree.evals - tree_evals_old
            leaf_counter += tree.n_leaves

    def update_tree(self, tree_idx: int, new_tree: Tree):
        old_tree_eval = self.trees[tree_idx].evaluate()
        self.trees[tree_idx] = new_tree
        new_tree.update_outputs()
        new_tree_eval = new_tree.evaluate()
        self.cache = self.cache - old_tree_eval + new_tree_eval
