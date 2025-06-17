import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from .params import Parameters
from .util import fast_choice
class Move(ABC):
    """
    Base class for moves in the BART sampler.
    """
    def __init__(self, current : Parameters, trees_changed: np.ndarray, 
                 possible_thresholds : Optional[dict] = None, tol : int = 100, **kwargs):
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
                 possible_thresholds : dict, tol : int = 100, **kwargs):
        if not possible_thresholds:
            raise ValueError("Possible thresholds must be provided for grow move.")
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)
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
                 possible_thresholds = None, tol : int = 100, **kwargs):
        super().__init__(current, trees_changed, possible_thresholds, tol = tol, **kwargs)
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
                 possible_thresholds : dict, tol : int = 100, **kwargs):
        if not possible_thresholds:
            raise ValueError("Possible thresholds must be provided for change move.")
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)
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
                 possible_thresholds = None, tol : int = 100, **kwargs):
        super().__init__(current, trees_changed, tol = tol, **kwargs)
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
    
class MultiGrow(Grow):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples=10, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples = n_samples
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        all_candidates = []

        for node_id in tree.leaves:
            for var in range(tree.dataX.shape[1]):
                for threshold in self.possible_thresholds[var]:
                    tree_copy = tree.copy()
                    if tree_copy.split_leaf(node_id, var, threshold):
                        all_candidates.append((node_id, var, threshold))


        if not all_candidates:
            return False

        if len(all_candidates) > self.n_samples:
            idxs = generator.choice(len(all_candidates), size=self.n_samples, replace=False)
            sampled_candidates = [all_candidates[i] for i in idxs]
        else:
            sampled_candidates = all_candidates
        
        candidates = []
        for node_id, var, threshold in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.split_leaf(node_id, var, threshold)
            tran_kernel = 1 / len(temp_tree.terminal_split_nodes) # T(y_i,x): prune back
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed) # log pi(y_i)
            weight = float(np.exp(log_pi) * tran_kernel)
            candidates.append((node_id, var, threshold, weight))

        bwd_weights = np.array([w for _, _, _, w in candidates])
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights) # Select y
        node_id, var, threshold, _ = candidates[idx]
        p_bwd = bwd_weights.mean() # Use mean instead of sum to avoid differences in n_samples

        tree = proposed.trees[self.trees_changed[0]]
        success = tree.split_leaf(node_id, var, threshold)

        # Calculate the log transition ratio
        temp_tree = tree
        all_prune_candidates = list(temp_tree.terminal_split_nodes)
        n_all = len(all_prune_candidates)
        n_samples = min(self.n_samples, n_all)

        # Make sure we sample the node_id itself and n_samples - 1 other candidates
        other_candidates = [nid for nid in all_prune_candidates if nid != node_id]
        if len(other_candidates) >= n_samples - 1:
            sampled_others = generator.choice(len(other_candidates), size=n_samples-1, replace=False)
            sampled_others = [other_candidates[i] for i in sampled_others]
        else:
            sampled_others = other_candidates
        prune_candidates = [node_id] + sampled_others

        fwd_weights = []
        for prune_node_id in prune_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.prune_split(prune_node_id)
            tran_kernel = 1 / temp2_tree.n_leaves # T(x_i*, y): grow
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed) # log pi(x_i*)
            weight = float(np.exp(log_pi) * tran_kernel)
            fwd_weights.append(weight)
        fwd_weights = np.array(fwd_weights)
        p_fwd = fwd_weights.mean()

        self.log_tran_ratio = np.log(p_bwd) - np.log(p_fwd)
        return success
    
class MultiPrune(Prune):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples=10, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples = n_samples
        if possible_thresholds is None:
            raise ValueError("possible_thresholds must be provided for MultiPrune.")
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        all_candidates = list(tree.terminal_split_nodes)

        if not all_candidates:
            return False

        if len(all_candidates) > self.n_samples:
            idxs = generator.choice(len(all_candidates), size=self.n_samples, replace=False)
            sampled_candidates = [all_candidates[i] for i in idxs]
        else:
            sampled_candidates = all_candidates

        candidates = []
        for node_id in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.prune_split(node_id)
            tran_kernel = 1 / temp_tree.n_leaves  # T(y_i, x): grow back
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed)
            weight = float(np.exp(log_pi) * tran_kernel)
            candidates.append((node_id, weight))

        bwd_weights = np.array([w for _, w in candidates])
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, _ = candidates[idx]
        p_bwd = bwd_weights.mean()

        tree = proposed.trees[self.trees_changed[0]]
        grow_candidate = (node_id, tree.vars[node_id], tree.thresholds[node_id]) # Record
        tree.prune_split(node_id)

        # Calculate the log transition ratio
        temp_tree = tree
        all_grow_candidates = []
        for leaf_id in temp_tree.leaves:
            for var in range(temp_tree.dataX.shape[1]):
                for threshold in self.possible_thresholds[var]:
                    tree_copy = temp_tree.copy()
                    if tree_copy.split_leaf(leaf_id, var, threshold):
                        all_grow_candidates.append((leaf_id, var, threshold))
        n_all = len(all_grow_candidates)
        n_samples = min(self.n_samples, n_all)

        other_candidates = [cand for cand in all_grow_candidates if cand != grow_candidate]
        if len(other_candidates) >= n_samples - 1:
            sampled_others = generator.choice(len(other_candidates), size=n_samples-1, replace=False)
            sampled_others = [other_candidates[i] for i in sampled_others]
        else:
            sampled_others = other_candidates
        grow_candidates = [grow_candidate] + sampled_others

        fwd_weights = []
        for leaf_id, var, threshold in grow_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.split_leaf(leaf_id, var, threshold)
            tran_kernel = 1 / len(temp2_tree.terminal_split_nodes)  # T(x_i*, y): prune
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed)
            weight = float(np.exp(log_pi) * tran_kernel)
            fwd_weights.append(weight)
        fwd_weights = np.array(fwd_weights)
        p_fwd = fwd_weights.mean()

        self.log_tran_ratio = np.log(p_bwd) - np.log(p_fwd)
        return True
    
class MultiChange(Change):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples=10, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples = n_samples
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        all_candidates = []
        for node_id in tree.split_nodes:
            for var in range(tree.dataX.shape[1]):
                for threshold in self.possible_thresholds[var]:
                    tree_copy = tree.copy()
                    if tree_copy.change_split(node_id, var, threshold):
                        all_candidates.append((node_id, var, threshold))

        if not all_candidates:
                    return False

        n_all = len(all_candidates)
        n_samples = min(self.n_samples, n_all)

        idxs = generator.choice(n_all, size=n_samples, replace=False)
        sampled_candidates = [all_candidates[i] for i in idxs]

        candidates = []
        for node_id, var, threshold in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.change_split(node_id, var, threshold)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed) # log pi(y_i)
            weight = float(np.exp(log_pi))
            candidates.append((node_id, var, threshold, weight))

        bwd_weights = np.array([w for _, _, _, w in candidates])
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, var, threshold, _ = candidates[idx]
        p_bwd = bwd_weights.sum()

        tree = proposed.trees[self.trees_changed[0]]
        old_var = tree.vars[node_id]
        old_threshold = tree.thresholds[node_id]
        success = tree.change_split(node_id, var, threshold)

        # Calculate the log transition ratio
        temp_tree = tree
        rev_candidate = (node_id, old_var, old_threshold)
        other_rev_candidates = [cand for cand in all_candidates if cand != rev_candidate]
        idxs = generator.choice(len(other_rev_candidates), size=n_samples-1, replace=False)
        sampled_others = [other_rev_candidates[i] for i in idxs]
        rev_candidates = [rev_candidate] + sampled_others

        fwd_weights = []
        for nid, v, t in rev_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.change_split(nid, v, t)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed)
            weight = float(np.exp(log_pi))
            fwd_weights.append(weight)
        fwd_weights = np.array(fwd_weights)
        p_fwd = fwd_weights.sum()

        self.log_tran_ratio = np.log(p_bwd) - np.log(p_fwd)
        return success

class MultiSwap(Swap):
    def __init__(self, current, trees_changed, possible_thresholds=None, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples=10, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples = n_samples
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        all_candidates = []
        for parent_id in tree.nonterminal_split_nodes:
            for lr in [1, 2]:
                child_id = 2 * parent_id + lr
                if tree.vars[child_id] == -1:
                    continue
                tree_copy = tree.copy()
                if tree_copy.swap_split(parent_id, child_id):
                    all_candidates.append((parent_id, child_id))

        if not all_candidates:
            return False

        n_all = len(all_candidates)
        n_samples = min(self.n_samples, n_all)

        idxs = generator.choice(n_all, size=n_samples, replace=False)
        sampled_candidates = [all_candidates[i] for i in idxs]

        candidates = []
        for parent_id, child_id in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.swap_split(parent_id, child_id)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed)
            weight = float(np.exp(log_pi))
            candidates.append((parent_id, child_id, weight))

        bwd_weights = np.array([w for _, _, w in candidates])
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        parent_id, child_id, _ = candidates[idx]
        p_bwd = bwd_weights.mean()

        tree = proposed.trees[self.trees_changed[0]]
        success = tree.swap_split(parent_id, child_id)

        # Calculate the log transition ratio
        temp_tree = tree
        all_rev_candidates = []
        for p_id in temp_tree.nonterminal_split_nodes:
            for lr in [1, 2]:
                c_id = 2 * p_id + lr
                if temp_tree.vars[c_id] == -1:
                    continue
                tree_copy = temp_tree.copy()
                if tree_copy.swap_split(parent_id, child_id):
                    all_rev_candidates.append((p_id, c_id))
        n_all_rev = len(all_rev_candidates)
        n_samples_rev = min(self.n_samples, n_all_rev)

        rev_candidate = (parent_id, child_id)
        other_rev_candidates = [cand for cand in all_rev_candidates if cand != rev_candidate]
        idxs = generator.choice(len(other_rev_candidates), size=n_samples_rev-1, replace=False)
        sampled_others = [other_rev_candidates[i] for i in idxs]
        rev_candidates = [rev_candidate] + sampled_others

        fwd_weights = []
        for p_id, c_id in rev_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.swap_split(p_id, c_id)
            self.proposed = temp2
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed)
            weight = float(np.exp(log_pi))
            fwd_weights.append(weight)
        fwd_weights = np.array(fwd_weights)
        p_fwd = fwd_weights.mean()

        self.log_tran_ratio = np.log(p_bwd) - np.log(p_fwd)
        return success

 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "multi_grow": MultiGrow,
            "multi_prune": MultiPrune,
            "multi_change": MultiChange,
            "multi_swap": MultiSwap}