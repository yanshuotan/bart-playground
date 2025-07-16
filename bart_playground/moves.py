import numpy as np
import math
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
    @property
    def _num_possible_proposals(self):
        return self.tol

    def propose(self, generator):
        """
        Propose a new state.
        """
        if self.is_feasible():
            for _ in range(self._num_possible_proposals):
                proposed = self.current.copy(self.trees_changed)
                success = self.try_propose(proposed, generator)
                if success:
                    self.proposed = proposed
                    return True
            # If exit loop without returning, have exceeded tol tries without 
            # finding a valid proposal.
        return False

    @abstractmethod
    def is_feasible(self) -> bool:
        """
        Check whether move is feasible.
        """
        pass

    @abstractmethod
    def try_propose(self, proposed, generator) -> bool:
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
        tree = self.current.trees[self.trees_changed[0]]
        self.cur_leaves = tree.leaves
        self.cur_n_terminal_splits = len(tree.terminal_split_nodes)
        return True
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, self.cur_leaves)
        var = generator.integers(tree.dataX.shape[1])
        threshold = fast_choice(generator, self.possible_thresholds[var])
        n_leaves = len(self.cur_leaves)
        
        success = tree.split_leaf(node_id, var, threshold)
        if node_id % 2:
            neighbor = node_id + 1
        else:
            neighbor = node_id - 1
        # Update the number of non-terminal splits
        # + 1 only if parent is a non-terminal split
        n_splits = self.cur_n_terminal_splits + 1 - tree.is_leaf(neighbor)
        self.log_tran_ratio = math.log(n_leaves) - math.log(n_splits)
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
        self.cur_terminal_split_nodes = tree.terminal_split_nodes
        return len(self.cur_terminal_split_nodes) > 0

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        node_id = fast_choice(generator, self.cur_terminal_split_nodes)
        n_splits = len(self.cur_terminal_split_nodes)
        
        tree.prune_split(node_id)
        n_leaves = tree.n_leaves
        self.log_tran_ratio = math.log(n_splits) - math.log(n_leaves)
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
        self.swappable_pairs = None
        self.idx = None
        
    @property
    def _num_possible_proposals(self):
        return min(self.tol, len(self.swappable_pairs))
    
    def _ini_swappable_pairs(self):
        tree = self.current.trees[self.trees_changed[0]]
        nonterminal_split_nodes = tree.nonterminal_split_nodes

        # Collect all valid parent-child pairs where the child is also a split node.
        self.swappable_pairs = [
            (parent_id, 2 * parent_id + lr)
            for parent_id in nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * parent_id + lr] != -1
        ]
        self.idx = 0

    def is_feasible(self):
        '''
        Note that this method has a side effect of initializing the swappable_pairs.
        '''
        self._ini_swappable_pairs()
        return self._num_possible_proposals > 0

    def try_propose(self, proposed, generator):
        if self.idx == 0: # Shuffle the pairs once at the start
            generator.shuffle(self.swappable_pairs)
            
        parent_id, child_id = self.swappable_pairs[self.idx]
        tree = proposed.trees[self.trees_changed[0]]
        success = tree.swap_split(parent_id, child_id)  # If no empty leaves are created
        self.idx += 1
        return success
    
class MultiGrow(Grow):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples_list=[10, 5], **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def _get_max_depth(self, tree):
        leaf_indices = [i for i, v in enumerate(tree.vars) if v == -1]
        max_leaf_id = max(leaf_indices)
        max_depth = int(np.log2(max_leaf_id + 1))
        return max_depth
    
    def _get_n_samples(self, tree):
        if isinstance(self.n_samples_list, int):
            return self.n_samples_list
        max_depth = self._get_max_depth(tree)
        if max_depth < len(self.n_samples_list):
            return self.n_samples_list[max_depth]
        else:
            return 1
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        n_samples = self._get_n_samples(tree)
        all_candidates = [
            (node_id, var, threshold)
            for node_id in tree.leaves
            for var in range(tree.dataX.shape[1])
            for threshold in self.possible_thresholds[var]
        ]

        generator.shuffle(all_candidates)

        sampled_candidates = []
        n_candidate_trials = 0
        for node_id, var, threshold in all_candidates:
            n_candidate_trials += 1
            tree_copy = tree.copy()
            if tree_copy.split_leaf(node_id, var, threshold):
                sampled_candidates.append((node_id, var, threshold))
                if len(sampled_candidates) >= n_samples:
                    break

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates))

        if not sampled_candidates:
            return False
        
        candidates = []
        for node_id, var, threshold in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.split_leaf(node_id, var, threshold)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed) # log pi(y_i)
            candidates.append((node_id, var, threshold, float(log_pi)))

        log_bwd_weights = np.array([0.5*w for _, _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        norm_log_bwd_weights = log_bwd_weights - max_log_bwd
        bwd_weights = np.exp(norm_log_bwd_weights)
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights) # Select y
        node_id, var, threshold, _ = candidates[idx]

        log_tran_fwd = -np.log(tree.n_leaves)
        success = tree.split_leaf(node_id, var, threshold)

        log_tran_bwd = -np.log(len(tree.terminal_split_nodes)) # log T(y_i,x): prune back
        log_p_bwd = log_tran_bwd + np.log(bwd_weights.mean()) + max_log_bwd

        # Calculate the log transition ratio
        all_prune_candidates = list(tree.terminal_split_nodes)
        n_all = len(all_prune_candidates)
        n_samples = min(self._get_n_samples(tree), n_all)

        # Make sure we sample the node_id itself and n_samples - 1 other candidates
        other_candidates = [nid for nid in all_prune_candidates if nid != node_id]
        others_idx = generator.choice(len(other_candidates), size=n_samples-1, replace=False)
        sampled_others = [other_candidates[i] for i in others_idx]

        prune_candidates = [node_id] + sampled_others

        log_fwd_weights = []
        for prune_node_id in prune_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.prune_split(prune_node_id)
            #
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed) # log pi(x_i*)
            #log_weight = float(log_pi + log_tran_kernel)
            log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        max_log_fwd = np.max(log_fwd_weights)
        norm_log_fwd_weights = log_fwd_weights - max_log_fwd
        fwd_weights = np.exp(norm_log_fwd_weights)
        log_p_fwd = log_tran_fwd + np.log(fwd_weights.mean()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return success
    
class MultiPrune(Prune):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples_list=[10, 5], **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        if possible_thresholds is None:
            raise ValueError("possible_thresholds must be provided for MultiPrune.")
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)
        
    def _get_max_depth(self, tree):
        leaf_indices = [i for i, v in enumerate(tree.vars) if v == -1]
        if not leaf_indices:
            return 0
        max_leaf_id = max(leaf_indices)
        max_depth = int(np.log2(max_leaf_id + 1))
        return max_depth

    def _get_n_samples(self, tree):
        if isinstance(self.n_samples_list, int):
            return self.n_samples_list
        max_depth = self._get_max_depth(tree)
        if max_depth < len(self.n_samples_list):
            return self.n_samples_list[max_depth]
        else:
            return 1
        
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        n_samples = self._get_n_samples(tree)
        all_candidates = list(tree.terminal_split_nodes)
        self.candidate_sampling_ratio = 1 # Just a placeholder, not used in MultiPrune because prune always succeeds

        if not all_candidates:
            return False

        if len(all_candidates) > n_samples:
            idxs = generator.choice(len(all_candidates), size=n_samples, replace=False)
            sampled_candidates = [all_candidates[i] for i in idxs]
        else:
            sampled_candidates = all_candidates

        candidates = []
        for node_id in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.prune_split(node_id)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed)
            candidates.append((node_id, float(log_pi)))

        log_bwd_weights = np.array([0.5*w for _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        norm_log_bwd_weights = log_bwd_weights - max_log_bwd
        bwd_weights = np.exp(norm_log_bwd_weights)
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, _ = candidates[idx]

        log_tran_fwd = -np.log(len(tree.terminal_split_nodes))
        grow_candidate = (node_id, tree.vars[node_id], tree.thresholds[node_id]) # Record
        tree.prune_split(node_id)
        log_tran_bwd = -np.log(tree.n_leaves)  # log T(y_i, x): grow back
        log_p_bwd = log_tran_bwd + np.log(bwd_weights.mean()) + max_log_bwd

        # Calculate the log transition ratio
        n_samples = self._get_n_samples(tree)
        all_grow_candidates = [
            (leaf_id, var, threshold)
            for leaf_id in tree.leaves
            for var in range(tree.dataX.shape[1])
            for threshold in self.possible_thresholds[var]
        ]

        generator.shuffle(all_grow_candidates)

        other_candidates = [cand for cand in all_grow_candidates if cand != grow_candidate]
        sampled_others = []
        for leaf_id, var, threshold in other_candidates:
            if len(sampled_others) >= n_samples - 1:
                break
            tree_copy = tree.copy()
            if tree_copy.split_leaf(leaf_id, var, threshold):
                sampled_others.append((leaf_id, var, threshold))

        grow_candidates = [grow_candidate] + sampled_others

        log_fwd_weights = []
        for leaf_id, var, threshold in grow_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.split_leaf(leaf_id, var, threshold)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed)
            log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        max_log_fwd = np.max(log_fwd_weights)
        norm_log_fwd_weights = log_fwd_weights - max_log_fwd
        fwd_weights = np.exp(norm_log_fwd_weights)
        log_p_fwd = log_tran_fwd + np.log(fwd_weights.mean()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return True
    
class MultiChange(Change):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples_list=[10, 5], **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def _get_max_depth(self, tree):
        leaf_indices = [i for i, v in enumerate(tree.vars) if v == -1]
        if not leaf_indices:
            return 0
        max_leaf_id = max(leaf_indices)
        max_depth = int(np.log2(max_leaf_id + 1))
        return max_depth

    def _get_n_samples(self, tree):
        if isinstance(self.n_samples_list, int):
            return self.n_samples_list
        max_depth = self._get_max_depth(tree)
        if max_depth < len(self.n_samples_list):
            return self.n_samples_list[max_depth]
        else:
            return 1

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        n_samples = self._get_n_samples(tree)
        all_candidates = [
            (node_id, var, threshold)
            for node_id in tree.split_nodes
            for var in range(tree.dataX.shape[1])
            for threshold in self.possible_thresholds[var]
        ]
        
        generator.shuffle(all_candidates)

        sampled_candidates = []
        n_candidate_trials = 0
        for node_id, var, threshold in all_candidates:
            n_candidate_trials += 1
            tree_copy = tree.copy()
            if tree_copy.change_split(node_id, var, threshold):
                sampled_candidates.append((node_id, var, threshold))
                if len(sampled_candidates) >= n_samples:
                    break

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates))

        if not sampled_candidates:
            return False

        candidates = []
        for node_id, var, threshold in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.change_split(node_id, var, threshold)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed) # log pi(y_i)
            candidates.append((node_id, var, threshold, float(log_pi)))

        log_bwd_weights = np.array([0.5*w for _, _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        norm_log_bwd_weights = log_bwd_weights - max_log_bwd
        bwd_weights = np.exp(norm_log_bwd_weights)
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, var, threshold, _ = candidates[idx]
        log_p_bwd = np.log(bwd_weights.mean()) + max_log_bwd

        old_var = tree.vars[node_id]
        old_threshold = tree.thresholds[node_id]
        success = tree.change_split(node_id, var, threshold)

        # Calculate the log transition ratio
        rev_candidate = (node_id, old_var, old_threshold)
        other_rev_candidates = [cand for cand in all_candidates if cand != rev_candidate]
        generator.shuffle(other_rev_candidates)

        sampled_others = []
        for node_id, var, threshold in other_rev_candidates:
            if len(sampled_others) >= n_samples - 1:
                break
            tree_copy = tree.copy()
            if tree_copy.change_split(node_id, var, threshold):
                sampled_others.append((node_id, var, threshold))

        rev_candidates = [rev_candidate] + sampled_others

        log_fwd_weights = []
        for nid, v, t in rev_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.change_split(nid, v, t)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed)
            log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        max_log_fwd = np.max(log_fwd_weights)
        norm_log_fwd_weights = log_fwd_weights - max_log_fwd
        fwd_weights = np.exp(norm_log_fwd_weights)
        log_p_fwd = np.log(fwd_weights.mean()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return success

class MultiSwap(Swap):
    def __init__(self, current, trees_changed, possible_thresholds=None, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples_list=[10, 5], **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def _get_max_depth(self, tree):
        leaf_indices = [i for i, v in enumerate(tree.vars) if v == -1]
        if not leaf_indices:
            return 0
        max_leaf_id = max(leaf_indices)
        max_depth = int(np.log2(max_leaf_id + 1))
        return max_depth

    def _get_n_samples(self, tree):
        if isinstance(self.n_samples_list, int):
            return self.n_samples_list
        max_depth = self._get_max_depth(tree)
        if max_depth < len(self.n_samples_list):
            return self.n_samples_list[max_depth]
        else:
            return 1

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        all_candidates = [
            (parent_id, 2 * parent_id + lr)
            for parent_id in tree.nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * parent_id + lr] != -1
        ]
        generator.shuffle(all_candidates)
        n_samples = self._get_n_samples(tree)

        sampled_candidates = []
        n_candidate_trials = 0
        for parent_id, child_id in all_candidates:
            n_candidate_trials += 1
            tree_copy = tree.copy()
            if tree_copy.swap_split(parent_id, child_id):
                sampled_candidates.append((parent_id, child_id))
                if len(sampled_candidates) >= n_samples:
                    break

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates)) if all_candidates else 1

        if not sampled_candidates:
            return False

        candidates = []
        for parent_id, child_id in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            temp_tree.swap_split(parent_id, child_id)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp, self.trees_changed)
            candidates.append((parent_id, child_id, float(log_pi)))

        log_bwd_weights = np.array([0.5*w for _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        norm_log_bwd_weights = log_bwd_weights - max_log_bwd
        bwd_weights = np.exp(norm_log_bwd_weights)
        weights = bwd_weights / bwd_weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        parent_id, child_id, _ = candidates[idx]
        log_p_bwd = np.log(bwd_weights.mean()) + max_log_bwd

        success = tree.swap_split(parent_id, child_id)

        # Calculate the log transition ratio
        all_rev_candidates = [
            (p_id, 2 * p_id + lr)
            for p_id in tree.nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * p_id + lr] != -1
        ]
        generator.shuffle(all_rev_candidates)

        rev_candidate = (parent_id, child_id)
        other_rev_candidates = [cand for cand in all_rev_candidates if cand != rev_candidate]
        sampled_others = []
        for p_id, c_id in other_rev_candidates:
            if len(sampled_others) >= n_samples - 1:
                break
            tree_copy = tree.copy()
            if tree_copy.swap_split(p_id, c_id):
                sampled_others.append((p_id, c_id))

        rev_candidates = [rev_candidate] + sampled_others

        log_fwd_weights = []
        for p_id, c_id in rev_candidates:
            temp2 = proposed.copy(self.trees_changed)
            temp2_tree = temp2.trees[self.trees_changed[0]]
            temp2_tree.swap_split(p_id, c_id)
            log_pi = self.likelihood.trees_log_marginal_lkhd(
                temp2, self.data_y, self.trees_changed
            ) + self.tree_prior.trees_log_prior(temp2, self.trees_changed)
            log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        max_log_fwd = np.max(log_fwd_weights)
        norm_log_fwd_weights = log_fwd_weights - max_log_fwd
        fwd_weights = np.exp(norm_log_fwd_weights)
        log_p_fwd = np.log(fwd_weights.mean()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return success

 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "multi_grow": MultiGrow,
            "multi_prune": MultiPrune,
            "multi_change": MultiChange,
            "multi_swap": MultiSwap}