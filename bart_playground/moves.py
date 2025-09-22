import re
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional
from .params import Parameters
from .util import fast_choice, fast_choice_with_weights


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
        self.s = current.global_params.get("s", None)
        self.tol = tol
        self.log_tran_ratio = 0 # The log of remaining transition ratio after cancellations in the MH acceptance probability. 

    @property
    def possible_thresholds(self):
        assert self._possible_thresholds, "possible_thresholds must be initialized"
        return self._possible_thresholds
    @property
    def _num_possible_proposals(self):
        return self.tol

    def _get_max_depth(self, tree):
        leaf_indices = [i for i, v in enumerate(tree.vars) if v == -1]
        if not leaf_indices:
            return 0
        max_leaf_id = max(leaf_indices)
        max_depth = int(np.log2(max_leaf_id + 1))
        return max_depth

    def get_n_samples(self, tree):
        if isinstance(self.n_samples_list, int):
            return self.n_samples_list
        max_depth = self._get_max_depth(tree)
        if max_depth < len(self.n_samples_list):
            return self.n_samples_list[max_depth]
        else:
            return 1

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

    def _calculate_simulated_likelihood(self, new_leaf_ids, new_n, residuals):
        """
        Calculate likelihood using simulated split data without modifying the tree.
        """
        from .priors import _single_tree_log_marginal_lkhd_numba
        return _single_tree_log_marginal_lkhd_numba(
            new_leaf_ids,
            new_n, 
            residuals,
            eps_sigma2=self.current.global_params["eps_sigma2"][0],
            f_sigma2=self.tree_prior.f_sigma2
        )

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
        var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
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
        var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
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
        self.swappable_pairs = []
        self.idx = 0
        
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
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)
        all_candidates = [
            (node_id, var)
            for node_id in tree.leaves
            for var in range(tree.dataX.shape[1])
        ]

        candidates = []
        n_candidate_trials = 0
        while len(candidates) < n_samples:
            node_id, var = fast_choice(generator, all_candidates)
            threshold = fast_choice(generator, self.possible_thresholds[var])
            n_candidate_trials += 1
            # Use the combined simulation function instead of copy + split_leaf
            new_leaf_ids, new_n, new_vars = tree.simulate_split_leaf(node_id, var, threshold)

            # Check if split is valid (both children have samples)
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            if new_n[left_child] > 0 and new_n[right_child] > 0:
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                
                # Calculate prior using simulated data
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                
                log_pi = log_likelihood + log_prior
                candidates.append((node_id, var, threshold, 0.5*float(log_pi)))

        if not candidates:
            return False
        
        self.candidate_sampling_ratio = n_candidate_trials / len(candidates)

        log_bwd_weights = np.array([w for _, _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights) # Select y
        node_id, var, threshold, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]

        log_tran_fwd = -np.log(tree.n_leaves)
        success = tree.split_leaf(node_id, var, threshold)

        log_tran_bwd = -np.log(len(tree.terminal_split_nodes)) # log T(y_i,x): prune back
        log_p_bwd = log_weight_yj + log_tran_bwd + np.log(bwd_weights.sum()) + max_log_bwd

        # Calculate the log transition ratio
        sampled_others = fast_choice(generator, tree.terminal_split_nodes, size=n_samples-1)
        prune_candidates = [node_id] + list(np.atleast_1d(sampled_others))

        log_fwd_weights = []
        log_pi_cache = {} # Cache to avoid redundant calculations
        for prune_node_id in prune_candidates:
            if prune_node_id in log_pi_cache:
                log_pi = log_pi_cache[prune_node_id]
            else:
                new_leaf_ids, new_n, new_vars = tree.simulate_prune_split(prune_node_id)
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                log_pi = log_likelihood + log_prior
                log_pi_cache[prune_node_id] = log_pi
            log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + log_tran_fwd + np.log(fwd_weights.sum()) + max_log_fwd

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
        
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)
        all_candidates = tree.terminal_split_nodes
        self.candidate_sampling_ratio = 1 # Just a placeholder, not used in MultiPrune because prune always succeeds

        if not all_candidates:
            return False

        sampled_candidates = list(np.atleast_1d(fast_choice(generator, all_candidates, size=n_samples)))

        log_pi_cache = {}
        candidates = []
        for node_id in sampled_candidates:
            if node_id in log_pi_cache:
                log_pi = log_pi_cache[node_id]
            else:
                # Use simulation function instead of copy + prune_split for candidate evaluation
                new_leaf_ids, new_n, new_vars = tree.simulate_prune_split(node_id)
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                # Calculate prior using simulated data
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                log_pi = log_likelihood + log_prior
                log_pi_cache[node_id] = log_pi
            candidates.append((node_id, 0.5*float(log_pi)))

        log_bwd_weights = np.array([w for _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights)
        node_id, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]

        log_tran_fwd = -np.log(len(tree.terminal_split_nodes))
        grow_candidate = (node_id, tree.vars[node_id], tree.thresholds[node_id]) # Record
        tree.prune_split(node_id)
        log_tran_bwd = -np.log(tree.n_leaves)  # log T(y_i, x): grow back
        log_p_bwd = log_weight_yj + log_tran_bwd + np.log(bwd_weights.sum()) + max_log_bwd

        # Calculate the log transition ratio
        n_samples = self.get_n_samples(tree)
        log_fwd_weights = []
    
        # First add the recorded grow candidate
        leaf_id, var, threshold = grow_candidate
        new_leaf_ids, new_n, new_vars = tree.simulate_split_leaf(leaf_id, var, threshold)
        left_child = leaf_id * 2 + 1
        right_child = leaf_id * 2 + 2
        log_likelihood = self.likelihood.calculate_simulated_likelihood(
            new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
        )
        log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
        log_pi = log_likelihood + log_prior
        log_fwd_weights.append(0.5 * float(log_pi))

        all_grow_candidates = [
            (leaf_id, var)
            for leaf_id in tree.leaves
            for var in range(tree.dataX.shape[1])
        ]

        while len(log_fwd_weights) < n_samples:
            node_id, var = fast_choice(generator, all_grow_candidates)
            threshold = fast_choice(generator, self.possible_thresholds[var])
            new_leaf_ids, new_n, new_vars = tree.simulate_split_leaf(node_id, var, threshold)
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            if new_n[left_child] > 0 and new_n[right_child] > 0:
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                log_pi = log_likelihood + log_prior
                log_fwd_weights.append(0.5 * float(log_pi))

        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + log_tran_fwd + np.log(fwd_weights.sum()) + max_log_fwd

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

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)
        all_candidates = [
            (node_id, var)
            for node_id in tree.split_nodes
            for var in range(tree.dataX.shape[1])
        ]

        candidates = []
        n_candidate_trials = 0
        while len(candidates) < n_samples:
            node_id, var = fast_choice(generator, all_candidates)
            threshold = fast_choice(generator, self.possible_thresholds[var])
            n_candidate_trials += 1
            new_leaf_ids, new_n, new_vars = tree.simulate_change_split(node_id, var, threshold)
            
            # Check if change is valid - all leaf nodes should have samples
            valid = True
            for i in range(node_id, len(new_vars)):
                if new_vars[i] != -2 and new_n[i] == 0:
                    valid = False
                    break
            
            if valid:
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                log_pi = log_likelihood
                candidates.append((node_id, var, threshold, 0.5*float(log_pi)))

        self.candidate_sampling_ratio = n_candidate_trials / n_samples
        if not candidates:
            return False

        log_bwd_weights = np.array([w for _, _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights)
        node_id, var, threshold, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]
        log_p_bwd = log_weight_yj + np.log(bwd_weights.sum()) + max_log_bwd

        old_var = tree.vars[node_id]
        old_threshold = tree.thresholds[node_id]
        success = tree.change_split(node_id, var, threshold)

        # Calculate the log transition ratio
        log_fwd_weights = []
        new_leaf_ids, new_n, new_vars = tree.simulate_change_split(node_id, old_var, old_threshold)
        log_likelihood = self.likelihood.calculate_simulated_likelihood(
            new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
        )
        log_pi = log_likelihood
        log_fwd_weights.append(0.5*float(log_pi))

        while len(log_fwd_weights) < n_samples:
            node_id, var = fast_choice(generator, all_candidates)
            threshold = fast_choice(generator, self.possible_thresholds[var])
            # Use simulation function instead of copy + change_split
            new_leaf_ids, new_n, new_vars = tree.simulate_change_split(node_id, var, threshold)

            # Check if change is valid - all leaf nodes should have samples
            valid = True
            for i in range(node_id, len(new_vars)):
                if new_vars[i] != -2 and new_n[i] == 0:
                    valid = False
                    break
            
            if valid:                
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                
                log_pi = log_likelihood
                log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + np.log(fwd_weights.sum()) + max_log_fwd

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

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        all_candidates = [
            (parent_id, 2 * parent_id + lr)
            for parent_id in tree.nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * parent_id + lr] != -1
        ]
        n_samples = self.get_n_samples(tree)

        log_pi_cache = {}
        candidates = []
        n_candidate_trials = 0
        while len(candidates) < n_samples:
            parent_id, child_id = fast_choice(generator, all_candidates)
            n_candidate_trials += 1
            cache_key = (parent_id, child_id)
            if cache_key in log_pi_cache:
                log_pi = log_pi_cache[cache_key]
            else:
                new_leaf_ids, new_n, new_vars = tree.simulate_swap_split(parent_id, child_id)
                # Check if swap is valid - all leaf nodes should have samples
                valid = True
                for i in range(parent_id, len(new_vars)):
                    if new_vars[i] != -2 and new_n[i] == 0:
                        valid = False
                        all_candidates.remove((parent_id, child_id))  # Remove invalid candidate
                        if not all_candidates:
                            return False
                        break
                if not valid:
                    continue
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                log_pi = log_likelihood
                log_pi_cache[cache_key] = log_pi
            candidates.append((parent_id, child_id, 0.5*float(log_pi)))

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates)) if all_candidates else 1

        log_bwd_weights = np.array([w for _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights)
        parent_id, child_id, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]
        log_p_bwd = log_weight_yj + np.log(bwd_weights.sum()) + max_log_bwd

        success = tree.swap_split(parent_id, child_id)

        # Calculate the log transition ratio
        ## First add the recorded swap candidate
        log_fwd_weights = []
        log_fwd_pi_cache = {}
        new_leaf_ids, new_n, new_vars = tree.simulate_swap_split(parent_id, child_id)
        log_likelihood = self.likelihood.calculate_simulated_likelihood(
            new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
        )
        log_pi = log_likelihood
        log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_pi_cache[(parent_id, child_id)] = log_pi

        all_candidates = [
            (parent_id, 2 * parent_id + lr)
            for parent_id in tree.nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * parent_id + lr] != -1
        ]

        while len(log_fwd_weights) < n_samples:
            p_id, c_id = fast_choice(generator, all_candidates)
            cache_key = (p_id, c_id)
            if cache_key in log_fwd_pi_cache:
                log_pi = log_fwd_pi_cache[cache_key]
            else:
                new_leaf_ids, new_n, new_vars = tree.simulate_swap_split(p_id, c_id)
                # Check if swap is valid - all leaf nodes should have samples
                valid = True
                for i in range(p_id, len(new_vars)):
                    if new_vars[i] != -2 and new_n[i] == 0:
                        valid = False
                        all_candidates.remove((p_id, c_id))  # Remove invalid candidate
                        break
                if not valid:
                    continue
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                log_pi = log_likelihood
                log_fwd_pi_cache[cache_key] = log_pi
            log_fwd_weights.append(0.5*float(log_pi))

        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + np.log(fwd_weights.sum()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return success

 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "multi_grow" : MultiGrow,
            "multi_prune" : MultiPrune,
            "multi_change" : MultiChange,
            "multi_swap" : MultiSwap}

# Mapping of each move to its contrary move used in MH ratio adjustments
contrary_moves = {
    "grow": "prune",
    "prune": "grow",
    "change": "change",
    "swap": "swap",
    "multi_grow": "multi_prune",
    "multi_prune": "multi_grow",
    "multi_change": "multi_change",
    "multi_swap": "multi_swap"
}
