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
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)
        all_candidates = [
            (node_id, var, threshold)
            for node_id in tree.leaves
            for var in range(tree.dataX.shape[1])
            for threshold in self.possible_thresholds[var]
        ]

        generator.shuffle(all_candidates)

        candidates = []
        n_candidate_trials = 0
        for node_id, var, threshold in all_candidates:
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
                if len(candidates) >= n_samples:
                    break

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates))

        if not candidates:
            return False

        log_bwd_weights = np.array([w for _, _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights) # Select y
        node_id, var, threshold, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]

        log_tran_fwd = -np.log(tree.n_leaves)
        success = tree.split_leaf(node_id, var, threshold)

        log_tran_bwd = -np.log(len(tree.terminal_split_nodes)) # log T(y_i,x): prune back
        log_p_bwd = log_weight_yj + log_tran_bwd + np.log(bwd_weights.mean()) + max_log_bwd

        # Calculate the log transition ratio
        all_prune_candidates = list(tree.terminal_split_nodes)
        n_all = len(all_prune_candidates)
        n_samples = min(self.get_n_samples(tree), n_all)

        # Make sure we sample the node_id itself and n_samples - 1 other candidates
        other_candidates = [nid for nid in all_prune_candidates if nid != node_id]
        others_idx = generator.choice(len(other_candidates), size=n_samples-1, replace=False)
        sampled_others = [other_candidates[i] for i in others_idx]

        prune_candidates = [node_id] + sampled_others

        log_fwd_weights = []
        for prune_node_id in prune_candidates:
            # Use simulation function instead of copy + prune_split
            new_leaf_ids, new_n, new_vars = tree.simulate_prune_split(prune_node_id)
            
            # Calculate likelihood using simulated data
            log_likelihood = self.likelihood.calculate_simulated_likelihood(
                new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
            )
            
            # Calculate prior using simulated data
            log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
            
            log_pi = log_likelihood + log_prior
            log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + log_tran_fwd + np.log(fwd_weights.mean()) + max_log_fwd

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
            # Use simulation function instead of copy + prune_split for candidate evaluation
            new_leaf_ids, new_n, new_vars = tree.simulate_prune_split(node_id)
            
            # Calculate likelihood using simulated data
            log_likelihood = self.likelihood.calculate_simulated_likelihood(
                new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
            )
            
            # Calculate prior using simulated data
            log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
            
            log_pi = log_likelihood + log_prior
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
        log_p_bwd = log_weight_yj + log_tran_bwd + np.log(bwd_weights.mean()) + max_log_bwd

        # Calculate the log transition ratio
        n_samples = self.get_n_samples(tree)
        all_grow_candidates = [
            (leaf_id, var, threshold)
            for leaf_id in tree.leaves
            for var in range(tree.dataX.shape[1])
            for threshold in self.possible_thresholds[var]
        ]

        rtol = 1e-5  # Relative tolerance
        grow_candidate = next(
            (cand for cand in all_grow_candidates
            if cand[0] == grow_candidate[0] and cand[1] == grow_candidate[1] and 
            math.isclose(cand[2], grow_candidate[2], rel_tol=rtol)),
            None
        )
        assert grow_candidate is not None, f"grow_candidate not found, thus cannot remove: {grow_candidate}"
        all_grow_candidates.remove(grow_candidate)
        generator.shuffle(all_grow_candidates)
        grow_candidates = [grow_candidate] + all_grow_candidates

        log_fwd_weights = []
        for leaf_id, var, threshold in grow_candidates:
            if len(log_fwd_weights) >= n_samples:
                break
            # Use simulation functions instead of copy + split_leaf
            new_leaf_ids, new_n, new_vars = tree.simulate_split_leaf(leaf_id, var, threshold)
            
            # Check if split is valid (both children have samples)
            left_child = leaf_id * 2 + 1
            right_child = leaf_id * 2 + 2
            if new_n[left_child] > 0 and new_n[right_child] > 0:
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                
                # Calculate prior using simulated data
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                
                log_pi = log_likelihood + log_prior
                log_fwd_weights.append(0.5*float(log_pi))

        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + log_tran_fwd + np.log(fwd_weights.mean()) + max_log_fwd

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
            (node_id, var, threshold)
            for node_id in tree.split_nodes
            for var in range(tree.dataX.shape[1])
            for threshold in self.possible_thresholds[var]
        ]
        
        generator.shuffle(all_candidates)

        candidates = []
        n_candidate_trials = 0
        for node_id, var, threshold in all_candidates:
            n_candidate_trials += 1
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
                
                # Note: No prior calculation needed for change operations 
                # because change does not affect the tree structure
                log_pi = log_likelihood
                candidates.append((node_id, var, threshold, 0.5*float(log_pi)))
                if len(candidates) >= n_samples:
                    break

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates))

        if not candidates:
            return False

        log_bwd_weights = np.array([w for _, _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights)
        node_id, var, threshold, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]
        log_p_bwd = log_weight_yj + np.log(bwd_weights.mean()) + max_log_bwd

        old_var = tree.vars[node_id]
        old_threshold = tree.thresholds[node_id]
        success = tree.change_split(node_id, var, threshold)

        # Calculate the log transition ratio
        rev_candidate = (node_id, old_var, old_threshold)
        rtol = 1e-5  # Relative tolerance
        rev_candidate = next(
            (cand for cand in all_candidates
            if cand[0] == node_id and cand[1] == old_var and 
            math.isclose(cand[2], old_threshold, rel_tol=rtol)),
            None
        )
        assert rev_candidate is not None, f"rev_candidate not found for node_id={node_id}, var={old_var}, threshold={old_threshold}, thus cannot remove"
        all_candidates.remove(rev_candidate)
        generator.shuffle(all_candidates)
        all_rev_candidates = [rev_candidate] + all_candidates

        log_fwd_weights = []
        for nid, v, t in all_rev_candidates:
            if len(log_fwd_weights) >= n_samples:
                break
            
            # Use simulation function instead of copy + change_split
            new_leaf_ids, new_n, new_vars = tree.simulate_change_split(nid, v, t)
            
            # Check if change is valid - all leaf nodes should have samples
            valid = True
            for i in range(nid, len(new_vars)):
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
        log_p_fwd = log_weight_x + np.log(fwd_weights.mean()) + max_log_fwd

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
        generator.shuffle(all_candidates)
        n_samples = self.get_n_samples(tree)

        candidates = []
        n_candidate_trials = 0
        for parent_id, child_id in all_candidates:
            n_candidate_trials += 1
            new_leaf_ids, new_n, new_vars = tree.simulate_swap_split(parent_id, child_id)
            
            # Check if swap is valid - all leaf nodes should have samples
            valid = True
            for i in range(parent_id, len(new_vars)):
                if new_vars[i] != -2 and new_n[i] == 0:
                    valid = False
                    break

            if valid:
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2
                )
                
                # Note: No prior calculation needed for swap operations 
                # because swap does not affect the tree structure
                log_pi = log_likelihood
                candidates.append((parent_id, child_id, 0.5*float(log_pi)))
                if len(candidates) >= n_samples:
                    break

        self.candidate_sampling_ratio = n_candidate_trials / min(n_samples, len(all_candidates)) if all_candidates else 1

        if not candidates:
            return False

        log_bwd_weights = np.array([w for _, _, w in candidates])
        max_log_bwd = np.max(log_bwd_weights)
        bwd_weights = np.exp(log_bwd_weights - max_log_bwd)
        idx = fast_choice_with_weights(generator, np.arange(len(candidates)), bwd_weights)
        parent_id, child_id, _ = candidates[idx]
        log_weight_yj = log_bwd_weights[[idx]]
        log_p_bwd = log_weight_yj + np.log(bwd_weights.mean()) + max_log_bwd

        success = tree.swap_split(parent_id, child_id)

        # Calculate the log transition ratio
        all_rev_candidates = [
            (p_id, 2 * p_id + lr)
            for p_id in tree.nonterminal_split_nodes
            for lr in [1, 2]
            if tree.vars[2 * p_id + lr] != -1
        ]

        rev_candidate = (parent_id, child_id)
        all_rev_candidates.remove(rev_candidate)
        generator.shuffle(all_rev_candidates)
        rev_candidates = [rev_candidate] + all_rev_candidates

        log_fwd_weights = []
        for p_id, c_id in rev_candidates:
            if len(log_fwd_weights) >= n_samples:
                break
            new_leaf_ids, new_n, new_vars = tree.simulate_swap_split(p_id, c_id)
            
            # Check if swap is valid - all leaf nodes should have samples
            valid = True
            for i in range(p_id, len(new_vars)):
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
        log_p_fwd = log_weight_x + np.log(fwd_weights.mean()) + max_log_fwd

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