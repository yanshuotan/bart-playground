import re
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Optional
from .params import Parameters
from .util import fast_choice, fast_choice_with_weights


INFORMED_V1 = "informed_v1"


def _normalized_variable_probs(n_features, splitting_weights):
    """Return the split-rule prior probabilities over variables."""
    if splitting_weights is None:
        return np.full(n_features, 1.0 / n_features, dtype=float)
    probs = np.asarray(splitting_weights, dtype=float).reshape(-1)
    if probs.size != n_features or not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("Invalid splitting weights for informed proposal kernel.")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Splitting weights must have positive sum.")
    return probs / total


def _find_threshold_index(thresholds, value):
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0 or not np.isfinite(value):
        return None
    idx = int(np.argmin(np.abs(thresholds - float(value))))
    scale = max(1.0, abs(float(value)), abs(float(thresholds[idx])))
    tolerance = 8.0 * np.finfo(np.float32).eps * scale
    return idx if abs(float(thresholds[idx]) - float(value)) <= tolerance else None


def _rule_prior_probability(possible_thresholds, variable_probs, var, threshold):
    thresholds = np.asarray(possible_thresholds[int(var)])
    idx = _find_threshold_index(thresholds, threshold)
    if idx is None or thresholds.size == 0:
        return 0.0
    return float(variable_probs[int(var)]) / float(thresholds.size)


def _stable_softmax(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    finite = np.isfinite(scores)
    if not np.any(finite):
        return np.zeros(scores.size, dtype=float)
    out = np.zeros(scores.size, dtype=float)
    shifted = scores[finite] - np.max(scores[finite])
    out[finite] = np.exp(shifted)
    total = float(out.sum())
    return out / total if total > 0 else np.zeros(scores.size, dtype=float)


def _leaf_informed_probs(tree, residuals, strength):
    """Favor leaves whose partial residuals still have substantial variation."""
    leaves = np.asarray(tree.leaves, dtype=int)
    raw_scores = np.zeros(leaves.size, dtype=float)
    for idx, leaf_id in enumerate(leaves):
        values = residuals[tree.leaf_ids == leaf_id]
        if values.size > 1:
            centered = values - np.mean(values)
            raw_scores[idx] = float(np.dot(centered, centered))
    total = float(raw_scores.sum())
    if total <= 0:
        return np.full(leaves.size, 1.0 / leaves.size, dtype=float)
    # The normalized score is in [0, 1], so strength has a stable meaning
    # across datasets and response scales.
    return _stable_softmax(float(strength) * raw_scores / total)


def _threshold_informed_probs(tree, residuals, node_id, var, thresholds, strength, min_leaf):
    """Score cutpoints by the fraction of within-node SSE removed by a split."""
    thresholds = np.asarray(thresholds)
    mask = tree.leaf_ids == int(node_id)
    x = tree.dataX[mask, int(var)]
    r = np.asarray(residuals)[mask]
    probs = np.zeros(thresholds.size, dtype=float)
    if r.size < 2 * int(min_leaf) or thresholds.size == 0:
        return probs

    centered = r - np.mean(r)
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(float).eps:
        denominator = 1.0

    scores = np.full(thresholds.size, -np.inf, dtype=float)
    total_sum = float(r.sum())
    total_n = int(r.size)
    parent_term = total_sum * total_sum / total_n
    for idx, threshold in enumerate(thresholds):
        left = x <= threshold
        n_left = int(left.sum())
        n_right = total_n - n_left
        if n_left < int(min_leaf) or n_right < int(min_leaf):
            continue
        sum_left = float(r[left].sum())
        sum_right = total_sum - sum_left
        gain = sum_left * sum_left / n_left + sum_right * sum_right / n_right - parent_term
        scores[idx] = float(strength) * max(0.0, gain) / denominator
    return _stable_softmax(scores)


def _grow_proposal_probability(
    tree,
    possible_thresholds,
    splitting_weights,
    residuals,
    node_id,
    var,
    threshold,
    config,
):
    """Evaluate the defensive-mixture grow proposal q(rule, leaf | tree)."""
    leaves = np.asarray(tree.leaves, dtype=int)
    leaf_matches = np.flatnonzero(leaves == int(node_id))
    if leaf_matches.size == 0:
        return 0.0
    n_features = int(tree.dataX.shape[1])
    variable_probs = _normalized_variable_probs(n_features, splitting_weights)
    thresholds = np.asarray(possible_thresholds[int(var)])
    threshold_idx = _find_threshold_index(thresholds, threshold)
    if threshold_idx is None or thresholds.size == 0:
        return 0.0

    q_uniform = (
        1.0 / float(leaves.size)
        * float(variable_probs[int(var)])
        / float(thresholds.size)
    )
    leaf_probs = _leaf_informed_probs(tree, residuals, config["leaf_score_strength"])
    threshold_probs = _threshold_informed_probs(
        tree,
        residuals,
        int(node_id),
        int(var),
        thresholds,
        config["threshold_score_strength"],
        config["min_leaf"],
    )
    q_informed = (
        float(leaf_probs[int(leaf_matches[0])])
        * float(variable_probs[int(var)])
        * float(threshold_probs[int(threshold_idx)])
    )
    weight = float(config["grow_informed_weight"])
    return (1.0 - weight) * q_uniform + weight * q_informed


def _sample_grow_rule(tree, possible_thresholds, splitting_weights, residuals, config, generator):
    leaves = np.asarray(tree.leaves, dtype=int)
    n_features = int(tree.dataX.shape[1])
    variable_probs = _normalized_variable_probs(n_features, splitting_weights)
    informed = bool(generator.uniform() < float(config["grow_informed_weight"]))

    if informed:
        leaf_probs = _leaf_informed_probs(tree, residuals, config["leaf_score_strength"])
        node_id = int(fast_choice_with_weights(generator, leaves, weights=leaf_probs))
    else:
        node_id = int(fast_choice(generator, leaves))
    var = int(fast_choice_with_weights(generator, np.arange(n_features), weights=variable_probs))
    thresholds = np.asarray(possible_thresholds[var])
    if thresholds.size == 0:
        return None, "informed" if informed else "uniform"

    if informed:
        threshold_probs = _threshold_informed_probs(
            tree,
            residuals,
            node_id,
            var,
            thresholds,
            config["threshold_score_strength"],
            config["min_leaf"],
        )
        if float(threshold_probs.sum()) <= 0:
            return None, "informed"
        threshold = fast_choice_with_weights(
            generator, thresholds, weights=threshold_probs
        )
    else:
        threshold = fast_choice(generator, thresholds)
    return (node_id, var, threshold), "informed" if informed else "uniform"


def _change_rule_probability(
    possible_thresholds,
    variable_probs,
    old_var,
    old_threshold,
    new_var,
    new_threshold,
    config,
):
    q_global = _rule_prior_probability(
        possible_thresholds, variable_probs, new_var, new_threshold
    )
    old_thresholds = np.asarray(possible_thresholds[int(old_var)])
    old_idx = _find_threshold_index(old_thresholds, old_threshold)
    local_indices = []
    if old_idx is not None:
        radius = int(config["change_local_radius"])
        lo = max(0, old_idx - radius)
        hi = min(old_thresholds.size, old_idx + radius + 1)
        local_indices = [idx for idx in range(lo, hi) if idx != old_idx]

    if not local_indices:
        q_local = q_global
    elif int(new_var) != int(old_var):
        q_local = 0.0
    else:
        new_idx = _find_threshold_index(old_thresholds, new_threshold)
        q_local = 1.0 / len(local_indices) if new_idx in local_indices else 0.0
    weight = float(config["change_local_weight"])
    return (1.0 - weight) * q_global + weight * q_local


def _valid_swappable_pairs(tree):
    pairs = [
        (parent_id, 2 * parent_id + lr)
        for parent_id in tree.nonterminal_split_nodes
        for lr in (1, 2)
        if tree.vars[2 * parent_id + lr] != -1
    ]
    valid = []
    for parent_id, child_id in pairs:
        _, new_n, new_vars = tree.simulate_swap_split(parent_id, child_id)
        active_leaves = new_vars == -1
        if np.all(new_n[active_leaves] > 0):
            valid.append((parent_id, child_id))
    return valid


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
        self.s_cumsum = current.global_params.get("s_cumsum", None)
        self.tol = tol
        self.proposal_kernel = str(kwargs.get("proposal_kernel", "legacy")).lower()
        self.proposal_config = kwargs.get("proposal_config", None)
        self.data_y = kwargs.get("data_y", None)
        self.diagnostics = {"kernel": self.proposal_kernel}
        self.log_tran_ratio = 0 # The log of remaining transition ratio after cancellations in the MH acceptance probability. 

    @property
    def possible_thresholds(self):
        assert self._possible_thresholds, "possible_thresholds must be initialized"
        return self._possible_thresholds
    @property
    def _num_possible_proposals(self):
        if self.proposal_kernel == INFORMED_V1:
            # Retrying until valid implicitly conditions q on proposal validity.
            # V1 instead makes one exact MH proposal; invalid proposals stay put.
            return 1
        return self.tol

    def _partial_residuals(self):
        if self.data_y is None:
            raise ValueError("data_y is required by the informed proposal kernel.")
        tree_id = int(self.trees_changed[0])
        tree = self.current.trees[tree_id]
        return np.asarray(self.data_y) - self.current.cache + tree.evals

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
        if self.proposal_kernel == INFORMED_V1:
            residuals = self._partial_residuals()
            sampled, component = _sample_grow_rule(
                tree,
                self.possible_thresholds,
                self.s,
                residuals,
                self.proposal_config,
                generator,
            )
            self.diagnostics["component"] = component
            if sampled is None:
                self.diagnostics["failure"] = "no_valid_informed_cutpoint"
                return False
            node_id, var, threshold = sampled
            q_forward = _grow_proposal_probability(
                tree,
                self.possible_thresholds,
                self.s,
                residuals,
                node_id,
                var,
                threshold,
                self.proposal_config,
            )
            variable_probs = _normalized_variable_probs(tree.dataX.shape[1], self.s)
            rule_prior = _rule_prior_probability(
                self.possible_thresholds, variable_probs, var, threshold
            )
            if q_forward <= 0 or rule_prior <= 0:
                self.diagnostics["failure"] = "zero_forward_probability"
                return False
            success = tree.split_leaf(node_id, var, threshold)
            if not success:
                self.diagnostics["failure"] = "empty_child"
                return False
            n_reverse_prunes = len(tree.terminal_split_nodes)
            q_reverse = 1.0 / float(n_reverse_prunes)
            self.log_tran_ratio = math.log(rule_prior) + math.log(q_reverse) - math.log(q_forward)
            self.diagnostics.update(
                {
                    "node_id": int(node_id),
                    "var": int(var),
                    "q_forward": float(q_forward),
                    "q_reverse": float(q_reverse),
                    "rule_prior": float(rule_prior),
                }
            )
            return True

        node_id = fast_choice(generator, self.cur_leaves)
        var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s, cum_weights=self.s_cumsum)
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
        if self.proposal_kernel == INFORMED_V1:
            residuals = self._partial_residuals()
            node_id = int(fast_choice(generator, self.cur_terminal_split_nodes))
            n_forward_prunes = len(self.cur_terminal_split_nodes)
            old_var = int(tree.vars[node_id])
            old_threshold = tree.thresholds[node_id]
            variable_probs = _normalized_variable_probs(tree.dataX.shape[1], self.s)
            rule_prior = _rule_prior_probability(
                self.possible_thresholds, variable_probs, old_var, old_threshold
            )
            if rule_prior <= 0:
                self.diagnostics["failure"] = "zero_rule_prior"
                return False
            tree.prune_split(node_id)
            q_reverse = _grow_proposal_probability(
                tree,
                self.possible_thresholds,
                self.s,
                residuals,
                node_id,
                old_var,
                old_threshold,
                self.proposal_config,
            )
            if q_reverse <= 0:
                self.diagnostics["failure"] = "zero_reverse_grow_probability"
                return False
            q_forward = 1.0 / float(n_forward_prunes)
            self.log_tran_ratio = -math.log(rule_prior) + math.log(q_reverse) - math.log(q_forward)
            self.diagnostics.update(
                {
                    "node_id": int(node_id),
                    "q_forward": float(q_forward),
                    "q_reverse": float(q_reverse),
                    "rule_prior": float(rule_prior),
                }
            )
            return True

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
        if self.proposal_kernel == INFORMED_V1:
            node_id = int(fast_choice(generator, tree.split_nodes))
            old_var = int(tree.vars[node_id])
            old_threshold = tree.thresholds[node_id]
            variable_probs = _normalized_variable_probs(tree.dataX.shape[1], self.s)
            local = bool(
                generator.uniform() < float(self.proposal_config["change_local_weight"])
            )
            old_thresholds = np.asarray(self.possible_thresholds[old_var])
            old_idx = _find_threshold_index(old_thresholds, old_threshold)
            local_indices = []
            if old_idx is not None:
                radius = int(self.proposal_config["change_local_radius"])
                lo = max(0, old_idx - radius)
                hi = min(old_thresholds.size, old_idx + radius + 1)
                local_indices = [idx for idx in range(lo, hi) if idx != old_idx]
            if local and local_indices:
                var = old_var
                threshold = old_thresholds[int(fast_choice(generator, local_indices))]
                component = "local"
            else:
                var = int(
                    fast_choice_with_weights(
                        generator,
                        np.arange(tree.dataX.shape[1]),
                        weights=variable_probs,
                    )
                )
                thresholds = np.asarray(self.possible_thresholds[var])
                if thresholds.size == 0:
                    self.diagnostics["failure"] = "no_global_cutpoint"
                    return False
                threshold = fast_choice(generator, thresholds)
                component = "global" if not local else "local_fallback_global"

            q_forward = _change_rule_probability(
                self.possible_thresholds,
                variable_probs,
                old_var,
                old_threshold,
                var,
                threshold,
                self.proposal_config,
            )
            q_reverse = _change_rule_probability(
                self.possible_thresholds,
                variable_probs,
                var,
                threshold,
                old_var,
                old_threshold,
                self.proposal_config,
            )
            old_prior = _rule_prior_probability(
                self.possible_thresholds, variable_probs, old_var, old_threshold
            )
            new_prior = _rule_prior_probability(
                self.possible_thresholds, variable_probs, var, threshold
            )
            if min(q_forward, q_reverse, old_prior, new_prior) <= 0:
                self.diagnostics["failure"] = "zero_change_probability"
                return False
            success = tree.change_split(node_id, var, threshold)
            if not success:
                self.diagnostics["failure"] = "empty_descendant"
                return False
            self.log_tran_ratio = (
                math.log(new_prior)
                - math.log(old_prior)
                + math.log(q_reverse)
                - math.log(q_forward)
            )
            self.diagnostics.update(
                {
                    "component": component,
                    "node_id": int(node_id),
                    "old_var": int(old_var),
                    "new_var": int(var),
                    "q_forward": float(q_forward),
                    "q_reverse": float(q_reverse),
                }
            )
            return True

        node_id = fast_choice(generator, tree.split_nodes)
        var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s, cum_weights=self.s_cumsum)
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
        if self.proposal_kernel == INFORMED_V1:
            tree = self.current.trees[self.trees_changed[0]]
            self.swappable_pairs = _valid_swappable_pairs(tree)
            self.idx = 0
        else:
            self._ini_swappable_pairs()
        return self._num_possible_proposals > 0

    def try_propose(self, proposed, generator):
        if self.proposal_kernel == INFORMED_V1:
            n_forward = len(self.swappable_pairs)
            parent_id, child_id = self.swappable_pairs[
                int(generator.integers(0, n_forward))
            ]
            tree = proposed.trees[self.trees_changed[0]]
            if not tree.swap_split(parent_id, child_id):
                self.diagnostics["failure"] = "enumerated_pair_became_invalid"
                return False
            reverse_pairs = _valid_swappable_pairs(tree)
            n_reverse = len(reverse_pairs)
            if n_reverse <= 0 or (parent_id, child_id) not in reverse_pairs:
                self.diagnostics["failure"] = "inverse_swap_missing"
                return False
            self.log_tran_ratio = math.log(n_forward) - math.log(n_reverse)
            self.diagnostics.update(
                {
                    "parent_id": int(parent_id),
                    "child_id": int(child_id),
                    "valid_pairs_forward": int(n_forward),
                    "valid_pairs_reverse": int(n_reverse),
                    "q_forward": 1.0 / float(n_forward),
                    "q_reverse": 1.0 / float(n_reverse),
                }
            )
            return True

        if self.idx == 0: # Shuffle the pairs once at the start
            generator.shuffle(self.swappable_pairs)
            
        parent_id, child_id = self.swappable_pairs[self.idx]
        tree = proposed.trees[self.trees_changed[0]]
        success = tree.swap_split(parent_id, child_id)  # If no empty leaves are created
        self.idx += 1
        return success
    
class MultiGrow(Grow):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None,
                 n_samples_list=[10, 5], temp: float = 1.0, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        self.temp = temp
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)
    
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)

        candidates = []
        for _ in range(n_samples):
            node_id = fast_choice(generator, tree.leaves)
            var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
            threshold = fast_choice(generator, self.possible_thresholds[var])
            
            # Use the combined simulation function instead of copy + split_leaf
            new_leaf_ids, new_n, new_vars = tree.simulate_split_leaf(node_id, var, threshold)

            # Check if split is valid (both children have samples)
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            if new_n[left_child] > 0 and new_n[right_child] > 0:
                # Calculate likelihood using simulated data
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
                )
                
                # Calculate prior using simulated data
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                
                # log_likelihood already uses temper-then-marginalize
                log_pi = log_likelihood + log_prior
                candidates.append((node_id, var, threshold, 0.5*float(log_pi)))
            else:
                # Invalid split - set weight to 0 (log weight to -inf)
                candidates.append((node_id, var, threshold, -np.inf))

        log_bwd_weights = np.array([w for _, _, _, w in candidates])

        # Check if all weights are -inf (all candidates invalid)
        if np.all(log_bwd_weights == -np.inf):
            return False
        
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
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
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
                 likelihood=None, tree_prior=None, data_y=None,
                 n_samples_list=[10, 5], temp: float = 1.0, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        self.temp = temp
        if possible_thresholds is None:
            raise ValueError("possible_thresholds must be provided for MultiPrune.")
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)
        
    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)
        all_candidates = tree.terminal_split_nodes

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
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
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
            new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
        )
        log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
        log_pi = log_likelihood + log_prior
        log_fwd_weights.append(0.5 * float(log_pi))

        for _ in range(n_samples - 1):
            node_id = fast_choice(generator, tree.leaves)
            var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
            threshold = fast_choice(generator, self.possible_thresholds[var])
            new_leaf_ids, new_n, new_vars = tree.simulate_split_leaf(node_id, var, threshold)
            left_child = node_id * 2 + 1
            right_child = node_id * 2 + 2
            if new_n[left_child] > 0 and new_n[right_child] > 0:
                log_likelihood = self.likelihood.calculate_simulated_likelihood(
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
                )
                log_prior = self.tree_prior.calculate_simulated_prior(new_vars)
                log_pi = log_likelihood + log_prior
                log_fwd_weights.append(0.5 * float(log_pi))
            else:
                log_fwd_weights.append(-np.inf)  # Invalid split

        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + log_tran_fwd + np.log(fwd_weights.sum()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return True
    
class MultiChange(Change):
    def __init__(self, current, trees_changed, possible_thresholds, tol=100,
                 likelihood=None, tree_prior=None, data_y=None,
                 n_samples_list=[10, 5], temp: float = 1.0, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        self.temp = temp
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        residuals = self.data_y - proposed.evaluate(all_except=self.trees_changed)
        eps_sigma2 = self.current.global_params["eps_sigma2"][0]
        n_samples = self.get_n_samples(tree)

        candidates = []
        n_candidate_trials = 0
        for _ in range(n_samples):
            node_id = fast_choice(generator, tree.split_nodes)
            var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
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
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
                )
                log_pi = log_likelihood
                candidates.append((node_id, var, threshold, 0.5*float(log_pi)))
            else:
                candidates.append((node_id, var, threshold, -np.inf))

        log_bwd_weights = np.array([w for _, _, _, w in candidates])

        # Check if all weights are -inf (all candidates invalid)
        if np.all(log_bwd_weights == -np.inf):
            return False

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
            new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
        )
        log_pi = log_likelihood
        log_fwd_weights.append(0.5*float(log_pi))

        for _ in range(n_samples - 1):
            node_id = fast_choice(generator, tree.split_nodes)
            var = fast_choice_with_weights(generator, np.arange(tree.dataX.shape[1]), weights=self.s)
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
                    new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
                )

                log_pi = log_likelihood
                log_fwd_weights.append(0.5*float(log_pi))
            else:
                log_fwd_weights.append(-np.inf)  # Invalid change

        log_fwd_weights = np.array(log_fwd_weights)
        log_weight_x = log_fwd_weights[[0]]
        max_log_fwd = np.max(log_fwd_weights)
        fwd_weights = np.exp(log_fwd_weights - max_log_fwd)
        log_p_fwd = log_weight_x + np.log(fwd_weights.sum()) + max_log_fwd

        self.log_tran_ratio = log_p_bwd - log_p_fwd
        return success

class MultiSwap(Swap):
    def __init__(self, current, trees_changed, possible_thresholds=None, tol=100,
                 likelihood=None, tree_prior=None, data_y=None,
                 n_samples_list=[10, 5], temp: float = 1.0, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples_list = n_samples_list
        self.temp = temp
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
        for _ in range(n_samples):
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
                        break
                if valid:
                    # Calculate likelihood using simulated data
                    log_likelihood = self.likelihood.calculate_simulated_likelihood(
                        new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
                    )
                    log_pi = log_likelihood
                else:
                    log_pi = -np.inf  # Invalid swap
                log_pi_cache[cache_key] = log_pi
            candidates.append((parent_id, child_id, 0.5*float(log_pi)))

        log_bwd_weights = np.array([w for _, _, w in candidates])

        # Check if all weights are -inf (all candidates invalid)
        if np.all(log_bwd_weights == -np.inf):
            return False

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
            new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
        )
        log_pi = log_likelihood
        log_fwd_weights.append(0.5*float(log_pi))
        log_fwd_pi_cache[(parent_id, child_id)] = log_pi

        for _ in range(n_samples - 1):
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
                        break
                if valid:
                    log_likelihood = self.likelihood.calculate_simulated_likelihood(
                        new_leaf_ids, new_n, residuals, eps_sigma2=eps_sigma2, temp=self.temp
                    )
                    log_pi = log_likelihood
                else:
                    log_pi = -np.inf  # Invalid swap
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
