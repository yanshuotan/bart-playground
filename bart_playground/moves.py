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
        super().__init__(current, trees_changed, tol = tol, **kwargs)
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
    
class InformedGrow(Grow):
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
                    all_candidates.append((node_id, var, threshold))

        if len(all_candidates) > self.n_samples:
            idxs = generator.choice(len(all_candidates), size=self.n_samples, replace=False)
            sampled_candidates = [all_candidates[i] for i in idxs]
        else:
            sampled_candidates = all_candidates
        
        candidates = []
        for node_id, var, threshold in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            if temp_tree.split_leaf(node_id, var, threshold):
                self.proposed = temp
                log_pi_ratio = self.likelihood.trees_log_marginal_lkhd_ratio(
                    self, self.data_y
                ) + self.tree_prior.trees_log_prior_ratio(self)
                weight = np.exp(log_pi_ratio) / (1 + np.exp(log_pi_ratio))
                candidates.append((node_id, var, threshold, weight))
                del self.proposed

        if not candidates:
            return False

        weights = np.array([w for _, _, _, w in candidates]).ravel()
        weights /= weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, var, threshold, _ = candidates[idx]

        tree = proposed.trees[self.trees_changed[0]]
        success = tree.split_leaf(node_id, var, threshold)
        self.log_tran_ratio = 0 #TODO: Calculate the log transition ratio
        return success
    
class InformedPrune(Prune):
    def __init__(self, current, trees_changed, possible_thresholds=None, tol=100,
                 likelihood=None, tree_prior=None, data_y=None, n_samples=10, **kwargs):
        self.likelihood = likelihood
        self.tree_prior = tree_prior
        self.data_y = data_y
        self.n_samples = n_samples
        super().__init__(current, trees_changed, possible_thresholds, tol, **kwargs)

    def try_propose(self, proposed, generator):
        tree = proposed.trees[self.trees_changed[0]]
        all_candidates = list(tree.terminal_split_nodes)

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
            self.proposed = temp
            log_pi_ratio = self.likelihood.trees_log_marginal_lkhd_ratio(
                self, self.data_y
            ) + self.tree_prior.trees_log_prior_ratio(self)
            weight = np.exp(log_pi_ratio) / (1 + np.exp(log_pi_ratio))
            candidates.append((node_id, weight))
            del self.proposed

        if not candidates:
            return False

        weights = np.array([w for _, w in candidates]).ravel()
        weights /= weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, _ = candidates[idx]

        tree = proposed.trees[self.trees_changed[0]]
        tree.prune_split(node_id)

        self.log_tran_ratio = 0 #TODO: Calculate the log transition ratio
        return True
    
class InformedChange(Change):
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
                    all_candidates.append((node_id, var, threshold))


        if len(all_candidates) > self.n_samples:
            idxs = generator.choice(len(all_candidates), size=self.n_samples, replace=False)
            sampled_candidates = [all_candidates[i] for i in idxs]
        else:
            sampled_candidates = all_candidates

        candidates = []
        for node_id, var, threshold in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            if temp_tree.change_split(node_id, var, threshold):
                self.proposed = temp
                log_pi_ratio = self.likelihood.trees_log_marginal_lkhd_ratio(
                    self, self.data_y
                ) + self.tree_prior.trees_log_prior_ratio(self)
                weight = np.exp(log_pi_ratio) / (1 + np.exp(log_pi_ratio))
                candidates.append((node_id, var, threshold, weight))
                del self.proposed

        if not candidates:
            return False

        weights = np.array([w for _, _, _, w in candidates]).ravel()
        weights /= weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        node_id, var, threshold, _ = candidates[idx]

        tree = proposed.trees[self.trees_changed[0]]
        success = tree.change_split(node_id, var, threshold)
        self.log_tran_ratio = 0
        return success

class InformedSwap(Swap):
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
                all_candidates.append((parent_id, child_id))

        if len(all_candidates) > self.n_samples:
            idxs = generator.choice(len(all_candidates), size=self.n_samples, replace=False)
            sampled_candidates = [all_candidates[i] for i in idxs]
        else:
            sampled_candidates = all_candidates

        candidates = []
        for parent_id, child_id in sampled_candidates:
            temp = proposed.copy(self.trees_changed)
            temp_tree = temp.trees[self.trees_changed[0]]
            if temp_tree.swap_split(parent_id, child_id):
                self.proposed = temp
                log_pi_ratio = self.likelihood.trees_log_marginal_lkhd_ratio(
                    self, self.data_y
                ) + self.tree_prior.trees_log_prior_ratio(self)
                weight = np.exp(log_pi_ratio) / (1 + np.exp(log_pi_ratio))
                candidates.append((parent_id, child_id, weight))
                del self.proposed

        if not candidates:
            return False

        weights = np.array([w for _, _, w in candidates]).ravel()
        weights /= weights.sum()
        idx = generator.choice(len(candidates), p=weights)
        parent_id, child_id, _ = candidates[idx]

        tree = proposed.trees[self.trees_changed[0]]
        success = tree.swap_split(parent_id, child_id)
        self.log_tran_ratio = 0
        return success

 
all_moves = {"grow" : Grow,
            "prune" : Prune,
            "change" : Change,
            "swap" : Swap,
            "informed_grow": InformedGrow,
            "informed_prune": InformedPrune,
            "informed_change": InformedChange,
            "informed_swap": InformedSwap}