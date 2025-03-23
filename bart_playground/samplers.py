import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable, Optional
from .params import Tree, Parameters
from .moves import all_moves
from .util import Dataset
from .priors import ComprehensivePrior

class TemperatureSchedule:
    def __init__(self, temp_schedule: Callable[[int], int] = lambda x: 1):
        self.temp_schedule = temp_schedule

    def __call__(self, t):
        return self.temp_schedule(t)

class Sampler(ABC):
    def __init__(self, prior, proposal_probs: dict,
                 generator: np.random.Generator, temp_schedule: TemperatureSchedule = TemperatureSchedule()):
        self._data: Optional[Dataset] = None
        self.prior = prior
        self.n_iter = None
        self.proposals = proposal_probs
        self.temp_schedule = temp_schedule
        self.trace = []
        self.generator = generator
        self.moves_cache = None
        self.moves_cache_iterator = None

    @property
    def data(self) -> Dataset:
        assert self._data, "Data has not been added yet."
        return self._data

    def add_data(self, data: Dataset):
        self._data = data

    def add_thresholds(self, thresholds):
        self.possible_thresholds = thresholds

    def run(self, n_iter, progress_bar=True, quietly=False, current=None):
        if quietly:
            progress_bar = False
        self.trace = []
        self.n_iter = n_iter
        if current is None:
            current = self.get_init_state()
        self.trace.append(current)
        iterator = tqdm(range(n_iter), desc="Iterations") if progress_bar else range(n_iter)
        for iter in iterator:
            if not progress_bar and iter % 10 == 0 and not quietly:
                print(f"Running iteration {iter}/{n_iter}")
            temp = self.temp_schedule(iter)
            current = self.one_iter(current, temp, return_trace=False)
            current.clear_cache()
            self.trace.append(current)
        return self.trace

    def sample_move(self):
        if self.moves_cache is None or self.moves_cache_iterator is None:
            moves_list = list(self.proposals.keys())
            move_probs = list(self.proposals.values())
            self.moves_cache = [all_moves[move] for move in self.generator.choice(moves_list, size=100, p=move_probs)]
            self.moves_cache_iterator = 0
        move = self.moves_cache[self.moves_cache_iterator]
        self.moves_cache_iterator += 1
        if self.moves_cache_iterator >= len(self.moves_cache):
            self.moves_cache = None
        return move

    @abstractmethod
    def get_init_state(self):
        pass

    @abstractmethod
    def one_iter(self, current, temp, return_trace=False):
        pass

    def continue_run(self, additional_iters, new_data=None, quietly=False, last_state=None):
        if last_state is None:
            if hasattr(self, 'trace') and self.trace:
                last_state = self.trace[-1]
            else:
                raise ValueError("No last_state provided and no trace available")
        if new_data is not None:
            old_n = self.data.n
            new_n = new_data.n
            self.add_data(new_data)
            if new_n > old_n:
                new_X = new_data.X[old_n:]
                if hasattr(new_data, 'Z'):
                    new_z = new_data.Z[old_n:]
                    current_state = last_state.add_data_points(new_X, new_z)
                else:
                    current_state = last_state.add_data_points(new_X)
            else:
                current_state = last_state
        else:
            current_state = last_state
        return self.run(additional_iters, quietly=quietly, current=current_state)

class DefaultSampler(Sampler):
    def __init__(self, prior: ComprehensivePrior, proposal_probs: dict,
                 generator: np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100,
                 marginalize: bool = False):
        self.tol = tol
        self.marginalize = marginalize
        if proposal_probs is None:
            proposal_probs = {"grow": 0.5, "prune": 0.5}
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood
        super().__init__(prior, proposal_probs, generator, temp_schedule)

    def get_init_state(self) -> Parameters:
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        trees = [Tree.new(self.data.X) for _ in range(self.tree_prior.n_trees)]
        global_params = self.global_prior.init_global_params(self.data)
        init_state = Parameters(trees, global_params, cache=None)
        # Initialize cache as sum over evaluations.
        init_state.cache = np.sum([tree.evaluate() for tree in init_state.trees], axis=0)
        return init_state

    def log_mh_ratio(self, move, marginalize: bool = False):
        return self.tree_prior.trees_log_prior_ratio(move) + \
               self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize) + \
               move.log_tran_ratio

    def one_iter(self, current, temp, return_trace=False):
        iter_current = current  # Use current state
        if iter_current.cache is None:
            iter_current.cache = np.sum([tree.evaluate() for tree in iter_current.trees], axis=0)
        for k in range(self.tree_prior.n_trees):
            move = self.sample_move()(iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol)
            if move.propose(self.generator):
                new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y=self.data.y, tree_ids=[k])
                move.proposed.update_leaf_vals([k], new_leaf_vals)
                # Instead of copying all trees, only copy tree k.
                iter_current.update_tree(k, move.proposed.trees[k])
                iter_current = iter_current.copy(modified_tree_ids=[k])
        iter_current.global_params = self.global_prior.resample_global_params(iter_current, data_y=self.data.y)
        if return_trace:
            return [(k, iter_current) for k in range(self.tree_prior.n_trees)]
        else:
            return iter_current
