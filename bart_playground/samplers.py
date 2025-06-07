import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .params import Tree, Parameters
from .moves import all_moves
from .util import Dataset
from .priors import *
from .priors import *
from bart_playground import moves
class TemperatureSchedule:

    def __init__(self, temp_schedule: Callable[[int], int] = lambda x: 1):
        self.temp_schedule = temp_schedule
    
    def __call__(self, t):
        return self.temp_schedule(t)
    
class Sampler(ABC):
    """
    Base class for the BART sampler.
    """
    def __init__(self, prior, proposal_probs: dict,  
                 generator : np.random.Generator, temp_schedule: TemperatureSchedule = TemperatureSchedule()):
        """
        Initialize the sampler with the given parameters.

        Parameters:
            prior: The prior distribution.
            proposal_probs (dict): A dictionary containing proposal probabilities.
            generator (np.random.Generator): A random number generator.
            temp_schedule (TemperatureSchedule): Temperature schedule for the sampler.

        Attributes:
            data: Placeholder for data, initially set to None.
            prior: The prior distribution.
            n_iter: Number of iterations, initially set to None.
            proposals (dict): A dictionary containing proposal probabilities.
            temp_schedule (TemperatureSchedule): Temperature schedule for the sampler.
            trace (list): A list to store the trace of the sampling process.
            generator (np.random.Generator): A random number generator.
        """
        self._data : Optional[Dataset] = None
        self.prior = prior
        self.n_iter = None
        self.proposals = proposal_probs
        self.temp_schedule = temp_schedule
        self.trace = []
        self.generator = generator
        # create cache for moves
        self.moves_cache = None
        # current move cache iterator
        self.moves_cache_iterator = None
        
    @property
    def data(self) -> Dataset:
        assert self._data, "Data has not been added yet."
        return self._data
    
    def add_data(self, data : Dataset):
        """
        Adds data to the sampler.

        Parameters:
        data (Dataset): The data to be added to the sampler.
        """
        self._data = data

    def add_thresholds(self, thresholds):
        self.possible_thresholds = thresholds
        
    def run(self, n_iter, progress_bar = True, quietly = False, current = None, n_skip = 0):
        """
        Run the sampler for a specified number of iterations from `current` or a fresh start.

        Parameters:
        n_iter (int): The number of iterations to run the sampler.
        """
        if quietly:
            progress_bar = False

        # Determine the actual starting state for this MCMC run
        current: Parameters
        if current is not None:
            current = current
        elif self.trace:  # If self.trace is already populated (e.g., by init_from_xgboost)
            current = self.trace[0]  # Use the pre-loaded state
        else:
            current = self.get_init_state() # Otherwise, generate a new initial state
        
        # This run will generate its own trace, stored locally first.
        self.trace = []
        self.n_iter = n_iter # n_iter for this specific run (might be different from previous ndpost)

        # If not skipping any iterations, the initial state (before any MCMC steps of this run)
        # is recorded.
        if n_skip == 0:
            # Make a copy because current_mcmc_iter_state will be modified by one_iter
            self.trace.append(current)

        iterator = tqdm(range(n_iter), desc="Iterations") if progress_bar else range(n_iter)

        for iter in iterator:
            if not progress_bar and iter % 10 == 0 and not quietly:
                print(f"Running iteration {iter}/{n_iter}")
            
            temp = self.temp_schedule(iter)
            current = self.one_iter(current, temp, return_trace=False)

            if iter >= n_skip:
                # Clear cache of the *previous* state in the trace before adding the new one
                if len(self.trace) > 0:
                    self.trace[-1].clear_cache()
                # Append a copy of the state *after* the current iteration
                self.trace.append(current)
        
        return self.trace
    
    def sample_move(self):
        """
        Samples a move based on the proposal probabilities.

        This method selects a move from the proposals dictionary, where the keys
        are the possible moves and the values are the corresponding probabilities.
        It uses the generator's choice method to randomly select a move according
        to these probabilities.

        Returns:
            The selected move from the all_moves list based on the sampled index.
        """
        if self.moves_cache is None or self.moves_cache_iterator is None:
            moves = list(self.proposals.keys())
            move_probs = list(self.proposals.values())
            self.moves_cache = [all_moves[move] for move in self.generator.choice(moves, size=100, p=move_probs)]
            self.moves_cache_iterator = 0
        move = self.moves_cache[self.moves_cache_iterator]
        self.moves_cache_iterator += 1
        if self.moves_cache_iterator >= len(self.moves_cache):
            self.moves_cache = None
        return move
    
    @abstractmethod
    def get_init_state(self):
        """
        Retrieve the initial state for the sampler.

        This method should be overridden by subclasses to provide the specific
        initial state required for the sampling process.

        Returns:
            The initial state for the sampler.
        """
        pass

    @abstractmethod
    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        pass

    def continue_run(self, additional_iters, new_data=None, quietly=False, last_state=None):
            """
            Continue sampling with updated data from a previous state.

            Parameters:
                additional_iters: Number of additional iterations
                new_data: Updated dataset (if None, uses existing data)
                quietly: Whether to suppress output
                last_state: Last state from previous run (if None, uses last state in trace)

            Returns:
                New trace segment
            """
            # Get last state
            if last_state is None:
                if hasattr(self, 'trace') and self.trace:
                    last_state = self.trace[-1]
                else:
                    raise ValueError("No last_state provided and no trace available")

            # Update parameter state with any new data points if needed
            if new_data is not None:
                old_n = self.data.n
                new_n = new_data.n
                
                self.add_data(new_data)

                if new_n > old_n:
                    new_X = new_data.X[old_n:]
                    if hasattr(new_data, 'Z'): # check if treatment assignments are available, e.g. for BCFDataset
                        new_z = new_data.Z[old_n:]
                        current_state = last_state.add_data_points(new_X, new_z)
                    else:
                        current_state = last_state.add_data_points(new_X)
                else:
                    current_state = last_state
            else:
                current_state = last_state

            # Run sampler for additional iterations
            return self.run(additional_iters, quietly=quietly, current=current_state)

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(
        self,
        prior: ComprehensivePrior,
        proposal_probs: dict,
        generator: np.random.Generator,
        temp_schedule=TemperatureSchedule(),
        tol: int = 100,
        init_trees: Optional[list[Tree]] = None  # NEW
    ):
        """
        Default implementation of the BART sampler.
        Accepts an optional list of pre-initialized trees without changing default behavior.
        """
        # preserve original default proposal behavior
        self.tol = tol
        if proposal_probs is None:
            proposal_probs = {"grow": 0.5, "prune": 0.5}

        # original prior unpacking
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood

        # initialize base sampler
        super().__init__(prior, proposal_probs, generator, temp_schedule)

        # store seed forest for XGBoost init
        self.init_trees = init_trees

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.
        If init_trees was provided, copy up to n_trees of them and
        pad the rest with fresh stumps; otherwise build all new stumps.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        N = self.tree_prior.n_trees

        if self.init_trees is not None:
            provided = len(self.init_trees)
            # Copy up to N of the provided trees
            trees = [t.copy() for t in self.init_trees[:N]]
            # Pad with fresh stumps if fewer than N
            if provided < N:
                trees += [Tree.new(self.data.X) for _ in range(N - provided)]
        else:
            trees = [Tree.new(self.data.X) for _ in range(N)]

        global_params = self.global_prior.init_global_params(self.data)
        return Parameters(trees, global_params)

    def log_mh_ratio(self, move : Move, temp, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio"""
        return (self.tree_prior.trees_log_prior_ratio(move) + \
            self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize)) / temp + \
            move.log_tran_ratio

    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        iter_current = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]
        for k in range(self.tree_prior.n_trees):
            move = self.sample_move()(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol
                )
            if move.propose(self.generator): # Check if a valid move was proposed
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move, temp):
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    iter_current = move.proposed
                    if return_trace:
                        iter_trace.append((k+1, move.proposed))
        iter_current.global_params = self.global_prior.resample_global_params(iter_current, data_y = self.data.y)
        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current
    
all_samplers = {"default" : DefaultSampler}

default_proposal_probs = {"grow" : 0.25,
                          "prune" : 0.25,
                          "change" : 0.4,
                          "swap" : 0.1}
