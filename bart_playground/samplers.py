
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .params import Tree, Parameters
from .moves import all_moves
from .util import Dataset
from .priors import *

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

    def run(self, n_iter, progress_bar = True):
        """
        Run the sampler for a specified number of iterations.

        Parameters:
        n_iter (int): The number of iterations to run the sampler.
        """
        self.trace = []
        self.n_iter = n_iter
        current = self.get_init_state()
        self.trace.append(current) # Add initial state to trace
        
        iterator = tqdm(range(n_iter), desc="Iterations") if progress_bar else range(n_iter)
    
        for iter in iterator:
            if not progress_bar and iter % 10 == 0:
                print(f"Running iteration {iter}/{n_iter}")
            # print(self.temp_schedule)
            temp = self.temp_schedule(iter)
            current = self.one_iter(current, temp, return_trace=False)
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
        moves = list(self.proposals.keys())
        move_probs = list(self.proposals.values())
        return all_moves[self.generator.choice(moves, p=move_probs)]
    
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

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(self, prior : ComprehensivePrior, proposal_probs: dict,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        self.tol = tol
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood
        super().__init__(prior, proposal_probs, generator, temp_schedule)

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        trees = [Tree.new(self.data.X) for _ in range(self.tree_prior.n_trees)]
        global_params = self.global_prior.init_global_params(self.data)
        init_state = Parameters(trees, global_params)
        return init_state
    
    def log_mh_ratio(self, move : Move, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio"""
        return self.tree_prior.trees_log_prior_ratio(move) + \
            self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize) + \
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
                if Z < np.exp(temp * self.log_mh_ratio(move)):
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
        
class NTreeSampler(Sampler):
    """
    Change the number of trees implementation of the BART sampler.
    """
    def __init__(self, prior : ComprehensivePrior, proposal_probs: dict,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), break_prob : float = 0.5, tol=100):
        self.break_prob = break_prob
        self.tol = tol
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood
        self.tree_num_prior = prior.tree_num_prior
        super().__init__(prior, proposal_probs, generator, temp_schedule)

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        trees = [Tree.new(self.data.X) for _ in range(self.tree_prior.n_trees)]
        global_params = self.global_prior.init_global_params(self.data)
        init_state = Parameters(trees, global_params)
        return init_state
    
    def log_mh_ratio(self, move : Move, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio"""
        return self.tree_prior.trees_log_prior_ratio(move) + \
            self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize) + \
            self.tree_num_prior.tree_num_log_prior_ratio(move)

    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        iter_current = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]

        # Break and Combine
        U = self.generator.uniform(0,1)
        if U < self.break_prob: # Break move
            break_id = [self.generator.choice(np.arange(0,self.tree_prior.n_trees))]
            move = Break(iter_current, break_id, self.tol)   
            if move.propose(self.generator):
                Z = self.generator.uniform(0, 1)
                if Z < np.exp(temp * self.log_mh_ratio(move)):
                    self.tree_prior.n_trees = self.tree_prior.n_trees + 1
                    new_leaf_vals_remain = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = break_id)
                    new_leaf_vals_new = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [-1])
                    move.proposed.update_leaf_vals(break_id, new_leaf_vals_remain)
                    move.proposed.update_leaf_vals([-1], new_leaf_vals_new)
                    move.proposed.update_tree_num()
                    iter_current = move.proposed
                    iter_trace.append((1, move.proposed))
        
        else: # Combine move 
            combine_ids = self.generator.choice(np.arange(0,self.tree_prior.n_trees),size=2, replace=False)
            combine_position = combine_ids[0] if combine_ids[0] < combine_ids[1] else combine_ids[0] - 1
            move = Combine(iter_current, combine_ids, self.tol)   
            if move.propose(self.generator):
                Z = self.generator.uniform(0, 1)
                if Z < np.exp(temp * self.log_mh_ratio(move)):
                    self.tree_prior.n_trees = self.tree_prior.n_trees - 1
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [combine_position])
                    move.proposed.update_leaf_vals([combine_position], new_leaf_vals)
                    move.proposed.update_tree_num()
                    iter_current = move.proposed
                    iter_trace.append((1, move.proposed))

        # Default BART
        for k in range(self.tree_prior.n_trees):
            move = self.sample_move()(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol
                )
            if move.propose(self.generator): # Check if a valid move was proposed
                Z = self.generator.uniform(0, 1)
                if Z < np.exp(temp * self.log_mh_ratio(move)):
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
    
all_samplers = {"default" : DefaultSampler,
                "ntree": NTreeSampler}

default_proposal_probs = {"grow" : 0.25,
                          "prune" : 0.25,
                          "change" : 0.4,
                          "swap" : 0.1}
