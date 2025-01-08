import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from .params import Tree, Parameters
from .moves import all_moves, Move
from .util import Dataset

class Sampler(ABC):
    """
    Base class for the BART sampler.
    """
    def __init__(self, prior, proposal_probs: dict,  
                 generator : np.random.Generator, temp_schedule: np.ndarray):
        """
        Initialize the sampler with the given parameters.

        Parameters:
            prior: The prior distribution.
            proposal_probs (dict): A dictionary containing proposal probabilities.
            generator (np.random.Generator): A random number generator.
            temp_schedule (np.ndarray): An array representing the temperature schedule. If None, defaults to an array of ones.

        Attributes:
            data: Placeholder for data, initially set to None.
            prior: The prior distribution.
            n_iter: Number of iterations, initially set to None.
            proposals (dict): A dictionary containing proposal probabilities.
            temp_schedule (np.ndarray): An array representing the temperature schedule.
            trace (list): A list to store the trace of the sampling process.
            generator (np.random.Generator): A random number generator.
        """
        self.data = None
        self.prior = prior
        self.n_iter = None
        self.proposals = proposal_probs
        if temp_schedule is None:
            temp_schedule = np.ones(n_iter)
        self.temp_schedule = temp_schedule
        self.trace = []
        self.generator = generator

    def add_data(self, data : Dataset):
        """
        Adds data to the sampler.

        Parameters:
        data (Dataset): The data to be added to the sampler.
        """
        self.data = data

    def run(self, n_iter):
        """
        Run the sampler for a specified number of iterations.

        Parameters:
        n_iter (int): The number of iterations to run the sampler.

        Raises:
        AttributeError: If data has not been added yet.

        """
        self.n_iter = n_iter
        if self.data is None:
            raise AttributeError("Data has not been added yet.")
        self.current = self.get_init_state()
        for iter in tqdm(range(n_iter)):
            self.current = self.one_iter(self.temp_schedule[iter])
            self.trace.append(self.current)
    
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
    def one_iter(self, temp=1, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        pass

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(self, prior, proposal_probs: dict,
                 generator : np.random.Generator, tol=100):
        self.tol = tol
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        super().__init__(prior, proposal_probs, None, generator)

    def get_init_state(self):
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        trees = [Tree() for _ in range(self.n_trees)]
        global_params = self.prior.init_global_params(self.data.X, self.data.y)
        init_state = Parameters(trees, global_params, self.data)
        return init_state

    def one_iter(self, temp=1, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        iter_trace = [self.current]
        iter_current = self.current
        for k in range(self.prior.n_trees):
            move = self.sample_move()(self.current, [k], self.tol)
            move.propose(self.generator)
            Z = self.generator.uniform(0, 1)
            if Z < np.exp(temp * move.get_log_MH_ratio()):
                new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, [k])
                move.proposed.update_leaf_params([k], new_leaf_vals)
                iter_trace.append(move.proposed)
                iter_current = move.proposed
            else:
                iter_trace.append(move.current)
        iter_current.global_params = self.prior.resample_global_params(iter_current)
        if return_trace:
            return iter_trace
        else:
            return iter_current
    
all_samplers = {"default" : DefaultSampler}

default_proposal_probs = {"grow" : 0.25,
                          "prune" : 0.25,
                          "change" : 0.4,
                          "swap" : 0.1}