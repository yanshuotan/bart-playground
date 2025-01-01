import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from params import Tree, Parameters
from moves import all_moves, Move

class Sampler(ABC):
    """
    Base class for the BART sampler.
    """
    def __init__(self, data, prior, n_iter: int, proposal_probs: dict,  
                 generator : np.random.Generator, temp_schedule: np.ndarray):
        """
        Initialize the sampler.

        Parameters:
        - ndpost: int
            Number of posterior samples to draw.
        - nskip: int
            Number of samples to skip.
        - proposal_probs: dict
            Probabilities for each type of move.
        - temperature_schedule: np.ndarray
            Schedule of temperatures for annealing.
        """
        self.data = data
        self.prior = prior
        self.n_iter = n_iter
        self.proposals = proposal_probs
        if temp_schedule is None:
            temp_schedule = np.ones(n_iter)
        self.temp_schedule = temp_schedule
        self.trace = []
        self.generator = generator

    def run(self):
        self.current = self.get_init_state()
        for iter in tqdm(range(self.n_iter)):
            self.current = self.one_iter(self.temp_schedule[iter])
            self.trace.append(self.current)
    
    def sample_move(self):
        moves = list(self.proposals.keys())
        move_probs = list(self.proposals.values())
        return all_moves[self.generator.choice(moves, p=move_probs)]
    
    @abstractmethod
    def get_init_state(self):
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
    def __init__(self, data, prior, n_iter: int, proposal_probs: dict,
                 generator : np.random.Generator, n_trees, tol=100):
        self.n_trees = n_trees
        self.tol = tol
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        super().__init__(data, prior, n_iter, proposal_probs, None, generator)

    def get_init_state(self):
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
        for k in range(self.n_trees):
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