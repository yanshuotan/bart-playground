import numpy as np
from tqdm import tqdm

from params import TreeParams, BARTParams
from moves import all_moves

class Sampler:
    """
    Base class for the BART sampler.
    """
    def __init__(self, X, y, prior, n_iter: int, proposal_probs: dict,  
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
        self.X = X
        self.y = y
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
    
    def get_init_state(self):
        pass

    def one_iter(self, temp):
        """
        Perform one iteration of the sampler.
        """
        pass

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(self, X, y, prior, n_iter: int, proposal_probs: dict,
                 generator : np.random.Generator, n_trees):
        self.n_trees = n_trees
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        super().__init__(X, y, prior, n_iter, proposal_probs, None, generator)

    def get_init_state(self):
        trees = [TreeParams() for _ in range(self.n_trees)]
        Z = self.generator.uniform(0, 1)
        sigma2 = self.prior.sigma2_icdf(Z) # Change to add hyperparameters
        init_state = BARTParams(trees, sigma2, self.prior, self.X, self.y)
        return init_state

    def one_iter(self, temp=1):
        """
        Perform one iteration of the sampler.
        """
        iter_trace = [self.current]
        self.iter_current = self.current
        for k in range(self.n_trees):
            move = self.sample_move()(self.current, [k])
            move.propose(self.generator)
            Z = self.generator.uniform(0, 1)
            if Z < np.exp(temp * move.get_log_MH_ratio()):
                move.proposed.resample_leaf_params([k])
                iter_trace.append(move.proposed)
                self.iter_current = move.proposed
            else:
                iter_trace.append(move.current)
        self.iter_current.resample_sigma2()
        return self.iter_current
    
all_samplers = {"default" : DefaultSampler}