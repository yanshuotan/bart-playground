import numpy as np

from params import TreeParams, BARTParams
from moves import all_moves, Move, Grow, Prune, Change, Swap


class Sampler:
    """
    Base class for the BART sampler.
    """
    def __init__(self, ndpost: int, nskip: int, proposal_probs: dict, temperature_schedule: np.ndarray, initialization : Initialization, generator : np.random.Generator):
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
        self.ndpost = ndpost
        self.nskip = nskip
        self.proposals = proposal_probs
        self.temperature_schedule = temperature_schedule
        self.trace = []
        self.initialization = initialization
        self.current = self.initialization.initialize()
        self.generator = generator

    def get_trace(self):
        pass

    def one_iter(self):
        """
        Perform one iteration of the sampler.
        """
        pass

    def sample_move(self):
        moves = list(self.proposals.keys())
        move_probs = list(self.proposals.values())
        return all_moves[self.generator.choice(moves, p=move_probs)]
        

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(self, ndpost: int, nskip: int, proposal_probs: dict, n_trees):
        self.n_trees = n_trees
        super().__init__(ndpost, nskip, proposal_probs, None)

    def one_iter(self, temperature=1):
        """
        Perform one iteration of the sampler.
        """
        iter_trace = [self.current]
        self.iter_current = self.current
        for k in range(self.n_trees):
            move = self.sample_move()(self.current, [k])
            move.propose(self.generator)

    def log_MH_ratio(self, move : Move):
        return move.get_log_prior_ratio() + move.get_log_marginal_lkhd_ratio()

    def update_one_tree(self):
        """
        Update a single tree in the model.
        """
        pass

class Initialization:

    def initialize(self, **kwargs):
        pass


class DefaultInitialization(Initialization):
    """
    Default implementation of the BART sampler.
    """
    def initialize(self, **kwargs):
        trees = [TreeParams() for _ in range(n_trees)]
        init_state = BARTParams(trees, sigma2, prior, X, y)
        return init_state