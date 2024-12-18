class Sampler:
    """
    Base class for the BART sampler.
    """
    def __init__(self, ndpost: int, nskip: int, proposal_probs: dict, temperature_schedule: np.ndarray):
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
        self.proposal_probs = proposal_probs
        self.temperature_schedule = temperature_schedule

    def one_iter(self):
        """
        Perform one iteration of the sampler.
        """
        pass

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def one_iter(self):
        """
        Perform one iteration of the sampler.
        """
        pass

    def update_one_tree(self):
        """
        Update a single tree in the model.
        """
        pass