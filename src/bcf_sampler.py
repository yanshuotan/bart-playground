
from .samplers import DefaultSampler, default_proposal_probs
from .moves import all_moves

import numpy as np

class BCFSampler:
    """Extends sampler for BCF's dual tree ensembles"""
    def __init__(self, prior, proposal_probs_mu=None, proposal_probs_tau=None,
                 tau_update_prob=0.5, **kwargs):
        super().__init__(prior, None, **kwargs)
        self.proposals_mu = proposal_probs_mu or default_proposal_probs
        self.proposals_tau = proposal_probs_tau or default_proposal_probs
        self.tau_update_prob = tau_update_prob  # Prob of updating tau trees vs mu trees

    def sample_tree_type(self):
        """Decide whether to update mu or tau trees"""
        return 'tau' if self.generator.uniform() < self.tau_update_prob else 'mu'

    def one_iter(self, temp=1):
        # TODO

        # Update global parameters
        self.current.global_params = self.prior.resample_global_params(self.current)
        return self.current

    def _apply_move(self, move, temp):
        # TODO
        pass
