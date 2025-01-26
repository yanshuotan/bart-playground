
# bcf_sampler.py

from ..samplers import default_proposal_probs, Sampler, TemperatureSchedule
from ..moves import all_moves
from .bcf_params import BCFParams
from ..params import Tree
from .bcf_prior import BCFPrior
from .bcf_util import BCFDataset, BCFParamSlice

import numpy as np

class BCFSampler(Sampler):
    """Extends sampler for BCF's dual tree ensembles"""
    def __init__(self, prior : BCFPrior, proposal_probs,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        self.proposals_mu = proposal_probs or default_proposal_probs
        self.proposals_tau = proposal_probs or default_proposal_probs
        self.tol = tol
        super().__init__(prior, proposal_probs, temp_schedule, generator)
        
    def get_init_state(self):
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        mu_trees = [Tree(self.data) for _ in range(self.prior.mu_prior.n_trees)]
        tau_trees = [Tree(self.data) for _ in range(self.prior.tau_prior.n_trees)]
        global_params = self.prior.init_global_params(self.data)

        init_state = BCFParams(mu_trees, tau_trees, global_params, self.data)
        return init_state

    def sample_move(self, ensemble_type):
        """Sample move type for specified tree ensemble"""
        if ensemble_type == 'mu':
            moves = list(self.proposals_mu.keys())
            probs = list(self.proposals_mu.values())
        else:
            moves = list(self.proposals_tau.keys())
            probs = list(self.proposals_tau.values())
        return all_moves[self.generator.choice(moves, p=probs)]
    
    def one_iter(self, return_trace=False):
        """One MCMC iteration: Update μ trees -> τ trees -> global params"""
        if self.current is None:
            self.current = self.get_init_state()
        iter_current = self.current
        iter_trace = [iter_current] if return_trace else None
        temp = self.temp_schedule(iter_current)

        # 1) Update mu (prognostic) ensemble
        for k in range(self.prior.mu_prior.n_trees):
            move_class = self.sample_move("mu")  
            move = move_class(
                current=BCFParamSlice(iter_current, "mu"), trees_changed=[k], tol=self.tol
            )
            move.propose(self.generator)

            # Metropolis–Hastings
            Z = self.generator.uniform(0,1)
            if Z < np.exp(temp * move.get_log_MH_ratio()):
                new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, 'mu', [k])
                # TODO: ???
                move.proposed.update_leaf_vals('mu', [k], new_leaf_vals)
                iter_current = move.proposed.bcf_params
            else:
                pass

            if return_trace:
                iter_trace.append(iter_current)

        # 2) Update tau (treatment effect) ensemble
        for k in range(self.prior.tau_prior.n_trees):
            move_class = self.sample_move("tau")  
            move = move_class(
                current=BCFParamSlice(iter_current, "tau"), trees_changed=[k], tol=self.tol
            )
            move.propose(self.generator)

            # Metropolis–Hastings
            Z = self.generator.uniform(0,1)
            if Z < np.exp(temp * move.get_log_MH_ratio()):
                new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, 'tau', [k])
                # TODO: ???
                move.proposed.update_leaf_vals('tau', [k], new_leaf_vals)
                iter_current = move.proposed.bcf_params
            else:
                pass

            if return_trace:
                iter_trace.append(iter_current)

        # 3) Resample global parameters
        new_globals = self.prior.resample_global_params(iter_current)
        iter_current = iter_current.update_global_params(new_globals)

        # self.current = iter_current

        if return_trace:
            return iter_trace
        else:
            return iter_current
    