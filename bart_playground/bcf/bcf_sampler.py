
# bcf_sampler.py

from ..samplers import default_proposal_probs, Sampler, TemperatureSchedule
from ..moves import all_moves
from .bcf_params import BCFParams
from ..params import Tree
from .bcf_prior import BCFPrior
from .bcf_util import BCFParamView
from .bcf_dataset import BCFDataset

import numpy as np

class BCFSampler(Sampler):
    """Extends sampler for BCF's dual tree ensembles"""
    def __init__(self, prior : BCFPrior, proposal_probs,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        self.proposals_mu = proposal_probs or default_proposal_probs
        self.proposals_tau = proposal_probs or default_proposal_probs
        self.tol = tol
        super().__init__(prior, proposal_probs, generator, temp_schedule)
        
    def add_data(self, data : BCFDataset):
        return super().add_data(data)
    def add_thresholds(self, thresholds):
        return super().add_thresholds(thresholds)
        
    def get_init_state(self):
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        mu_trees = [Tree.new(self.data.X) for _ in range(self.prior.mu_prior.n_trees)]
        tau_trees = [Tree.new(self.data.X[self.data.treated]) for _ in range(self.prior.tau_prior.n_trees)]
        global_params = self.prior.init_global_params(self.data)

        init_state = BCFParams(mu_trees, tau_trees, global_params)
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
    
    def create_views(self, bcfp : BCFParams):
        raise Exception("create_views should not be called now")
        if not bcfp.mu_view:
            bcfp.mu_view = BCFParamView(bcfp, "mu", None)
        if not bcfp.tau_view:
            bcfp.tau_view = BCFParamView(bcfp, "tau", None)
    
    def one_iter(self, current : BCFParams, temp, return_trace=False):
        """One MCMC iteration: Update μ trees -> τ trees -> global params"""
        iter_current = current
        
        iter_trace = [iter_current] if return_trace else None
        temp = self.temp_schedule(iter_current)
        self.prior : BCFPrior

        # 1) Update mu (prognostic) ensemble
        remaining_y = self.data.y
        remaining_y[self.data.treated] -= iter_current.tau_view.evaluate()
        for k in range(self.prior.mu_prior.n_trees):
            move_class = self.sample_move("mu")  
            move = move_class(
                current=iter_current.mu_view, trees_changed=[k], possible_thresholds = self.possible_thresholds, tol=self.tol
            )
            if move.propose(self.generator): # Check if a valid move was proposed
                # Metropolis–Hastings
                Z = self.generator.uniform(0,1)
                # TODO
                if Z < np.exp(temp * self.prior.trees_log_mh_ratio(move, data_y = remaining_y, ensemble_id = 'mu')):
                    new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, data_y = remaining_y, ensemble_id = 'mu', tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    iter_current = move.proposed.bcf_params

            if return_trace:
                iter_trace.append(iter_current)

        # 2) Update tau (treatment effect) ensemble
        remaining_y = (self.data.y - iter_current.mu_view.evaluate())[self.data.treated]
        from ..util import DefaultPreprocessor
        prep = DefaultPreprocessor()
        thresholds_treated = prep.gen_thresholds(self.data.X[self.data.treated])
        for k in range(self.prior.tau_prior.n_trees):
            move_class = self.sample_move("tau")  
            move = move_class(
                # TODO: Check if this is correct
                current=iter_current.tau_view, trees_changed=[k], possible_thresholds = thresholds_treated, tol=self.tol
            )
            if move.propose(self.generator): # Check if a valid move was proposed
                # Metropolis–Hastings
                Z = self.generator.uniform(0,1)
                if Z < np.exp(temp * self.prior.trees_log_mh_ratio(move, data_y = remaining_y, ensemble_id = 'tau')):
                    new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, data_y = remaining_y, ensemble_id = 'tau', tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    iter_current = move.proposed.bcf_params

            if return_trace:
                iter_trace.append(iter_current)

        # 3) Resample global parameters
        iter_current.global_params = self.prior.resample_global_params(iter_current, data_y = self.data.y, treated=self.data.treated)

        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current
    