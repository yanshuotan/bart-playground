# bcf_sampler.py

from typing import Optional
from ..samplers import default_proposal_probs, Sampler, TemperatureSchedule
from ..moves import all_moves
from .bcf_params import BCFParams
from ..params import Tree
from .bcf_prior import BCFPrior
from .bcf_util import BCFEnsembleIndex, EnsembleName, BCFDataset

import numpy as np

class BCFSampler(Sampler):
    """Extends sampler for BCF's dual tree ensembles"""
    def __init__(self, prior : BCFPrior, proposal_probs,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        self.proposals_mu = proposal_probs or default_proposal_probs
        self.proposals_tau = proposal_probs or default_proposal_probs
        self._data : Optional[BCFDataset] = None
        self.tol = tol
        # Initialize move caches for both ensemble types
        self.moves_cache_mu = None
        self.moves_cache_mu_iterator = None
        self.moves_cache_tau = None
        self.moves_cache_tau_iterator = None
        super().__init__(prior, proposal_probs, generator, temp_schedule)
        
    @property
    def data(self) -> BCFDataset:
        assert self._data, "Data has not been added yet."
        return self._data
    
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
        tau_trees_list = [
            [Tree.new(self.data.X[self.data.treated_by(i)]) for _ in range(self.prior.tau_prior_list[i].n_trees)]
            for i in range(self.prior.n_treat_arms)
            ]
        global_params = self.prior.init_global_params(self.data)

        init_state = BCFParams(mu_trees, tau_trees_list, global_params)
        return init_state

    def sample_move(self, ensemble_type : EnsembleName):
        """Sample move type for specified tree ensemble"""
        if ensemble_type == EnsembleName.MU:
            # Use cached moves for mu ensemble
            if self.moves_cache_mu is None or self.moves_cache_mu_iterator is None:
                moves = list(self.proposals_mu.keys())
                probs = list(self.proposals_mu.values())
                self.moves_cache_mu = [all_moves[move] for move in self.generator.choice(moves, size=100, p=probs)]
                self.moves_cache_mu_iterator = 0
            move = self.moves_cache_mu[self.moves_cache_mu_iterator]
            self.moves_cache_mu_iterator += 1
            if self.moves_cache_mu_iterator >= len(self.moves_cache_mu):
                self.moves_cache_mu = None
        else:
            # Use cached moves for tau ensemble
            if self.moves_cache_tau is None or self.moves_cache_tau_iterator is None:
                moves = list(self.proposals_tau.keys())
                probs = list(self.proposals_tau.values())
                self.moves_cache_tau = [all_moves[move] for move in self.generator.choice(moves, size=100, p=probs)]
                self.moves_cache_tau_iterator = 0
            move = self.moves_cache_tau[self.moves_cache_tau_iterator]
            self.moves_cache_tau_iterator += 1
            if self.moves_cache_tau_iterator >= len(self.moves_cache_tau):
                self.moves_cache_tau = None
        return move
    
    def one_iter(self, current : BCFParams, temp, return_trace=False):
        """One MCMC iteration: Update μ trees -> τ trees -> global params"""
        iter_current = current.copy()
        
        iter_trace = [iter_current] if return_trace else None
        temp = self.temp_schedule(iter_current)
        self.prior : BCFPrior

        # 1) Update mu (prognostic) ensemble
        remaining_y = self.data.y.copy()
        for i in range(iter_current.n_treat_arms):
            remaining_y[self.data.treated_by(i)] -= iter_current.tau_view_list[i].evaluate()
        for k in range(self.prior.mu_prior.n_trees):
            move_class = self.sample_move(EnsembleName.MU)  
            move = move_class(
                current=iter_current.mu_view, trees_changed=[k], possible_thresholds = self.possible_thresholds, tol=self.tol
            )
            if move.propose(self.generator): # Check if a valid move was proposed
                # Metropolis–Hastings
                Z = self.generator.uniform(0,1)
                if np.log(Z) < temp * self.prior.trees_log_mh_ratio(move, data_y = remaining_y, ensemble_id = BCFEnsembleIndex(EnsembleName.MU)):
                    new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, data_y = remaining_y, ensemble_id = BCFEnsembleIndex(EnsembleName.MU), tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    # iter_current.mu_view = move.proposed
                    iter_current = move.proposed.bcf_params

            if iter_trace:
                iter_trace.append(iter_current)

        # 2) Update tau (treatment effect) ensemble
        for i in range(iter_current.n_treat_arms):
            
            from ..util import DefaultPreprocessor
            prep = DefaultPreprocessor()
            thresholds_treated = prep.gen_thresholds(self.data.X[self.data.treated_by(i)])

            # Advanced indexing, deep copy
            remaining_y = self.data.y - iter_current.mu_view.evaluate()
            for j in range(iter_current.n_treat_arms):
                if j != i:
                    remaining_y[self.data.treated_by(j)] -= iter_current.tau_view_list[j].evaluate()
            remaining_y = remaining_y[self.data.treated_by(i)]

            for k in range(self.prior.tau_prior_list[i].n_trees):
                move_class = self.sample_move(EnsembleName.TAU)  
                move = move_class(
                    current=iter_current.tau_view_list[i], trees_changed=[k], possible_thresholds = thresholds_treated, tol=self.tol
                )
                if move.propose(self.generator): # Check if a valid move was proposed
                    # Metropolis–Hastings
                    Z = self.generator.uniform(0,1)
                    if np.log(Z) < temp * self.prior.trees_log_mh_ratio(move, data_y = remaining_y, ensemble_id=BCFEnsembleIndex(EnsembleName.TAU, i)):
                        new_leaf_vals = self.prior.resample_leaf_vals(move.proposed, data_y = remaining_y, ensemble_id=BCFEnsembleIndex(EnsembleName.TAU, i), tree_ids = [k])
                        move.proposed.update_leaf_vals([k], new_leaf_vals)
                        # iter_current.tau_view = move.proposed
                        iter_current = move.proposed.bcf_params

                if iter_trace:
                    iter_trace.append(iter_current)

        # 3) Resample global parameters
        iter_current.global_params = self.prior.resample_global_params(iter_current, data_y = self.data.y, z=self.data.z)

        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current
