
from .samplers import DefaultSampler, default_proposal_probs, Sampler
from .moves import all_moves
from .bcf_params import BCFParams
from .util import Tree

import numpy as np

class BCFSampler(Sampler):
    """Extends sampler for BCF's dual tree ensembles"""
    def __init__(self, prior, proposal_probs=None, **kwargs):
        self.proposals_mu = proposal_probs or default_proposal_probs
        self.proposals_tau = proposal_probs or default_proposal_probs
        self.prior = prior
        
    def get_init_state(self):
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        trees = [Tree() for _ in range(self.prior.mu_prior.n_trees)]
        global_params = self.prior.init_global_params(self.data.X, self.data.y)
        init_state = BCFParams(trees, global_params, self.data)
        return init_state

    def sample_move(self, tree_type):
        """Sample move type for specified tree ensemble"""
        if tree_type == 'mu':
            moves = list(self.proposals_mu.keys())
            probs = list(self.proposals_mu.values())
        else:
            moves = list(self.proposals_tau.keys())
            probs = list(self.proposals_tau.values())
        return all_moves[self.generator.choice(moves, p=probs)]
    
    def one_iter(self, temp=1):
        """One MCMC iteration: Update μ trees -> τ trees -> global params"""
        current = self.current
        accepted = False
        
        # Update prognostic trees (μ)
        for k in range(self.prior.mu_prior.n_trees):
            MoveClass = self.sample_move('mu')
            move = MoveClass(current.mu_params, [k], self.tol)
            proposed_mu = move.propose(self.generator)
            
            # Calculate MH ratio
            log_prior = self.prior.mu_prior.trees_log_prior(proposed_mu, [k]) - \
                        self.prior.mu_prior.trees_log_prior(current.mu_params, [k])
            log_likelihood = proposed_mu.log_likelihood() - current.mu_params.log_likelihood()
            log_ratio = log_prior + log_likelihood
            
            if np.log(self.generator.uniform()) < temp * log_ratio:
                current = BCFParams(
                    proposed_mu, 
                    current.tau_params,
                    current.global_params,
                    current.data
                )
                accepted = True
        
        # Update treatment effect trees (τ)
        for k in range(self.prior.tau_prior.n_trees):
            MoveClass = self.sample_move('tau')
            move = MoveClass(current.tau_params, [k], self.tol)
            proposed_tau = move.propose(self.generator)
            
            # Calculate MH ratio including z*tau term
            log_prior = self.prior.tau_prior.trees_log_prior(proposed_tau, [k]) - \
                        self.prior.tau_prior.trees_log_prior(current.tau_params, [k])
            log_likelihood = proposed_tau.log_likelihood() - current.tau_params.log_likelihood()
            log_ratio = log_prior + log_likelihood
            
            if np.log(self.generator.uniform()) < temp * log_ratio:
                current = BCFParams(
                    current.mu_params,
                    proposed_tau,
                    current.global_params,
                    current.data
                )
                accepted = True

        # Resample leaf values and global parameters
        if accepted:
            current = self.prior.resample_leaf_vals(current)
        current.global_params = self.prior.resample_global_params(current)
        
        return current
    
    def run(self, n_iter):
        """
        Run the sampler for a specified number of iterations.

        Parameters:
        n_iter (int): The number of iterations to run the sampler.
        """
        self.trace = []
        for _ in range(n_iter):
            self.current = self.one_iter()
            self.trace.append(self.current)

    def _apply_move(self, move, temp):
        # TODO
        pass
