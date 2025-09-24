import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, Union
from scipy.stats import truncnorm   

from .params import Tree, Parameters
from .moves import all_moves, Move, contrary_moves
from .util import Dataset
from .priors import  ComprehensivePrior, ProbitPrior, LogisticPrior

class TemperatureSchedule:

    def __init__(self, temp_schedule: Callable[[int], int] = lambda x: 1):
        self.temp_schedule = temp_schedule
    
    def __call__(self, t):
        return self.temp_schedule(t)
    
class Sampler(ABC):
    """
    Base class for the BART sampler.
    """
    def __init__(self, prior, proposal_probs: dict,  
                 generator : np.random.Generator, temp_schedule: TemperatureSchedule = TemperatureSchedule(),
                 tol: int = 100):
        """
        Initialize the sampler with the given parameters.

        Parameters:
            prior: The prior distribution.
            proposal_probs (dict): A dictionary containing proposal probabilities.
            generator (np.random.Generator): A random number generator.
            temp_schedule (TemperatureSchedule): Temperature schedule for the sampler.

        Attributes:
            data: Placeholder for data, initially set to None.
            prior: The prior distribution.
            n_iter: Number of iterations, initially set to None.
            proposals (dict): A dictionary containing proposal probabilities.
            temp_schedule (TemperatureSchedule): Temperature schedule for the sampler.
            trace (list): A list to store the trace of the sampling process.
            generator (np.random.Generator): A random number generator.
        """
        self._data : Optional[Dataset] = None
        self.prior = prior

        self.tree_prior = prior.tree_prior
        self.likelihood = prior.likelihood
        self.tol = tol

        self.n_iter = None
        self.proposals = proposal_probs
        self.temp_schedule = temp_schedule
        self.trace = []
        self.generator = generator
        # create cache for moves
        self.moves_str_cache = None
        # current move cache iterator
        self.moves_cache_iterator = None

        # --- Add move statistics ---
        self.move_selected_counts = {k: 0 for k in self.proposals}
        self.move_success_counts = {k: 0 for k in self.proposals}
        self.move_accepted_counts = {k: 0 for k in self.proposals}
        
    @property
    def data(self) -> Dataset:
        assert self._data, "Data has not been added yet."
        return self._data
    
    def add_data(self, data : Dataset):
        """
        Adds data to the sampler.

        Parameters:
        data (Dataset): The data to be added to the sampler.
        """
        self._data = data

    def add_thresholds(self, thresholds):
        self.possible_thresholds = thresholds
        
    def clear_last_cache(self):
        '''
        This method clears the cache of the last trace in the sampler.
        '''
        if len(self.trace) > 0:
            self.trace[-1].clear_cache()
        
    def run(self, n_iter: int, progress_bar: bool = True, quietly: bool = False, current: Parameters | list[Parameters] | None = None, n_skip: int = 0):
        """
        Run the sampler for a specified number of iterations from `current` or a fresh start.

        Parameters:
        n_iter (int): The number of iterations to run the sampler.
        """
        if quietly:
            progress_bar = False

        # Determine the actual starting state for this MCMC run
        if current is not None:
            current_state = current
        elif self.trace:  # If self.trace is already populated (e.g., by init_from_xgboost)
            current_state = self.trace[0]  # Use the pre-loaded state
        else:
            current_state = self.get_init_state() # Otherwise, generate a new initial state
        
        self.trace = []
        self.n_iter = n_iter

        if n_skip == 0:
            self.trace.append(current_state) # Add initial state to trace

        iterator = tqdm(range(n_iter), desc="Iterations") if progress_bar else range(n_iter)

        for iter in iterator:
            if not progress_bar and iter % 10 == 0 and not quietly:
                print(f"Running iteration {iter}/{n_iter}")
            
            temp = self.temp_schedule(iter)
            current_state = self.one_iter(current_state, temp, return_trace=False)

            if iter >= n_skip:
                self.clear_last_cache()  # Clear cache of the last trace
                self.trace.append(current_state)
        
        return self.trace
    
    def sample_move(self):
        """
        Samples a move based on the proposal probabilities.

        This method selects a move from the proposals dictionary, where the keys
        are the possible moves and the values are the corresponding probabilities.
        It uses the generator's choice method to randomly select a move according
        to these probabilities.

        Returns:
            A tuple of (move_str, move_cls) where move_str is the move name and
            move_cls is the corresponding class from all_moves.
        """
        if self.moves_str_cache is None or self.moves_cache_iterator is None:
            moves = list(self.proposals.keys())
            move_probs = list(self.proposals.values())
            # Cache a batch of move names (strings)
            self.moves_str_cache = [m for m in self.generator.choice(moves, size=100, p=move_probs)]
            self.moves_cache_iterator = 0
        move_str = self.moves_str_cache[self.moves_cache_iterator]
        self.moves_cache_iterator += 1
        if self.moves_cache_iterator >= len(self.moves_str_cache):
            self.moves_str_cache = None
        move_cls = all_moves[move_str]
        return move_str, move_cls
    
    def log_mh_ratio(self, move : Move, temp, move_key: str, data_y = None, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio with proposal-probability correction."""
        data_y = self.data.y if data_y is None else data_y
        return (self.tree_prior.trees_log_prior_ratio(move) + \
            self.likelihood.trees_log_marginal_lkhd_ratio(move, data_y, marginalize)) / temp + \
            move.log_tran_ratio + \
            np.log(self.proposals[contrary_moves[move_key]]) - np.log(self.proposals[move_key])
    
    @abstractmethod
    def get_init_state(self) -> Any:
        """
        Retrieve the initial state for the sampler.

        This method should be overridden by subclasses to provide the specific
        initial state required for the sampling process.

        Returns:
            The initial state for the sampler.
        """
        pass

    @abstractmethod
    def one_iter(self, current, temp, return_trace=False) -> Any:
        """
        Perform one iteration of the sampler.
        """
        pass

    def continue_run(self, additional_iters, new_data=None, quietly=False, last_state=None):
            """
            Continue sampling with updated data from a previous state.

            Parameters:
                additional_iters: Number of additional iterations
                new_data: Updated dataset (if None, uses existing data)
                quietly: Whether to suppress output
                last_state: Last state from previous run (if None, uses last state in trace)

            Returns:
                New trace segment
            """
            # Get last state
            if last_state is None:
                if hasattr(self, 'trace') and self.trace:
                    last_state = self.trace[-1]
                else:
                    raise ValueError("No last_state provided and no trace available")

            # Update parameter state with any new data points if needed
            if new_data is not None:
                old_n = self.data.n
                new_n = new_data.n
                
                self.add_data(new_data)

                if new_n > old_n:
                    # new_X = new_data.X[old_n:]
                    if hasattr(new_data, 'Z'): # check if treatment assignments are available, e.g. for BCFDataset
                        raise NotImplementedError("Adding new data with treatment assignments (Z) has a changed interface and should be handled before using.")
                        new_z = new_data.Z[old_n:]
                        current_state = last_state.add_data_points(new_X, new_z)
                    elif isinstance(last_state, list):
                        # If last_state is a list (e.g., in LogisticSampler), handle each category
                        current_state = []
                        for i, state in enumerate(last_state):
                            current_state.append(state.update_data(new_data.X))
                    else: # Default case for Parameters
                        current_state = last_state.update_data(new_data.X)
                else: # No new data, just continue from last state
                    current_state = last_state
            else: # No new data, just continue from last state
                current_state = last_state

            # Run sampler for additional iterations
            return self.run(additional_iters, quietly=quietly, current=current_state)

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(
        self,
        prior: ComprehensivePrior,
        proposal_probs: dict,
        generator: np.random.Generator,
        temp_schedule=TemperatureSchedule(),
        tol: int = 100,
        init_trees: Optional[list[Tree]] = None  # NEW
    ):
        """
        Default implementation of the BART sampler.
        Accepts an optional list of pre-initialized trees without changing default behavior.
        """
        # preserve original default proposal behavior
        if proposal_probs is None:
            proposal_probs = {"grow": 0.5, "prune": 0.5}

        # original prior unpacking
        self.global_prior = prior.global_prior

        # initialize base sampler
        super().__init__(prior, proposal_probs, generator, temp_schedule, tol)

        # store seed forest for XGBoost init
        self.init_trees = init_trees
        # --- Add move statistics ---
        self.move_selected_counts = {k: 0 for k in self.proposals}
        self.move_success_counts = {k: 0 for k in self.proposals}
        self.move_accepted_counts = {k: 0 for k in self.proposals}

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.
        If init_trees was provided, copy up to n_trees of them and
        pad the rest with fresh stumps; otherwise build all new stumps.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        N = self.tree_prior.n_trees

        if self.init_trees is not None:
            provided = len(self.init_trees)
            # Copy up to N of the provided trees
            trees = [t.copy() for t in self.init_trees[:N]]
            # Pad with fresh stumps if fewer than N
            if provided < N:
                trees += [Tree.new(self.data.X) for _ in range(N - provided)]
        else:
            trees = [Tree.new(self.data.X) for _ in range(N)]

        global_params = self.global_prior.init_global_params(self.data)
        return Parameters(trees, global_params)
    
    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        iter_current = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]
        for k in range(self.tree_prior.n_trees):
            move_key, move_cls = self.sample_move()
            self.move_selected_counts[move_key] += 1
            move = move_cls(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol
              )
            if move.propose(self.generator): # Check if a valid move was proposed
                self.move_success_counts[move_key] += 1
                Z = self.generator.uniform(0, 1)
                marginalize = getattr(self, 'marginalize', False)
                if np.log(Z) < self.log_mh_ratio(move, temp, move_key=move_key, marginalize=marginalize):
                    self.move_accepted_counts[move_key] += 1
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    iter_current = move.proposed
                    if return_trace:
                        iter_trace.append((k+1, move.proposed))
        iter_current.global_params = self.global_prior.resample_global_params(iter_current, data_y = self.data.y)
        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current
    
class ProbitSampler(Sampler):
    """
    Probit sampler for binary BART.
    """
    def __init__(self, prior : ProbitPrior, proposal_probs: dict,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        super().__init__(prior, proposal_probs, generator, temp_schedule, tol)

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        trees = [Tree.new(self.data.X) for _ in range(self.tree_prior.n_trees)]
        init_state = Parameters(trees, {"eps_sigma2": 1})
        return init_state
    
    def __sample_Z(self, y, Gx):
        Z = np.empty_like(Gx)

        mask1 = (y == 1)
        mask0 = ~mask1

        # For Y_i = 1: Z_i ~ TruncNormal(Gx[i], 1, lower=0, upper=inf)
        if np.any(mask1):
            a1 = (0 - Gx[mask1]) / 1
            b1 = np.full_like(a1, np.inf)
            Z[mask1] = truncnorm.rvs(a1, b1, loc=Gx[mask1], scale=1, random_state=self.generator)

        # For Y_i = 0: Z_i ~ TruncNormal(Gx[i], 1, lower=-inf, upper=0)
        if np.any(mask0):
            a0 = np.full_like(Gx[mask0], -np.inf)
            b0 = (0 - Gx[mask0]) / 1
            Z[mask0] = truncnorm.rvs(a0, b0, loc=Gx[mask0], scale=1, random_state=self.generator)

        return Z

    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        iter_current : Parameters = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]
        
        # sample latents Z
        latents = self.__sample_Z(self.data.y, iter_current.evaluate())
        
        for k in range(self.tree_prior.n_trees):
            move_str, move_cls = self.sample_move()
            self.move_selected_counts[move_str] += 1
            move = move_cls(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol
                )
            if move.propose(self.generator): # Check if a valid move was proposed
                self.move_success_counts[move_str] += 1
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move, temp, move_key=move_str, data_y=latents):
                    self.move_accepted_counts[move_str] += 1
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = latents, tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    iter_current = move.proposed
                    if return_trace:
                        iter_trace.append((k+1, move.proposed))
        
        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current
        
class MultiSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(self, prior : ComprehensivePrior, proposal_probs: dict,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=1, 
                 multi_tries: Union[int, list[int]] = 10, init_trees: Optional[list[Tree]] = None):
        self.tol = tol
        self.init_trees = init_trees
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood
        self.multi_tries = multi_tries
        super().__init__(prior, proposal_probs, generator, temp_schedule)
        # --- Add move statistics ---
        self.move_selected_counts = {k: 0 for k in self.proposals}
        self.move_success_counts = {k: 0 for k in self.proposals}
        self.move_accepted_counts = {k: 0 for k in self.proposals}
        self.multi_ratios = {
            "multigrow": [],
            "multiprune": [],
            "multichange": [],
            "multiswap": []
        }


    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        N = self.tree_prior.n_trees

        if self.init_trees is not None:
            provided = len(self.init_trees)
            # Copy up to N of the provided trees
            trees = [t.copy() for t in self.init_trees[:N]]
            # Pad with fresh stumps if fewer than N
            if provided < N:
                trees += [Tree.new(self.data.X) for _ in range(N - provided)]
        else:
            trees = [Tree.new(self.data.X) for _ in range(N)]
        global_params = self.global_prior.init_global_params(self.data)
        init_state = Parameters(trees, global_params)
        return init_state
    
    def log_mh_ratio(self, move : Move, temp, data_y = None, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio"""
        data_y = self.data.y if data_y is None else data_y
        return move.log_tran_ratio # Already considers prior and likelihood in move.py

    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        iter_current = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]
        for k in range(self.tree_prior.n_trees):
            move_key, move_cls = self.sample_move()
            self.move_selected_counts[move_key] += 1
            move = move_cls(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol,
                likelihood=self.likelihood, tree_prior=self.tree_prior, data_y=self.data.y,
                n_samples_list=self.multi_tries
            )
            if move.propose(self.generator): # Check if a valid move was proposed
                self.move_success_counts[move_key] += 1
                move_name = type(move).__name__.lower()
                if hasattr(move, "candidate_sampling_ratio"): # Record the sampling ratio for multi-moves
                    for key in self.multi_ratios:
                        if key in move_name:
                            self.multi_ratios[key].append(move.candidate_sampling_ratio)
                            break
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move, temp): # Already consider prior and likelihood in move
                    self.move_accepted_counts[move_key] += 1
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [k])
                    move.proposed.update_leaf_vals([k], new_leaf_vals)
                    iter_current = move.proposed
                    if return_trace:
                        iter_trace.append((k+1, move.proposed))
        iter_current.global_params = self.global_prior.resample_global_params(iter_current, data_y = self.data.y)
        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current
    
    
class LogisticSampler(Sampler):
    """
    Logistic sampler for BART.
    """
    def __init__(self, prior : LogisticPrior, proposal_probs: dict,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        self.n_i = None
        super().__init__(prior, proposal_probs, generator, temp_schedule, tol)
    
    @property
    def n_categories(self) -> int:
        """
        Get the number of categories for the sampler.
        """
        if not hasattr(self, '_n_cat'):
            raise AttributeError("Number of categories has not been set.")
        return self._n_cat
    @n_categories.setter
    def n_categories(self, n_cat: int):
        """
        Set the number of categories for the sampler.
        """
        if n_cat < 2:
            raise ValueError("Number of categories must be at least 2.")
        self._n_cat = n_cat

    def add_data(self, data: Dataset):
        """
        Adds data to the sampler and initializes the number of observations per category.

        Parameters:
        data (Dataset): The data to be added to the sampler.
        """
        super().add_data(data)
        self.n_i = np.zeros(self.data.X.shape[0], dtype=int)
        for category in range(self.n_categories):
            self.n_i += (self.data.y == category)
        self.is_exp = np.all(self.n_i == 1)
        
    def get_init_state(self) -> list[Parameters]:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        init_state = []
        for category in range(self.n_categories):
            trees = [Tree.new(self.data.X) for _ in range(self.prior.tree_prior.n_trees)]
            init_state.append(
                Parameters(trees, {"eps_sigma2": 1})
            )
        return init_state
        
    def __sample_phi(self, sumFx):
        # for every i
        # phi | y ~ Gamma(n, sumFx)
        # sumFx = f^{(0)}(x_i) + f^{(1)}(x_i) is rate parameter
        # n_i is the number of observation for each x_i
        if np.any(sumFx <= 0):
            raise ValueError("All sumFx must be strictly positive.")
        # Exponential should be faster than gamma (marginal speed gain, ~1%)
        if self.is_exp:
            phis = self.generator.exponential(scale=1.0/sumFx)
        else:
            from scipy.stats import gamma
            phis = gamma.rvs(a=self.n_i,
                 scale=1.0/sumFx,
                 random_state=self.generator)
        return phis

    def clear_last_cache(self):
        """
        This method clears the cache of the last trace in the sampler.
        """
        if len(self.trace) > 0:
            for category in range(self.n_categories):
                # Clear cache for each category's parameters
                self.trace[-1][category].clear_cache()
    
    def one_iter(self, current, temp, return_trace=False):
        """
        Perform one iteration of the sampler.
        """
        # First make a copy
        iter_current : list[Parameters] = []
        for category in range(self.n_categories):
            iter_current.append(current[category].copy())
        
        iter_trace = []
        for j in range(self.n_categories):
            iter_trace.append([(0, iter_current[j])])
        
        # sample latents phi
        all_sumGx = np.stack([iter_current[j].evaluate() 
                      for j in range(self.n_categories)],
                     axis=0)  # (n_categories, n_samples)
        Fx = np.exp(all_sumGx)  
        sumFx = Fx.sum(axis=0)  
        latents = self.__sample_phi(sumFx)
        
        self.prior.set_latents(latents)
                    
        for category in range(self.n_categories):
            for h in range(self.tree_prior.n_trees):
                move_str, move_cls = self.sample_move()
                self.move_selected_counts[move_str] += 1
                move = move_cls(
                    iter_current[category], [h], possible_thresholds=self.possible_thresholds, tol=self.tol
                )
                if move.propose(self.generator):  # Check if a valid move was proposed
                    self.move_success_counts[move_str] += 1
                    yi_match = (self.data.y == category)
                    Z = self.generator.uniform(0, 1)
                    if np.log(Z) < self.log_mh_ratio(move, temp=temp, data_y=yi_match, move_key=move_str):
                        self.move_accepted_counts[move_str] += 1
                        new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y=yi_match, tree_ids=[h])
                        move.proposed.update_leaf_vals([h], new_leaf_vals)
                        iter_current[category] = move.proposed
                        if return_trace:
                            iter_trace[category].append((h+1, move.proposed))

        if return_trace:
            return iter_trace
        else:
            del iter_trace
            return iter_current

all_samplers = {"default" : DefaultSampler, "multi": MultiSampler, "binary": ProbitSampler, "logistic": LogisticSampler}

default_proposal_probs = {"grow" : 0.25,
                          "prune" : 0.25,
                          "change" : 0.4,
                          "swap" : 0.1}
