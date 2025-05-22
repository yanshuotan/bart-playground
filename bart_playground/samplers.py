from matplotlib.pylab import f
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .params import Tree, Parameters
from .moves import all_moves
from .util import Dataset
from .priors import *
from .priors import *
from bart_playground import moves
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
                 change_theta_0: bool = False, min_theta_0: float = 10, theta_0_nskip_prop: float = 0.5):
        """
        Initialize the sampler with the given parameters.

        Parameters:
            prior: The prior distribution.
            proposal_probs (dict): A dictionary containing proposal probabilities.
            generator (np.random.Generator): A random number generator.
            temp_schedule (TemperatureSchedule): Temperature schedule for the sampler.
            change_theta_0 (bool): Whether to enable theta_0 functionality.
            min_theta_0 (float): Minimum value for theta_0 during updates (if enabled).

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
        self.n_iter = None
        self.proposals = proposal_probs
        self.temp_schedule = temp_schedule
        self.trace = []
        self.generator = generator
        self.change_theta_0 = change_theta_0  # Enable or disable theta_0 functionality
        self.min_theta_0 = min_theta_0  # Minimum theta_0 value if enabled
        self.theta_0_nskip_prop = theta_0_nskip_prop  # Proportion of n_skip for theta_0 decay

        # create cache for moves
        self.moves_cache = None
        # current move cache iterator
        self.moves_cache_iterator = None

       # Save the initial value of theta_0 if enabled
        if self.change_theta_0 and hasattr(prior.global_prior, 'theta_0'):
            self.initial_theta_0 = prior.global_prior.theta_0  # Save the initial value
        else:
            self.initial_theta_0 = None  # Handle cases where theta_0 is not defined
        
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

    def update_theta_0(self, iteration, total_iterations):
        """
        Linearly decrease theta_0 in self.global_prior as iterations progress.

        Parameters:
            iteration (int): Current iteration number.
            total_iterations (int): Total number of iterations.
        """
        if not self.change_theta_0:
            return  # Skip if theta_0 functionality is disabled

        if self.initial_theta_0 is not None and total_iterations > 0:
            max_theta_0 = self.initial_theta_0  # Use the saved initial value
            # Use the user-defined minimum theta_0
            self.global_prior.theta_0 = max(
                self.min_theta_0,
                max_theta_0 - (max_theta_0 - self.min_theta_0) * (iteration / total_iterations)
            )
        
    def run(self, n_iter, progress_bar = True, quietly = False, current = None, n_skip = 0):
        """
        Run the sampler for a specified number of iterations from `current` or a fresh start.

        Parameters:
        n_iter (int): The number of iterations to run the sampler.
        """
        if quietly:
            progress_bar = False

        self.trace = []
        self.n_iter = n_iter
        if current is None:
            current = self.get_init_state()
        # assert isinstance(current, Parameters), "Current state must be of type Parameters."
        if n_skip == 0:
            self.trace.append(current) # Add initial state to trace
        
        iterator = tqdm(range(n_iter), desc="Iterations") if progress_bar else range(n_iter)
    
        for iter in iterator:
            if not progress_bar and iter % 10 == 0 and not quietly:
                print(f"Running iteration {iter}/{n_iter}")
            # print(self.temp_schedule)
            temp = self.temp_schedule(iter)
            current = self.one_iter(current, temp, return_trace=False, iteration=iter)
            # Update theta_0
            if self.change_theta_0 and iter <= int(self.theta_0_nskip_prop * n_skip):
                self.update_theta_0(iter, int(self.theta_0_nskip_prop*n_skip))
            if iter >= n_skip:
                if len(self.trace) > 0:
                    self.trace[-1].clear_cache()
                self.trace.append(current)

        return self.trace
    
    def sample_move(self):
        """
        Samples a move based on the proposal probabilities.

        This method selects a move from the proposals dictionary, where the keys
        are the possible moves and the values are the corresponding probabilities.
        It uses the generator's choice method to randomly select a move according
        to these probabilities.

        Returns:
            The selected move from the all_moves list based on the sampled index.
        """
        if self.moves_cache is None or self.moves_cache_iterator is None:
            moves = list(self.proposals.keys())
            move_probs = list(self.proposals.values())
            self.moves_cache = [all_moves[move] for move in self.generator.choice(moves, size=100, p=move_probs)]
            self.moves_cache_iterator = 0
        move = self.moves_cache[self.moves_cache_iterator]
        self.moves_cache_iterator += 1
        if self.moves_cache_iterator >= len(self.moves_cache):
            self.moves_cache = None
        return move
    
    @abstractmethod
    def get_init_state(self):
        """
        Retrieve the initial state for the sampler.

        This method should be overridden by subclasses to provide the specific
        initial state required for the sampling process.

        Returns:
            The initial state for the sampler.
        """
        pass

    @abstractmethod
    def one_iter(self, current, temp, return_trace=False):
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
                    new_X = new_data.X[old_n:]
                    if hasattr(new_data, 'Z'): # check if treatment assignments are available, e.g. for BCFDataset
                        new_z = new_data.Z[old_n:]
                        current_state = last_state.add_data_points(new_X, new_z)
                    else:
                        current_state = last_state.add_data_points(new_X)
                else:
                    current_state = last_state
            else:
                current_state = last_state

            # Run sampler for additional iterations
            return self.run(additional_iters, quietly=quietly, current=current_state)

class DefaultSampler(Sampler):
    """
    Default implementation of the BART sampler.
    """
    def __init__(self, prior : ComprehensivePrior, proposal_probs: dict,
                 generator : np.random.Generator, temp_schedule=TemperatureSchedule(), tol=100):
        self.tol = tol
        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood
        super().__init__(prior, proposal_probs, generator, temp_schedule)

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        trees = [Tree.new(self.data.X) for _ in range(self.tree_prior.n_trees)]
        global_params = self.global_prior.init_global_params(self.data)
        init_state = Parameters(trees, global_params)
        return init_state
    
    def log_mh_ratio(self, move : Move, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio"""
        return self.tree_prior.trees_log_prior_ratio(move) + \
            self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize) + \
            move.log_tran_ratio

    def one_iter(self, current, temp, return_trace=False, iteration=None):
        """
        Perform one iteration of the sampler.
        """
        iter_current = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]
        for k in range(self.tree_prior.n_trees):
            move = self.sample_move()(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol
                )
            if move.propose(self.generator): # Check if a valid move was proposed
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move) / temp:
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
        
class NTreeSampler(Sampler):
    """
    Change the number of trees implementation of the BART sampler.
    """
    def __init__(self, prior : ComprehensivePrior, proposal_probs: dict,
                 generator: np.random.Generator, temp_schedule=TemperatureSchedule(),
                 special_probs: dict = None, tol=100, special_move_interval=10, 
                 change_theta_0=True, min_theta_0=10, theta_0_nskip_prop: float = 0.5):
        self.tol = tol
        # Number of iterations of default moves before considering special moves.
        self.special_move_interval = special_move_interval 
        # Default probabilities for special moves
        if special_probs is None:
            special_probs = {"birth": 0.25, "death": 0.25, "break": 0.25, "combine": 0.25}
        self.special_probs = special_probs

        # Record tree mh ratios
        self.break_mh_ratios = []
        self.combine_mh_ratios = []
        self.birth_mh_ratios = []
        self.death_mh_ratios = []

        # Accepted movements
        self.accepted_moves = {"birth": 0, "death": 0, "break": 0, "combine": 0}

        if proposal_probs is None:
            proposal_probs = {"grow" : 0.5,
                              "prune" : 0.5}
        self.tree_prior = prior.tree_prior
        self.global_prior = prior.global_prior
        self.likelihood = prior.likelihood
        self.tree_num_prior = prior.tree_num_prior
        self.leaf_val_prior = prior.leaf_val_prior

        # Initial number of trees
        self.ini_ntrees = self.tree_prior.n_trees

        super().__init__(prior, proposal_probs, generator, temp_schedule, 
                         change_theta_0=change_theta_0, min_theta_0=min_theta_0, theta_0_nskip_prop=theta_0_nskip_prop)

    def get_init_state(self) -> Parameters:
        """
        Retrieve the initial state for the sampler.

        Returns:
            The initial state for the sampler.
        """
        if self.data is None:
            raise AttributeError("Need data before running sampler.")
        trees = [Tree.new(self.data.X) for _ in range(self.tree_prior.n_trees)]
        global_params = self.global_prior.init_global_params(self.data)
        init_state = Parameters(trees, global_params)
        return init_state
    
    def log_mh_ratio(self, move : Move, marginalize : bool=False):
        """Calculate total log Metropolis-Hastings ratio"""
        if isinstance(move, (Break, Combine, Birth, Death)): # Omit likelihood and leaf value prior for special moves
            return self.tree_prior.trees_log_prior_ratio(move) + \
                self.tree_num_prior.tree_num_log_prior_ratio(move) + \
                self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize) + \
                self.leaf_val_prior.leaf_vals_log_prior_ratio(move) + \
                move.log_tran_ratio
        else: # Default BART moves containing likelihood
            return self.tree_prior.trees_log_prior_ratio(move) + \
                self.likelihood.trees_log_marginal_lkhd_ratio(move, self.data.y, marginalize) + \
                move.log_tran_ratio
        
    def special_moves(self, iter_current, temp, iter_trace):
        """
        Perform special moves: birth, death, break, combine.
        """
        # Randomly permute the positions of all trees in iter_current.trees
        permuted_indices = self.generator.permutation(len(iter_current.trees))
        iter_current.trees = [iter_current.trees[i] for i in permuted_indices]

        # Special moves: Birth, Death, Break, Combine
        special_moves = ["birth", "death", "break", "combine"]
        special_probs = [self.special_probs.get(move, 0) for move in special_moves]
        selected_move = self.generator.choice(special_moves, p=special_probs)

        if selected_move == "birth" and (
            self.tree_num_prior.prior_type != "bernoulli" or 
            self.tree_prior.n_trees < self.ini_ntrees + 1
        ):
            birth_id = self.generator.integers(0, len(iter_current.trees)) # Just a dummy id for easier mh ratio calculation
            move = Birth(iter_current, [birth_id], tol=self.tol)
            if move.propose(self.generator):
                Z = self.generator.uniform(0, 1)
                self.birth_mh_ratios.append(np.exp(self.log_mh_ratio(move) / temp))
                if np.log(Z) < self.log_mh_ratio(move) / temp:
                    self.accepted_moves["birth"] += 1
                    self.tree_prior.n_trees += 1
                    self.tree_prior.update_f_sigma2(self.tree_prior.n_trees)
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [-1])
                    move.proposed.update_leaf_vals([-1], new_leaf_vals)
                    iter_current = move.proposed
                    iter_trace.append((1, move.proposed))

        elif selected_move == "death" and (
            (self.tree_num_prior.prior_type != "bernoulli" and self.tree_prior.n_trees > 1) or
            (self.tree_num_prior.prior_type == "bernoulli" and self.tree_prior.n_trees > self.ini_ntrees)
        ):
            death_id = 0 # Select the first tree after permutation (might not be only_root)
            possible_indices = [i for i in range(len(iter_current.trees)) if i != death_id]
            random_id = self.generator.choice(possible_indices) # Just a dummy id for easier mh ratio calculation
            move = Death(iter_current, [random_id, death_id], tol=self.tol)
            if move.propose(self.generator):
                Z = self.generator.uniform(0, 1)
                self.death_mh_ratios.append(np.exp(self.log_mh_ratio(move) / temp))
                if np.log(Z) < self.log_mh_ratio(move) / temp:
                    self.accepted_moves["death"] += 1
                    self.tree_prior.n_trees -= 1
                    self.tree_prior.update_f_sigma2(self.tree_prior.n_trees)
                    iter_current = move.proposed
                    iter_trace.append((1, move.proposed))

        elif selected_move == "break" and (
            self.tree_num_prior.prior_type != "bernoulli" or 
            self.tree_prior.n_trees < self.ini_ntrees + 1
        ):
            break_id = [0] # Select the first tree after permutation
            move = Break(iter_current, break_id, self.tol)   
            if move.propose(self.generator):
                self.break_mh_ratios.append(np.exp(self.log_mh_ratio(move) / temp))
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move) / temp:
                    self.accepted_moves["break"] += 1
                    self.tree_prior.n_trees += 1
                    self.tree_prior.update_f_sigma2(self.tree_prior.n_trees)
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = break_id + [-1])
                    move.proposed.update_leaf_vals(break_id + [-1], new_leaf_vals)
                    iter_current = move.proposed
                    iter_trace.append((1, move.proposed))
        
        elif selected_move == "combine" and (
            (self.tree_num_prior.prior_type != "bernoulli" and self.tree_prior.n_trees > 1) or
            (self.tree_num_prior.prior_type == "bernoulli" and self.tree_prior.n_trees > self.ini_ntrees)
        ):
            combine_ids = [0, 1] # Select the first two trees after permutation
            combine_position = combine_ids[0] if combine_ids[0] < combine_ids[1] else combine_ids[0] - 1
            move = Combine(iter_current, combine_ids, self.tol)   
            if move.propose(self.generator):
                self.combine_mh_ratios.append(np.exp(self.log_mh_ratio(move) / temp))
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move) / temp:
                    self.accepted_moves["combine"] += 1
                    self.tree_prior.n_trees -= 1
                    self.tree_prior.update_f_sigma2(self.tree_prior.n_trees)
                    new_leaf_vals = self.tree_prior.resample_leaf_vals(move.proposed, data_y = self.data.y, tree_ids = [combine_position])
                    move.proposed.update_leaf_vals([combine_position], new_leaf_vals)
                    iter_current = move.proposed
                    iter_trace.append((1, move.proposed))
        return iter_current

    def one_iter(self, current, temp, return_trace=False, iteration=0):
        """
        Perform one iteration of the sampler.
        """
        iter_current = current.copy() # First make a copy
        iter_trace = [(0, iter_current)]

        # Perform special moves if the iteration is a multiple of special_move_interval
        if iteration % self.special_move_interval == 0:
            iter_current = self.special_moves(iter_current, temp, iter_trace)

        # Default BART
        for k in range(self.tree_prior.n_trees):
            move = self.sample_move()(
                iter_current, [k], possible_thresholds=self.possible_thresholds, tol=self.tol
                )
            if move.propose(self.generator): # Check if a valid move was proposed
                Z = self.generator.uniform(0, 1)
                if np.log(Z) < self.log_mh_ratio(move) / temp:
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
    
all_samplers = {"default" : DefaultSampler,
                "ntree": NTreeSampler}

default_proposal_probs = {"grow" : 0.25,
                          "prune" : 0.25,
                          "change" : 0.4,
                          "swap" : 0.1}

default_special_probs = {"birth": 0.25, 
                         "death": 0.25, 
                         "break": 0.25, 
                         "combine": 0.25}