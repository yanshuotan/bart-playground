import numpy as np
from tqdm import tqdm
import time, logging
import pandas as pd
import math
from typing import Callable, Optional

sim_logger = logging.getLogger(__name__)

class Scenario:
    def __init__(self, P, K, sigma2, random_generator=None):
        """
        Parameters:
            P (int): Number of covariates (features).
            K (int): Number of arms (including control).
            sigma2 (float): Noise variance.
            random_generator: Random number generator instance. If None, np.random.default_rng is used.
        """
        self.P = P
        self.K = K
        self.sigma2 = sigma2
        self.rng = random_generator if random_generator is not None else np.random.default_rng()
        self.init_params()

    def init_params(self):
        pass

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.
        """
        if seed is not None:
            sim_logger.info(f"Setting seed to {seed} in {self.__class__.__name__}.set_seed()")
            self.rng = np.random.default_rng(seed)
        else:
            sim_logger.warning(f"No seed provided, using np.random.default_rng() in {self.__class__.__name__}.set_seed()")
            self.rng = np.random.default_rng()

    def shuffle(self, random_state=None):
        """
        Shuffle the scenario and reset parameters. Default implementation just reinitializes.
        """
        if random_state is not None:
            self.set_seed(random_state)
        self.init_params()

    def generate_covariates(self):
        # Generate a vector of P covariates (features) sampled from a uniform distribution.
        return np.asarray(self.rng.uniform(-1, 1, size=self.P), dtype=np.float32)

    def reward_function(self, x):
        """
        Given a feature vector x, compute:
          - outcome_mean: Expected rewards for each arm.
          - reward: Outcome_mean plus noise.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @property
    def max_draws(self):
        return math.inf # Maximum number of draws for the scenario, default is infinity.

    @property
    def rng_state(self):
        return self.rng.bit_generator.state

def simulate(scenario, agents, n_draws, agent_names: list[str]=[], on_draw: Optional[Callable[[int], None]] = None):
    """
    Simulate a bandit problem using the provided scenario and agents. The `simulate` function takes a scenario, a list of agents, and the number of draws. For each draw:

    1. Generate covariates.
    2. Compute the outcome means and rewards.
    3. For each agent, choose an arm based on the current covariates.
    4. Update cumulative regret for the agent.
    5. Update the agent's state with the observed reward.
    
    Parameters:
        scenario: An instance of a Scenario subclass.
        agents (list): List of agent instances (e.g. BARTTSAgent).
        n_draws (int): Number of simulation rounds.
    
    Returns:
        cum_regrets (np.ndarray): Cumulative regrets for each agent over draws.
        time_agent (np.ndarray): Total computation time (in seconds) for each agent.
    """
    n_agents = len(agents)
    cum_regrets = np.zeros((n_draws, n_agents))
    time_agents = np.zeros((n_draws, n_agents))

    def _is_perfect_square(n):
        if n < 0:
            return False
        root = int(math.sqrt(n))
        return root * root == n
    
    for draw in tqdm(range(n_draws), desc="Simulating", miniters=1):
        x = scenario.generate_covariates()
        u = scenario.reward_function(x)
        outcome_mean = u["outcome_mean"]
        for i, agent in enumerate(agents):
            t0 = time.time()
            arm = agent.choose_arm(x)
            # Calculate instantaneous regret: difference between best expected reward and the reward of the chosen arm.
            inst_regret = max(outcome_mean) - outcome_mean[arm]
            # Accumulate regret over draws.
            if draw == 0:
                cum_regrets[draw, i] = inst_regret
            else:
                cum_regrets[draw, i] = cum_regrets[draw - 1, i] + inst_regret
            # Update agent's state with the chosen arm's data.
            agent.update_state(arm, x, u["reward"][arm])
            time_agents[draw, i] = time.time() - t0
        # Allows external code to snapshot state (0-based index).
        if on_draw is not None:
            try:
                on_draw(draw)
            except Exception as e:
                # Swallow errors to avoid breaking core simulation
                sim_logger.error(f"Error in on_draw callback: {e}")
                pass

        # Log current status based on square number intervals
        draw_idx = draw + 1
        should_log = False
        if draw_idx == 10:
            should_log = True
        elif draw_idx > 10 and _is_perfect_square(draw_idx):
            should_log = True
        elif draw_idx == n_draws:  # Always log the final draw
            should_log = True
            
        if should_log:   
                df = pd.DataFrame({
                    "AgentName":       agent_names,
                    "CumRegret":   cum_regrets[draw, :],
                    "CumTime":     np.sum(time_agents[:draw, :], axis=0)
                })
                # log it with fixed‐width columns and 6 decimal places
                sim_logger.debug(f"Draw {draw_idx}/{n_draws}: \n" + df.to_string(
                    index=False,
                    float_format="%.6f"
                ))
            
    for agent in agents:
        if hasattr(agent, 'clean_up'):
            agent.clean_up()
        
    return cum_regrets, time_agents


