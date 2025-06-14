import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import math

class Scenario:
    def __init__(self, P, K, sigma2, random_generator=None):
        """
        Parameters:
            P (int): Number of covariates (features).
            K (int): Number of arms (including control).
            sigma2 (float): Noise variance.
            random_generator: Random number generator instance. If None, np.random is used.
        """
        self.P = P
        self.K = K
        self.sigma2 = sigma2
        self.rng = random_generator if random_generator is not None else np.random
        self.init_params()

    def init_params(self):
        pass 

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.
        """
        self.rng.seed(seed)

    def generate_covariates(self):
        # Generate a vector of P covariates (features) sampled from a normal distribution.
        return self.rng.normal(0, 1, size=self.P).astype(np.float32)

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

class LinearScenario(Scenario):
    def __init__(self, P, K, sigma2, random_generator=None):
        super().__init__(P, K, sigma2, random_generator)

    def init_params(self):
        # Generate a K x P matrix of arm-specific coefficients uniformly between -1 and 1.
        self.mu_a = self.rng.uniform(-1, 1, size=(self.K, self.P))
    
    def reward_function(self, x):
        # Compute noise for each arm.
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        # Compute expected rewards (outcome means) for each arm.
        outcome_mean = 10 * self.mu_a.dot(x)
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}
    
class LinearOffsetScenario(Scenario):
    def __init__(self, P, K, sigma2, random_generator=None):
        super().__init__(P, K, sigma2, random_generator)
        
    def init_params(self):
        # Uniformly distributed covariates in unit cube
        self.mu = self.rng.uniform(-1, 1, size=(1, self.P))
        # Generate a K x 1 matrix of arm-specific offsets uniformly between -5 and 5.
        self.arm_offsets = self.rng.uniform(-5, 5, size=self.K)
    
    def reward_function(self, x):
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * self.mu.dot(x) + self.arm_offsets
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

class OffsetScenario(Scenario):
    def __init__(self, P, K, sigma2, lambda_val=3, random_generator=None):
        self.lambda_val = lambda_val
        super().__init__(P, K, sigma2, random_generator)

    def init_params(self):
        # Uniformly distributed covariates in unit cube
        self.mu = self.rng.uniform(-1, 1, size=(1, self.P))
        # Generate a K x 1 matrix of arm-specific offsets uniformly between -5 and 5.
        self.arm_offsets = self.rng.uniform(-5, 5, size=self.K)
    
    def reward_function(self, x):
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * np.sin(self.mu.dot(x)) + self.lambda_val * self.arm_offsets
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

class FriedmanScenario(Scenario):
    def __init__(self, P, K, sigma2, lambda_val=3, random_generator=None):
        if P < 5:
            raise ValueError("Friedman is for P>=5")
        self.lambda_val = lambda_val
        super().__init__(P, K, sigma2, random_generator)
    
    def init_params(self):
        self.arm_offsets = self.rng.uniform(-5, 5, size=self.K)

    def reward_function(self, x):
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * np.sin(np.pi * x[0] * x[1]) + \
                      20 * (x[2] - 0.5) ** 2 + \
                      10 * x[3] + 5 * x[4] + \
                      self.lambda_val * self.arm_offsets
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize, OrdinalEncoder
from sklearn.utils import shuffle
class OpenMLScenario(Scenario):
    def __init__(self, dataset='mushroom', version=1, random_generator=None):
        X, y = fetch_openml(dataset, version=version, return_X_y=True)
        # type annotations
        X : pd.DataFrame
        y : pd.Series
        # avoid nan, set nan as -1
        for col in X.select_dtypes('category'):
            # -1 in codes indicates NaN by pandas convention
            X[col] = X[col].cat.codes
        X = normalize(X)
        self.X, y = shuffle(X, y, random_state=random_generator)
        y_array = y.to_numpy().reshape(-1, 1)
        self.y_arm = OrdinalEncoder(
            dtype= int).fit_transform(y_array)
        self.P = self.X.shape[1]
        self.K = len(np.unique(self.y_arm))
        
        self._cursor = 0
        super().__init__(self.P, self.K, sigma2=0.0, random_generator=random_generator)
        
    def reshuffle(self, random_state=None):
        """
        Reshuffle the dataset and reset the cursor.
        """
        self.X, self.y_arm = shuffle(self.X, self.y_arm, random_state=random_state)
        self._cursor = 0
    
    def generate_covariates(self):
        cov = self.X[self._cursor, :].reshape(1, -1)
        self._cursor += 1
        return cov
    
    def reward_function(self, x):
        # Check if the input x matches the current data point
        x_cursor = self._cursor - 1
        if not np.all(x == self.X[x_cursor, :].reshape(1, -1)):
            raise ValueError("Input x does not match the current data point in the OpenMLScenario.")

        reward = np.zeros(self.K)
        reward[self.y_arm[x_cursor, 0]] = 1
        return {"outcome_mean": reward, "reward": reward}
    
    @property
    def max_draws(self):
        return self.X.shape[0]
    
    def finish(self):
        return self._cursor >= self.max_draws

class Friedman2Scenario(Scenario):
    def __init__(self, P, K, sigma2, lambda_val=3, random_generator=None):
        if P < 5:
            raise ValueError("Friedman is for P>=5")
        self.lambda_val = lambda_val
        super().__init__(P, K, sigma2, random_generator)
    
    def init_params(self):
        self.arm_offsets = self.rng.uniform(-5, 5, size=self.K)
        self.indices = self.rng.choice(self.P, size=5, replace=False)
    
    def reward_function(self, x):
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * np.sin(np.pi * x[0] * x[1]) + \
                      20 * (x[2] - 0.5) ** 2 + \
                      10 * x[3] + 5 * x[4] + \
                      self.lambda_val * self.arm_offsets * (
                        10 * np.sin(np.pi * x[self.indices[0]] * x[self.indices[1]]) + \
                        20 * (x[self.indices[2]] - 0.5) ** 2 + \
                        10 * x[self.indices[3]] + 5 * x[self.indices[4]]
                      )
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

def simulate(scenario, agents, n_draws):
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
    mem_agents = np.zeros(n_agents)
    
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

            # suppress memory usage for now
            mem_usage = 0 # asizeof.asizeof(agent)
            mem_agents[i] = mem_usage
    return cum_regrets, time_agents, mem_agents
