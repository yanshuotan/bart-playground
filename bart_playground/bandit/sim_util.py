from typing import Optional, Union
import numpy as np
from tqdm import tqdm
import time, logging
import pandas as pd
import math
from sklearn.datasets import fetch_openml, make_friedman1
from sklearn.preprocessing import normalize, OrdinalEncoder
from sklearn.utils import shuffle

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
        self.rng = np.random.default_rng(seed)

    def shuffle(self, random_state=None):
        """
        Shuffle the scenario and reset parameters. Default implementation just reinitializes.
        """
        if random_state is not None:
            self.set_seed(random_state)
        self.init_params()

    def generate_covariates(self):
        # Generate a vector of P covariates (features) sampled from a uniform distribution.
        return self.rng.uniform(-1, 1, size=self.P).astype(np.float32)

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

class LinearScenario(Scenario):
    def __init__(self, P, K, sigma2, d=None, random_generator=None):
        self.d = d if d is not None else P
        super().__init__(P, K, sigma2, random_generator)

    def init_params(self):
        # Generate a K x P (d useful) matrix of arm-specific coefficients normally distributed with mean 0 and std 1.
        self.mu_a = self.rng.normal(0, 1, size=(self.K, self.d))
        self.mu_a = np.append(self.mu_a, np.zeros((self.K, self.P - self.d)), axis=1)

    def reward_function(self, x):
        # Compute noise for each arm.
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        # Compute expected rewards (outcome means) for each arm.
        outcome_mean = self.mu_a.dot(x)
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

class GLMScenario(Scenario):
    def __init__(self, P, K, sigma2, random_generator=None):
        super().__init__(P, K, sigma2, random_generator)

    def init_params(self):
        self.mu_a = self.rng.normal(0, 1, size=(self.K, self.P))
        self.mu_a = normalize(self.mu_a, axis=1)
    
    def reward_function(self, x):
        # Compute noise for each arm.
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = self.mu_a.dot(x)
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

def _friedman1(x):
    _f1 = 10 * np.sin(np.pi * x[0] * x[1]) + \
            20 * (x[2] - 0.5) ** 2 + \
            10 * x[3] + 5 * x[4]
    return _f1

def _friedman_helper(x):
    # x_1 = self.rng.uniform(0, 100)
    # x_2 = self.rng.uniform(40*math.pi, 560*math.pi)
    # x_3 = self.rng.uniform(0, 1)
    # x_4 = self.rng.uniform(1, 11)
    res = x.copy()
    res[0] = res[0] * 100
    res[1] = res[1] * 520 * math.pi + 40 * math.pi
    res[2] = res[2] * 1
    res[3] = res[3] * 10 + 1
    return res

def _friedman2(x):
    x = _friedman_helper(x)
    _f2 = np.sqrt(x[0]**2 +
                   (x[1]*x[2] - 1/(x[1]*x[3]))**2)
    return _f2 / 125 # equivalent to standard deviation of 125 by mlbench
    
def _friedman3(x):
    x = _friedman_helper(x)
    _f3 = np.arctan((x[1]*x[2] - 1/(x[1]*x[3])) / x[0])
    return _f3 / 0.1

class FriedmanScenario(Scenario):
    def __init__(self, P, K, sigma2, random_generator=None, f_type='friedman1'):
        if K != 2:
            raise ValueError("Friedman is for K=2")
        if f_type not in ['friedman1', 'friedman2', 'friedman3']:
            raise ValueError("f_type must be one of 'friedman1', 'friedman2', 'friedman3'")
        elif f_type == 'friedman1':
            self._friedman = _friedman1
            if P < 5:
                raise ValueError("Friedman1 requires P >= 5")
        elif P < 4:
            raise ValueError("Friedman2 and Friedman3 require P >= 4")
        else:
            if f_type == 'friedman2':
                self._friedman = _friedman2
            else:
                self._friedman = _friedman3
        
        super().__init__(P, K, sigma2, random_generator)
        
    def generate_covariates(self):
        # Uniform [0, 1]
        x = self.rng.uniform(0, 1, size=self.P).astype(np.float32)
        return x
    
    def init_params(self):
        pass

    def reward_function(self, x):
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        x_reverse = x[::-1]
        outcome_mean = np.hstack([self._friedman(x), self._friedman(x_reverse)], dtype=np.float32)
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}
    
class LinearFriedmanScenario(Scenario):
    def __init__(self, P, K, sigma2, random_generator=None):
        assert P >= 5 and K == 2, "LinearFriedmanScenario is for P>=5, K=2"
        super().__init__(P, K, sigma2, random_generator)

    def init_params(self):
        self.mu_a = self.rng.normal(0, 1, size=(1, self.P))

    def reward_function(self, x):
        epsilon_t = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean0 = self.mu_a.dot(x)
        outcome_mean = np.hstack([outcome_mean0, outcome_mean0 + _friedman1(x)], dtype=np.float32)
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}
    
class OpenMLScenario(Scenario):
    def __init__(self, dataset='mushroom', version=1, random_generator:Union[np.random.Generator, int, None]=None, **kwargs):
        X, y = fetch_openml(dataset, version=version, return_X_y=True, **kwargs)
        # type annotations
        X : pd.DataFrame
        y : pd.Series
        for col in X.select_dtypes('category'):
            # -1 in codes indicates NaN by pandas convention
            X[col] = X[col].cat.codes
        
        # cat_cols = X.select_dtypes('category').columns
# 
        # # one-hot encode them
        # X = pd.get_dummies(
        #     X,
        #     columns=cat_cols,
        #     prefix=cat_cols,       # keep the original names as prefixes
        #     # drop_first=True,    # drops the first real level
        #     # dummy_na=True
        # )
        
        X_arr = X.to_numpy() # normalize(X)
        y_arr = y.to_numpy().reshape(-1, 1)
        # Encode categorical labels as integers
        y_encoded = OrdinalEncoder(dtype=int).fit_transform(y_arr)
        
        self.original_X = X_arr.copy()
        self.original_y = y_encoded.copy()

        P = X_arr.shape[1]
        K = len(np.unique(y_encoded))
        self.dataset_name = dataset

        # Use the provided random_generator or create one with the seed
        if isinstance(random_generator, np.random.Generator):
            self.rng = random_generator
        elif isinstance(random_generator, int):
            self.rng = np.random.default_rng(random_generator)
        else:
            self.rng = np.random.default_rng()

        super().__init__(P, K, sigma2=0.0, random_generator=self.rng)

    def init_params(self):
        random_state = self.rng.integers(0, 2**31 - 1)
        self.X, self.y = shuffle(self.original_X, self.original_y, random_state=random_state)
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
        reward[self.y[x_cursor, 0]] = 1
        return {"outcome_mean": reward, "reward": reward}
    
    @property
    def max_draws(self):
        return self.X.shape[0]

class FriedmanDScenario(Scenario):
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

sim_logger = logging.getLogger("bandit_simulator")

def simulate(scenario, agents, n_draws, agent_names: list[str]=[]):
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
        
        # Log current status every sqrt(n_draws) draws.
        logging_frequency = int(math.sqrt(n_draws))
        if (draw + 1) % logging_frequency == 0 or draw == n_draws - 1:   
                df = pd.DataFrame({
                    "AgentName":       agent_names,
                    "CumRegret":   cum_regrets[draw, :],
                    "CumTime":     np.sum(time_agents[:draw, :], axis=0)
                })
                # log it with fixed‐width columns and 6 decimal places
                sim_logger.debug(f"Draw {draw+1}/{n_draws}: \n" + df.to_string(
                    index=False,
                    float_format="%.6f"
                ))
            
    for agent in agents:
        if hasattr(agent, 'clean_up'):
            agent.clean_up()
        
    return cum_regrets, time_agents

