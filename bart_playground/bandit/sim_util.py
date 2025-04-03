import numpy as np
from tqdm import tqdm
import time

'''
### Scenario Classes

We define a base `Scenario` class and a `LinearScenario` subclass. The `generate_covariates` method produces a vector of features (here, sampled from a standard normal distribution), and the `reward_function` computes the expected reward for each arm and adds noise.
'''
class Scenario:
    def __init__(self, P, K, sigma2):
        """
        Parameters:
            P (int): Number of covariates (features).
            K (int): Number of arms (including control).
            sigma2 (float): Noise variance.
        """
        self.P = P
        self.K = K
        self.sigma2 = sigma2

    def generate_covariates(self):
        # Generate a vector of P covariates (features) sampled from a normal distribution.
        return np.random.normal(0, 1, size=self.P).astype(np.float32)

    def reward_function(self, x):
        """
        Given a feature vector x, compute:
          - outcome_mean: Expected rewards for each arm.
          - reward: Outcome_mean plus noise.
        Must be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

class LinearScenario(Scenario):
    def __init__(self, P, K, sigma2):
        super().__init__(P, K, sigma2)
        # Generate a K x P matrix of arm-specific coefficients uniformly between -1 and 1.
        self.mu_a = np.random.uniform(-1, 1, size=(K, P))
    
    def reward_function(self, x):
        # Compute noise for each arm.
        epsilon_t = np.random.normal(0, np.sqrt(self.sigma2), size=self.K)
        # Compute expected rewards (outcome means) for each arm.
        outcome_mean = 10 * self.mu_a.dot(x)
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}
    
class LinearOffsetScenario(Scenario):
    def __init__(self, P, K, sigma2):
        super().__init__(P, K, sigma2)
        # Generate a 1 x P matrix of global coefficients uniformly between -1 and 1.
        self.mu = np.random.uniform(-1, 1, size=(1, P))
        # Generate a K x 1 matrix of arm-specific offsets uniformly between -5 and 5.
        self.arm_offsets = np.random.uniform(-5, 5, size=K)
    
    def reward_function(self, x):
        epsilon_t = np.random.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * self.mu.dot(x) + self.arm_offsets
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

class OffsetScenario(Scenario):
    def __init__(self, P, K, sigma2, lambda_val=3):
        super().__init__(P, K, sigma2)
        # Uniformly distributed covariates in unit cube
        self.mu = np.random.uniform(-1, 1, size=(1, P))
        self.lambda_val = lambda_val
        # Generate a K x 1 matrix of arm-specific offsets uniformly between -5 and 5.
        self.arm_offsets = np.random.uniform(-5, 5, size=K)
    
    def reward_function(self, x):
        epsilon_t = np.random.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * np.sin(self.mu.dot(x)) + self.lambda_val * self.arm_offsets
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

class FriedmanScenario(Scenario):
    def __init__(self, P, K, sigma2, lambda_val=3):
        super().__init__(P, K, sigma2)
        if P < 5:
            raise ValueError("Friedman is for P>=5")
        
        self.lambda_val = lambda_val
    
    def reward_function(self, x):
        epsilon_t = np.random.normal(0, np.sqrt(self.sigma2), size=self.K)
        outcome_mean = 10 * np.sin(np.pi * x[0] * x[1]) + \
                      20 * (x[2] - 0.5) ** 2 + \
                      10 * x[3] + 5 * x[4] + \
                      self.lambda_val * np.arange(1, self.K+1)
        return {"outcome_mean": outcome_mean, "reward": outcome_mean + epsilon_t}

'''
### Simulation Function

The `simulate` function takes a scenario, a list of agents, and the number of draws. For each draw:

1. Generate covariates.
2. Compute the outcome means and rewards.
3. For each agent, choose an arm based on the current covariates.
4. Update cumulative regret for the agent.
5. Update the agent’s state with the observed reward.

We use `tqdm` to track progress.
'''
def simulate(scenario, agents, n_draws):
    """
    Simulate a bandit problem using the provided scenario and agents.
    
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
    time_agent = np.zeros(n_agents)
    
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
            time_agent[i] += time.time() - t0
    return cum_regrets, time_agent
