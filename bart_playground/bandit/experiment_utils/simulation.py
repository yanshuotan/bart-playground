import numpy as np
from tqdm import tqdm
import time, logging
import pandas as pd
import math

sim_logger = logging.getLogger(__name__)

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


