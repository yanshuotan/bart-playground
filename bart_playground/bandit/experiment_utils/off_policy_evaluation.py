# Off Policy Evaluation (OPE) for Bandit Problems

from typing import Dict, List, Union, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from bart_playground.bandit.agents.agent import BanditAgent

def sim_propensities(data: Dict[str, Union[np.ndarray, pd.DataFrame]], 
                   agents: List[BanditAgent], 
                   n_arms: int,
                   agent_names: Optional[List[str]] = None,
                   n_choices_per_iter: int = 1,
                   show_progress: bool = True,
                   return_final_states: bool = False) -> Dict[str, Dict[str, Union[np.ndarray, BanditAgent]]]:
    """Simulate bandit agents and generate actions for off-policy evaluation.
    
    Args:
        data: Dictionary containing 'context', 'action', and 'reward' arrays
        agents: List of bandit agents to evaluate
        n_arms: Number of arms in the bandit problem
        agent_names: Optional list of agent names
        n_choices_per_iter: Number of choose_arm calls per iteration to calculate pi_t probability
        show_progress: Whether to show progress bar
        return_final_states: Whether to return final states of agents
    
    Returns:
        Dict containing simulation results for each agent:
        - 'actions': array of logged actions
        - 'pi_t': array of policy probabilities (proportion of correct choices out of n_choices_per_iter)
        - 'rewards': array of rewards
        - 'final_state': agent's final state (if return_final_states=True)
    """
    
    # Extract and prepare data
    context_matrix = np.array(data['context'])
    actions = np.array(data['action'])
    rewards = np.array(data['reward'])
    
    if context_matrix.ndim == 1:
        context_matrix = context_matrix.reshape(-1, 1)
    
    # Convert to 0-indexed actions if needed
    if actions.min() == 1:
        actions = actions - 1
    
    n_draw = context_matrix.shape[0]
    
    # Prepare agent names
    if agent_names is None:
        agent_names = [f"Agent_{i}" for i in range(len(agents))]
    
    simulation_results = {}
    
    for agent_idx, agent in enumerate(agents):
        agent_name = agent_names[agent_idx]
        print(f"Simulating agent: {agent_name}")
        
        pi_t_values = np.zeros(n_draw)
        
        iterator = tqdm(range(n_draw), desc=f"Simulating {agent_name}") if show_progress else range(n_draw)
        
        for t in iterator:
            x_t = context_matrix[t]
            a_t = int(actions[t])
            r_t = float(rewards[t])
            
            # Get agent's action multiple times and calculate probability
            correct_choices = 0
            for _ in range(n_choices_per_iter):
                agent_action = agent.choose_arm(x_t)
                if agent_action == a_t:
                    correct_choices += 1
            
            # Calculate pi_t as proportion of correct choices
            pi_t_values[t] = correct_choices / n_choices_per_iter
            
            # Update agent state
            _ = agent.update_state(a_t, x_t, r_t)
        
        result_dict: Dict[str, Union[np.ndarray, BanditAgent]] = {
            'actions': actions,
            'pi_t': pi_t_values,
            'rewards': rewards
        }
        
        if return_final_states:
            result_dict['final_state'] = agent
            
        simulation_results[agent_name] = result_dict
    
    return simulation_results


def calculate_policy_values(simulation_results: Dict[str, Dict[str, Union[np.ndarray, BanditAgent]]], 
                          propensity_scores: np.ndarray) -> Dict[str, float]:
    """Calculate policy values using importance sampling from simulation results.
    
    Args:
        simulation_results: Dictionary containing simulation results from sim_propensities
        propensity_scores: Matrix of propensity scores (n_samples x n_arms)
        
    Returns:
        Dictionary mapping agent names to their estimated policy values
    """
    
    agent_results = {}
    
    for agent_name, results in simulation_results.items():
        pi_t = results['pi_t']
        rewards = results['rewards']
        actions = results['actions']
        
        # Cast to ensure proper types
        pi_t_arr = np.array(pi_t)
        rewards_arr = np.array(rewards)
        actions_arr = np.array(actions)
        
        # Extract propensity scores for logged actions
        propensity_values = np.array([propensity_scores[t, int(actions_arr[t])] for t in range(len(actions_arr))])
        
        # Calculate importance weights: w_t = pi_t / p_t
        importance_weights = pi_t_arr / propensity_values
        
        numerator = np.sum(importance_weights * rewards_arr)
        denominator = np.sum(importance_weights)
        
        # Compute policy value estimate
        V_hat = numerator / denominator if denominator > 0 else 0.0
        if denominator == 0:
            print(f"Warning: Zero denominator for agent {agent_name}")
        
        agent_results[agent_name] = V_hat
        print(f"Estimated policy value for {agent_name}: {V_hat:.4f}")
    
    return agent_results
