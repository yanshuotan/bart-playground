# Off Policy Evaluation (OPE) for Bandit Problems
import time
from typing import Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray

from bart_playground.bandit.agents.agent import BanditAgent

def sim_propensities(data: dict[str, NDArray[Any] | pd.DataFrame], 
                   agents: list[BanditAgent], 
                   agent_names: list[str] | None = None,
                   n_choices_per_iter: int = 1,
                   show_progress: bool = True,
                   return_final_states: bool = False) -> dict[str, dict[str, NDArray[Any] | BanditAgent]]:
    """Simulate bandit agents and generate actions for off-policy evaluation.
    
    Args:
        data: Dictionary containing 'context', 'action', and 'reward' arrays
        agents: List of bandit agents to evaluate
        agent_names: Optional list of agent names
        n_choices_per_iter: Number of choose_arm calls per iteration to calculate agent_actions probability
        show_progress: Whether to show progress bar
        return_final_states: Whether to return final states of agents
    
    Returns:
        Dict containing simulation results for each agent:
        - 'actions': array of logged actions (0-indexed)
        - 'agent_actions': per-time per-arm selection frequency (counts / n_choices_per_iter)
        - 'rewards': array of rewards
        - 'final_state': agent's final state (if return_final_states=True)
    """
    
    # Extract and prepare data
    context_matrix: NDArray[Any] = np.array(data['context'])
    actions: NDArray[Any] = np.array(data['action'])
    rewards: NDArray[Any] = np.array(data['reward'])
    
    if context_matrix.ndim == 1:
        context_matrix = context_matrix.reshape(-1, 1)
    
    # Convert to 0-indexed actions if needed
    if actions.min() == 1:
        actions = actions - 1
    
    n_draw: int = context_matrix.shape[0]
    
    # Prepare agent names
    if agent_names is None:
        agent_names = [f"Agent_{i}" for i in range(len(agents))]
    
    simulation_results: dict[str, dict[str, NDArray[Any] | BanditAgent]] = {}
    
    for agent_idx, agent in enumerate(agents):
        agent_name = agent_names[agent_idx]
        print(f"Simulating agent: {agent_name}")
        
        # Determine number of arms for this agent
        n_arms: int = int(getattr(agent, 'n_arms'))
        # Store per-time probabilities of selecting each arm (counts / n_choices_per_iter)
        agent_actions: NDArray[Any] = np.zeros((n_draw, n_arms), dtype=float)
        # Store computation time per step
        agent_times: NDArray[Any] = np.zeros(n_draw, dtype=float)
        
        iterator = tqdm(range(n_draw), desc=f"Simulating {agent_name}") if show_progress else range(n_draw)
        
        for t in iterator:
            x_t = context_matrix[t]
            a_t = int(actions[t])
            r_t = float(rewards[t])
            
            t0 = time.time()
            # Get agent's action multiple times and track per-arm frequencies
            counts = np.zeros(n_arms, dtype=int)
            for _ in range(n_choices_per_iter):
                agent_action = agent.choose_arm(x_t)
                counts[agent_action] += 1
            
            # Record per-arm selection frequency at time t
            agent_actions[t, :] = counts.astype(float) / float(n_choices_per_iter)
            
            # Update agent state
            _ = agent.update_state(a_t, x_t, r_t)
            agent_times[t] = time.time() - t0
        
        result_dict: dict[str, NDArray[Any] | BanditAgent] = {
            'actions': actions, # This is the real action that was in the data
            'agent_actions': agent_actions, # This is the action that the agent chose
            'rewards': rewards,
            'times': agent_times
        }
        
        if return_final_states:
            result_dict['final_state'] = agent
            
        simulation_results[agent_name] = result_dict
    
    return simulation_results
