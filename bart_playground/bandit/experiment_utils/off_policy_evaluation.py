# Off Policy Evaluation (OPE) for Bandit Problems

from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from bart_playground.bandit.agents.agent import BanditAgent

def evaluate_agents(data: Dict[str, Union[np.ndarray, pd.DataFrame]], 
                   agents: List[BanditAgent], 
                   n_arms: int,
                   agent_names: Optional[List[str]] = None,
                   propensity_scores: Optional[np.ndarray] = None,
                   show_progress: bool = True) -> Dict[str, float]:
    """Evaluate bandit agents using off-policy evaluation with importance sampling."""
    
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
    
    if propensity_scores is None:
        raise ValueError("Propensity scores are required for off-policy evaluation.")
    else:
        ps_matrix = propensity_scores
        if ps_matrix.shape[0] != n_draw or ps_matrix.shape[1] != n_arms:
            raise ValueError("Provided propensity scores shape does not match the context matrix.")
    
    # Prepare agent names
    if agent_names is None:
        agent_names = [f"Agent_{i}" for i in range(len(agents))]
    
    agent_results = {}
    
    for agent_idx, agent in enumerate(agents):
        agent_name = agent_names[agent_idx]
        print(f"Evaluating agent: {agent_name}")
        
        numerator = 0.0
        denominator = 0.0
        
        iterator = tqdm(range(n_draw), desc=f"Simulating {agent_name}") if show_progress else range(n_draw)
        
        for t in iterator:
            x_t = context_matrix[t]
            a_t = int(actions[t])
            r_t = float(rewards[t])
            
            # Get agent's action (random policy)
            # TODO

            # Get agent's action (deterministic policy)
            agent_action = agent.choose_arm(x_t)
            pi_t = 1.0 if agent_action == a_t else 0.0
            
            # Logging policy propensity
            p_t = ps_matrix[t, a_t]
            
            # Importance weight
            w_t = pi_t / p_t
            
            numerator += w_t * r_t
            denominator += w_t
            
            # Update agent state
            agent.update_state(a_t, x_t, r_t)
        
        # Compute policy value estimate
        V_hat = numerator / denominator if denominator > 0 else 0.0
        if denominator == 0:
            print(f"Warning: Zero denominator for agent {agent_name}")
        
        agent_results[agent_name] = V_hat
        print(f"Estimated policy value for {agent_name}: {V_hat:.4f}")
    
    return agent_results
