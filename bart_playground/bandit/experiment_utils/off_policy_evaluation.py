# Off Policy Evaluation (OPE) for Bandit Problems

import warnings
from typing import Dict, List, Union, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from bart_playground.bandit.agents.agent import BanditAgent

def estimate_propensity_scores(context_matrix: np.ndarray, actions: np.ndarray, n_arms: int) -> np.ndarray:
    """Estimate propensity scores using multinomial logistic regression."""
    n_samples = context_matrix.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if n_arms == 2:
            model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
            model.fit(context_matrix, actions)
            probs = model.predict_proba(context_matrix)
            
            if probs.shape[1] == 1:
                prob_class = model.classes_[0]
                ps_matrix = np.zeros((n_samples, n_arms))
                ps_matrix[:, prob_class] = probs[:, 0]
                ps_matrix[:, 1 - prob_class] = 1 - probs[:, 0]
            else:
                ps_matrix = probs
        else:
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                     random_state=42, max_iter=1000)
            le = LabelEncoder()
            actions_encoded = le.fit_transform(actions)
            model.fit(context_matrix, actions_encoded)
            probs = model.predict_proba(context_matrix)
            
            ps_matrix = np.zeros((n_samples, n_arms))
            for i, original_action in enumerate(le.classes_):
                if original_action < n_arms:
                    ps_matrix[:, original_action] = probs[:, i]
            
            # Handle missing classes
            missing_prob = 1e-8
            for arm in range(n_arms):
                if arm not in le.classes_:
                    ps_matrix[:, arm] = missing_prob
            
            # Renormalize
            row_sums = ps_matrix.sum(axis=1, keepdims=True)
            ps_matrix = ps_matrix / np.maximum(row_sums, 1e-8)
    
    return ps_matrix

def instantiate_agents(agent_specs: List[Tuple[str, type, Dict]], 
                              n_arms: int, n_features: int, 
                              sim: int = 0) -> List[BanditAgent]:
    """Create fresh agent instances using the same pattern as compare_agents.py"""
    agents = []
    for name, cls, base_kwargs in agent_specs:
        kwargs = base_kwargs.copy()
        kwargs['n_arms'] = n_arms
        kwargs['n_features'] = n_features
        
        # Offset seed for reproducibility
        if 'random_state' in base_kwargs:
            kwargs['random_state'] = sim
            
        agents.append(cls(**kwargs))
    return agents

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
        print("Estimating propensity scores...")
        ps_matrix = estimate_propensity_scores(context_matrix, actions, n_arms)
        ps_matrix = np.clip(ps_matrix, 1e-8, 1.0)
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
