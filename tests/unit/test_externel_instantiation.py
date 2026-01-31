import numpy as np
import logging
from bart_playground.bandit.agents.external_agents import BartzTSAgent, StochTreeTSAgent
from bart_playground.bandit.agents.agent import instantiate_agents

logging.basicConfig(level=logging.INFO)

def test_instantiation():
    n_arms = 3
    n_features = 5
    agent_specs = [
        ("BartzTS", BartzTSAgent, {"need_random_state": True, "bart_kwargs": {"ndpost": 10}}),
        ("StochTreeTS", StochTreeTSAgent, {"need_random_state": True, "bart_kwargs": {"ndpost": 10}})
    ]
    
    # Simulate what src/experiment/simulation.py does
    agents = instantiate_agents(agent_specs, n_arms, n_features, random_state=42)
    
    for name, agent in zip(["BartzTS", "StochTreeTS"], agents):
        print(f"Instantiated {name}: {type(agent)}")
        
        # Test basic methods
        x = np.random.rand(n_features)
        arm = agent.choose_arm(x)
        print(f"  {name} chose arm: {arm}")
        
        # Test update_state (this triggers fit)
        # We need more than 5 selections if initial_random_selections is 5
        for i in range(20):
            x_i = np.random.rand(n_features)
            y_i = np.random.rand()
            agent.update_state(0, x_i, y_i)
        
        print(f"  {name} updated state 11 times. is_model_fitted: {agent.is_model_fitted}")
        
        if agent.is_model_fitted:
            arm_post = agent.choose_arm(x)
            print(f"  {name} chose arm after fit: {arm_post}")
            
            # Test diagnostics (probes)
            try:
                X_probes = np.random.rand(2, n_features)
                draws, n_post = agent.posterior_draws_on_probes(X_probes)
                print(f"  {name} diagnostics: draws shape {draws.shape}, n_post {n_post}")
                
                # Test DiagnosticsMixin methods
                diag = agent.diagnostics_probes(X_probes)
                print(f"  {name} diagnostics_probes success: {diag['metrics'].shape}")
            except Exception as e:
                print(f"  {name} diagnostics failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    test_instantiation()
