
from bart_playground.bandit.sim_util import LinearScenario, simulate
from bart_playground.bandit.basic_agents import LinearTSAgent
from bart_playground.bandit.rome.rome_agent import RoMEAgent, _featurize
from bart_playground.bandit.rome.baseline_agents import StandardTSAgent, ActionCenteredTSAgent, IntelligentPoolingAgent
from bart_playground.bandit.bcf_agent import BCFAgent
import matplotlib.pyplot as plt
import numpy as np

n_arms = 2
n_features = 10

# Create a scenario
scenario = LinearScenario(P=n_features, K=n_arms, sigma2=1.0)

# Create agents
linear_ts = LinearTSAgent(n_arms=n_arms, n_features=n_features)
rome_agent = RoMEAgent(n_arms=n_arms, n_features=n_features, 
                       featurize=_featurize,
                       t_max=500,
                       pool_users=False)
ac_agent = ActionCenteredTSAgent(n_arms=n_arms, n_features=n_features,
                               featurize=_featurize)
std_agent = StandardTSAgent(n_arms=n_arms, n_features=n_features,
                        featurize=_featurize)
ip_agent = IntelligentPoolingAgent(n_arms=n_arms, n_features=n_features,
                                    featurize=_featurize, t_max=500)
bcf_agent = BCFAgent(n_arms=n_arms, n_features=n_features, nskip=100,
                ndpost=10,
                nadd=3,
                nbatch=1)
                                    

# Run simulation
cum_regrets, time_agent = simulate(
    scenario=scenario,
    agents=[linear_ts, rome_agent, std_agent, ac_agent, ip_agent, bcf_agent],
    n_draws=500
)

def plot_cum_regrets(cum_regrets, agent_names, time_agent=None):
    """
    Plot cumulative regret curves for multiple agents.
    
    Parameters:
        cum_regrets (np.ndarray): Array of shape (n_draws, n_agents) with cumulative regrets.
        agent_names (list): List of agent names for the legend.
        time_agent (np.ndarray, optional): Array of shape (n_agents) with computation times.
    """
    plt.figure(figsize=(10, 6))
    
    n_draws, n_agents = cum_regrets.shape
    x = np.arange(n_draws)
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    for i in range(n_agents):
        plt.plot(x, cum_regrets[:, i], color=colors[i % len(colors)], 
                 label=f"{agent_names[i]} (Final: {cum_regrets[-1, i]:.2f})")
    
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title('Cumulative Regret Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Show computation time if provided
    if time_agent is not None:
        time_text = "Computation time (s):\n"
        for i, agent in enumerate(agent_names):
            time_text += f"{agent}: {time_agent[i]:.2f}s\n"
        plt.figtext(0.02, 0.02, time_text, fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()

# Usage example with your simulation results
agent_names = ["Linear TS", "RoME", "Standard TS", "Action-Centered TS", "Intelligent Pooling", "MBCF-TS"]
fig = plot_cum_regrets(cum_regrets, agent_names, time_agent)

# Save the figure if needed
plt.savefig('regret_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# You can also print some summary statistics
print("\nSummary Statistics:")
print(f"{'Agent':<10} {'Final Regret':<15} {'Computation Time (s)':<20}")
print("-" * 45)
for i, name in enumerate(agent_names):
    print(f"{name:<10} {cum_regrets[-1, i]:<15.2f} {time_agent[i]:<20.2f}")
