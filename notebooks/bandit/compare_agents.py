import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from bart_playground.bandit.sim_util import simulate, Scenario, LinearScenario, LinearOffsetScenario, OffsetScenario, FriedmanScenario
from bart_playground.bandit.bcf_agent import BCFAgent
from bart_playground.bandit.basic_agents import SillyAgent, LinearTSAgent
from bart_playground.bandit.agent import BanditAgent


def generate_simulation_data_for_agents(scenario: Scenario, agents: List[BanditAgent], agent_names: List[str], n_simulations: int = 10, n_draws: int = 500):
    """
    Generate simulation data for multiple agents on a given scenario.
    
    Args:
        scenario (Scenario): The scenario instance to use for simulation
        agents (List[BanditAgent]): List of agent classes to instantiate and test
        agent_names (List[str]): Names for each agent for display purposes
        n_simulations (int): Number of simulation runs
        n_draws (int): Number of draws per simulation
        
    Returns:
        Dict: Dictionary containing simulation results
    """
    n_agents = len(agents)
    all_regrets = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    all_times = {name: np.zeros(n_simulations) for name in agent_names}
    
    for sim in tqdm(range(n_simulations), desc=f"Simulating"):
        # Create agents with different seeds for each simulation
        sim_agents = []
        for agent_cls in agents:
            if agent_cls == BCFAgent:
                # BCFAgent with fixed parameters (nadd=2, nbatch=1)
                agent = BCFAgent(
                    n_arms=scenario.K,
                    n_features=scenario.P,
                    nskip=100,
                    ndpost=10,
                    nadd=2,
                    nbatch=1,
                    random_state=1000 + sim
                )
            elif agent_cls == LinearTSAgent:
                agent = LinearTSAgent(
                    n_arms=scenario.K,
                    n_features=scenario.P,
                    v = 1
                    #R=10.0
                )
            else:
                # For other agents, just initialize with standard params
                agent = agent_cls(
                    n_arms=scenario.K,
                    n_features=scenario.P
                )
            sim_agents.append(agent)
        
        # Run simulation
        cum_regrets, time_agents = simulate(scenario, sim_agents, n_draws=n_draws)
        
        # Store results
        for i, name in enumerate(agent_names):
            all_regrets[name][sim, :] = cum_regrets[:, i]
            all_times[name][sim] = time_agents[i]
    
    return {
        'regrets': all_regrets,
        'times': all_times
    }


def compare_agents_across_scenarios(scenarios: Dict[str, Scenario], n_simulations: int = 10, n_draws: int = 500):
    """
    Compare multiple agents across different scenarios.
    
    Args:
        scenarios (Dict[str, Scenario]): Dictionary of scenario name to scenario instance
        n_simulations (int): Number of simulations per scenario
        n_draws (int): Number of draws per simulation
        
    Returns:
        Dict: Results for each scenario and agent
    """
    # Define agents to compare
    agent_classes = [SillyAgent, LinearTSAgent, BCFAgent]
    agent_names = ["Random", "LinearTS", "BCF"]
    
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\nEvaluating {scenario_name} scenario...")
        scenario_results = generate_simulation_data_for_agents(
            scenario=scenario,
            agents=agent_classes,
            agent_names=agent_names,
            n_simulations=n_simulations,
            n_draws=n_draws
        )
        results[scenario_name] = scenario_results
    
    return results


def plot_comparison_results(results: Dict[str, Dict], n_draws: int, save_path: str = None):
    """
    Plot comparison results across all scenarios and agents.
    
    Args:
        results (Dict): Results from compare_agents_across_scenarios
        n_draws (int): Number of draws per simulation
        save_path (str, optional): Path to save the figure
    """
    n_scenarios = len(results)
    n_rows = (n_scenarios + 1) // 2  # Ceiling division for number of rows
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = [axes]
    
    # Define colors for each agent
    agent_colors = {
        'Random': '#1f77b4',
        'LinearTS': '#ff7f0e',
        'BCF': '#2ca02c',
    }
    
    # Plot each scenario
    scenario_idx = 0
    for scenario_name, scenario_results in results.items():
        row = scenario_idx // 2
        col = scenario_idx % 2
        
        ax = axes[row][col]
        
        regrets = scenario_results['regrets']
        times = scenario_results['times']
        
        # For each agent
        for agent_name, agent_regrets in regrets.items():
            mean_regret = np.mean(agent_regrets, axis=0)
            lower_ci = np.percentile(agent_regrets, 25, axis=0)
            upper_ci = np.percentile(agent_regrets, 75, axis=0)
            
            # Plot mean regret
            ax.plot(range(n_draws), mean_regret, label=f"{agent_name}", 
                   color=agent_colors[agent_name], linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(range(n_draws), lower_ci, upper_ci, 
                          color=agent_colors[agent_name], alpha=0.2)
        
        # Add computation time annotations
        for agent_name, agent_times in times.items():
            mean_time = np.mean(agent_times)
            ax.annotate(f"{agent_name}: {mean_time:.2f}s", 
                       xy=(0.5, 0.02 + list(times.keys()).index(agent_name) * 0.05),
                       xycoords='axes fraction',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        ax.set_title(f"{scenario_name} Scenario", fontsize=14)
        ax.set_xlabel("Draw", fontsize=12)
        ax.set_ylabel("Cumulative Regret", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        scenario_idx += 1
    
    # If there's an odd number of scenarios, remove the empty subplot
    if n_scenarios % 2 == 1:
        fig.delaxes(axes[-1][-1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()


def print_summary_results(results: Dict[str, Dict]):
    """
    Print a summary of results across scenarios and agents.
    
    Args:
        results (Dict): Results from compare_agents_across_scenarios
    """
    # For each scenario
    for scenario_name, scenario_results in results.items():
        print(f"\n=== {scenario_name} Scenario ===")
        regrets = scenario_results['regrets']
        times = scenario_results['times']
        
        # Print final regrets
        print("\nFinal cumulative regrets (mean ± std):")
        for agent_name, agent_regrets in regrets.items():
            final_regrets = agent_regrets[:, -1]  # Get the last column (final regrets)
            print(f"  {agent_name}: {np.mean(final_regrets):.2f} (±{np.std(final_regrets):.2f})")
        
        # Print computation times
        print("\nAverage computation times (seconds):")
        for agent_name, agent_times in times.items():
            print(f"  {agent_name}: {np.mean(agent_times):.4f} (±{np.std(agent_times):.4f})")
        
        print("\n" + "=" * 40)
