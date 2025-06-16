import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable
from tqdm import tqdm
import ray
import pickle

from bart_playground.bandit.sim_util import OpenMLScenario, simulate, Scenario
from bart_playground.bandit.bcf_agent import BCFAgent, BCFAgentPSOff
from bart_playground.bandit.basic_agents import SillyAgent, LinearTSAgent
from bart_playground.bandit.agent import BanditAgent

@ray.remote
def _run_single_simulation_remote(sim, scenario, agent_classes, class_to_agents, n_draws):
    """
    Remote function to run a single simulation with the given scenario and agents.
    """
    return _run_single_simulation(sim, scenario, agent_classes, class_to_agents, n_draws)

def _run_single_simulation(sim, scenario, agent_classes, class_to_agents, n_draws):
    """
    Run a single simulation with the given scenario and agents.
    
    Args:
        sim (int): Simulation number (used for random seed)
        scenario (Scenario): The scenario instance to use for simulation
        agent_classes (List): List of agent classes to instantiate
        agent_names (List[str]): Names for each agent
        n_draws (int): Number of draws per simulation
        
    Returns:
        Tuple: (sim_index, regrets, computation_times)
    """
    # Create agents with different seeds for this simulation
    sim_agents = class_to_agents(sim=sim, scenario_K=scenario.K, scenario_P=scenario.P, agent_classes=agent_classes)
    
    if hasattr(scenario, 'reshuffle'):
        print(f"Reshuffling scenario for simulation {sim}...")
        scenario.reshuffle(random_state=42+sim)

    # Run simulation
    cum_regrets, time_agents, mem_agents = simulate(scenario, sim_agents, n_draws=n_draws)
    
    # Return results for this simulation
    return sim, cum_regrets, time_agents #, sim_agents

def generate_simulation_data_for_agents(scenario: Scenario, agents: List[BanditAgent], agent_names: List[str], n_simulations: int = 10, n_draws: int = 500, parallel=True, class_to_agents: Callable = None):
    """
    Generate simulation data for multiple agents on a given scenario.
    
    Args:
        scenario (Scenario): The scenario instance to use for simulation
        agents (List[BanditAgent]): List of agent classes to instantiate and test
        agent_names (List[str]): Names for each agent for display purposes
        n_simulations (int): Number of simulation runs
        n_draws (int): Number of draws per simulation
        parallel (bool): Whether to run simulations in parallel using Ray
        
    Returns:
        Dict: Dictionary containing simulation results
    """
    n_agents = len(agents)
    all_regrets = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    all_times = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}

    if parallel:
        # Run simulations in parallel
        print(f"Running {n_simulations} simulations in parallel using Ray...")
        # Launch all simulation tasks
        results_futures = [
            _run_single_simulation_remote.remote(sim, scenario, agents, class_to_agents, n_draws)
            for sim in range(n_simulations)
        ]
        # Wait for all tasks to complete and retrieve the results.
        results = [ray.get(future) for future in tqdm(results_futures, desc="Simulating")]
    else:
        # Run simulations sequentially
        results = []
        if n_simulations > 1:
            for sim in tqdm(range(n_simulations), desc="Simulating sequentially"):
                result = _run_single_simulation(sim, scenario, agents, class_to_agents, n_draws)
                internal_scenario_name = scenario.__class__.__name__
                if isinstance(scenario, OpenMLScenario):
                    internal_scenario_name = scenario.dataset_name
                print(f"Saving temporary results for simulation {sim} in {internal_scenario_name}...")
                dir_name = "test_results"
                if os.path.exists(dir_name) is False:
                    os.makedirs(dir_name)
                pickle.dump(result, open(f"{dir_name}/temp_simulation_{internal_scenario_name}_{sim}.pkl", "wb"))
                results.append(result)
        else:
            print("Running a single simulation...")
            result = _run_single_simulation(0, scenario, agents, class_to_agents, n_draws)
            results.append(result)
    
    # all_agents = []
    # Process results from parallel jobs
    for sim, cum_regrets, time_agents in results:
        # Store results
        for i, name in enumerate(agent_names):
            all_regrets[name][sim, :] = cum_regrets[:, i]
            all_times[name][sim, :] = time_agents[:, i]
        # all_agents.append(sim_agents)

    return {
        'regrets': all_regrets,
        'times': all_times,
        # 'temp_agent': all_agents  # Store the last agent instance for reference
    }


def compare_agents_across_scenarios(scenarios: Dict[str, Scenario], n_simulations: int = 10, max_draws: int = 500, 
    agent_classes = [SillyAgent, LinearTSAgent, BCFAgent], 
    agent_names = ["Random", "LinearTS", "BCF"],
    parallel: bool = True,
    class_to_agents : Callable = None):
    """
    Compare multiple agents across different scenarios.
    
    Args:
        scenarios (Dict[str, Scenario]): Dictionary of scenario name to scenario instance
        n_simulations (int): Number of simulations per scenario
        max_draws (int): Max number of draws per simulation
        parallel (bool): Whether to run simulations in parallel using Ray
        agent_classes (List[BanditAgent]): List of agent classes to instantiate and test
        agent_names (List[str]): Names for each agent for result dict keys
    Returns:
        Dict: Results for each scenario and agent
    """
    if parallel:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)
        print("Ray initialized for parallel simulations.")
    
    results = {}
    
    try:
        for scenario_name, scenario in scenarios.items():
            print(f"\nEvaluating {scenario_name} scenario...")
            n_draws = int(min(max_draws, scenario.max_draws))
            scenario_results = generate_simulation_data_for_agents(
                scenario=scenario,
                agents=agent_classes,
                agent_names=agent_names,
                n_simulations=n_simulations,
                n_draws=n_draws,
                parallel=parallel,
                class_to_agents=class_to_agents
            )
            results[scenario_name] = scenario_results
    finally:
        # IMPORTANT: Shut down Ray when all work is finished.
        if parallel and ray.is_initialized():
            ray.shutdown()
            print("Ray has been shut down.")
    
    return results


def plot_comparison_results(results: Dict[str, Dict], save_path: str = None, show_time = False):
    """
    Plot comparison results across all scenarios and agents.
    
    Args:
        results (Dict): Results from compare_agents_across_scenarios
        save_path (str, optional): Path to save the figure
    """
    n_scenarios = len(results)
    n_rows = (n_scenarios + 1) // 2  # Ceiling division for number of rows
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = [axes]
    
    # Get all agent names and generate a consistent color mapping
    all_agent_names = set()
    for scenario_results in results.values():
        all_agent_names.update(scenario_results['regrets'].keys())
    
    # Use the default color cycle from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    # Map each agent to a color
    agent_colors = {name: colors[i % len(colors)] for i, name in enumerate(sorted(all_agent_names))}
    
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
            sd_regret = np.std(agent_regrets, axis=0)
            lower_ci = mean_regret - sd_regret # np.percentile(agent_regrets, 25, axis=0)
            upper_ci = mean_regret + sd_regret # np.percentile(agent_regrets, 75, axis=0)
            
            n_draws = mean_regret.shape[0]
            # Plot mean regret
            ax.plot(range(n_draws), mean_regret, label=f"{agent_name}", 
                   color=agent_colors[agent_name], linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(range(n_draws), lower_ci, upper_ci, 
                          color=agent_colors[agent_name], alpha=0.2)
        
        if show_time:
            # Add computation time annotations
            raise NotImplementedError("Computation time plotting not implemented yet.")
        
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
            agent_time = np.sum(agent_times, axis=1)
            print(f"  {agent_name}: {np.mean(agent_time):.4f} (±{np.std(agent_time):.4f})")
        
        print("\n" + "=" * 40)
