from math import log
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from joblib import Parallel, delayed

from bart_playground.bandit.ensemble_agent import EnsembleAgent
from bart_playground.bandit.sim_util import simulate, Scenario, LinearScenario, LinearOffsetScenario, OffsetScenario, FriedmanScenario
from bart_playground.bandit.bcf_agent import BCFAgent, BCFAgentPSOff
from bart_playground.bandit.basic_agents import SillyAgent, LinearTSAgent
from bart_playground.bandit.agent import BanditAgent
# Add ROME agent imports
from bart_playground.bandit.rome.rome_agent import RoMEAgent, _featurize
from bart_playground.bandit.rome.baseline_agents import StandardTSAgent, ActionCenteredTSAgent, IntelligentPoolingAgent

def class_to_agents(sim, scenario, agent_classes: List[Any], n_draws = 500) -> Tuple[List[BanditAgent], List[str]]:
    """
    Convert agent classes to instances and names.
    
    Args:
        agent_classes (List[Any]): List of agent classes to instantiate
        agent_names (List[str]): List of names for each agent
    """
        # Create agents with different seeds for this simulation
    sim_agents = []
    for agent_cls in agent_classes:
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
        elif agent_cls == EnsembleAgent:
            agent = EnsembleAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                bcf_kwargs = dict(nskip=100,
                ndpost=10,
                nadd=2,
                nbatch=1,
                random_state=1000 + sim),
                linear_ts_kwargs = dict(v=1)
            )
        elif agent_cls == BCFAgentPSOff:
            agent = BCFAgentPSOff(
                n_arms=scenario.K,
                n_features=scenario.P,
                nskip=100,
                ndpost=10,
                nadd=3,
                nbatch=1,
                random_state=1000 + sim
            )
        elif agent_cls == LinearTSAgent:
            agent = LinearTSAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                v = 1
            )
        # Add handlers for the ROME agents
        elif agent_cls == RoMEAgent:
            agent = RoMEAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                featurize=_featurize,
                t_max=n_draws,
                pool_users=False
            )
        elif agent_cls == StandardTSAgent:
            agent = StandardTSAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                featurize=_featurize
            )
        elif agent_cls == ActionCenteredTSAgent:
            agent = ActionCenteredTSAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                featurize=_featurize
            )
        elif agent_cls == IntelligentPoolingAgent:
            agent = IntelligentPoolingAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                featurize=_featurize,
                t_max=n_draws
            )
        else:
            # For other agents, just initialize with standard params
            agent = agent_cls(
                n_arms=scenario.K,
                n_features=scenario.P
            )
        sim_agents.append(agent)
    
    return sim_agents

def _run_scenario_simulation(scenario_name, scenario, sim, agent_classes, agent_names, n_draws):
    """
    Run a single simulation for a specific scenario.
    
    Args:
        scenario_name (str): Name of the scenario
        scenario (Scenario): The scenario instance
        sim (int): Simulation number
        agent_classes (List): List of agent classes to instantiate
        agent_names (List[str]): Names for each agent
        n_draws (int): Number of draws per simulation
        
    Returns:
        Tuple: (scenario_name, sim_index, regrets, computation_times)
    """
    # Create agents with different seeds for this simulation
    sim_agents = class_to_agents(sim=sim, scenario=scenario, agent_classes=agent_classes, n_draws=n_draws)
    
    # Run simulation
    scenario.set_seed(sim)
    scenario.init_params()
    cum_regrets, time_agents, mem_agents = simulate(scenario, sim_agents, n_draws=n_draws)
    
    # Return results for this simulation, including scenario name
    return scenario_name, sim, cum_regrets, time_agents, mem_agents

def compare_agents_across_scenarios(scenarios: Dict[str, Scenario], n_simulations: int = 10, n_draws: int = 500, 
    agent_classes = [SillyAgent, LinearTSAgent, BCFAgent], 
    agent_names = ["Random", "LinearTS", "BCF"],
    n_jobs=6, batch_size=None):
    """
    Compare multiple agents across different scenarios with memory-efficient batching.
    
    Args:
        scenarios (Dict[str, Scenario]): Dictionary of scenario name to scenario instance
        n_simulations (int): Number of simulations per scenario
        n_draws (int): Number of draws per simulation
        agent_classes (List): List of agent classes to instantiate
        agent_names (List[str]): Names for each agent
        n_jobs (int): Number of parallel jobs to run
        batch_size (int, optional): Number of scenarios to process in each batch.
            If None, all scenarios are processed in a single batch.
        
    Returns:
        Dict: Results for each scenario and agent
    """
    # Initialize results dictionary
    results = {}
    for scenario_name in scenarios.keys():
        results[scenario_name] = {
            'regrets': {name: np.zeros((n_simulations, n_draws)) for name in agent_names},
            'times': {name: np.zeros(n_simulations) for name in agent_names},
            'memory usage': {name: np.zeros(n_simulations) for name in agent_names}
        }
    
    # Process scenarios by batches
    scenario_items = list(scenarios.items())
    
    # If batch_size is None or larger than number of scenarios, process all at once
    if batch_size is None or batch_size >= len(scenario_items):
        batch_size = len(scenario_items)
    
    batch_ranges = [(i, min(i + batch_size, len(scenario_items))) 
                        for i in range(0, len(scenario_items), batch_size)]
    
    # Process each batch
    for batch_idx, (batch_start, batch_end) in enumerate(batch_ranges):
        batch_scenarios = dict(scenario_items[batch_start:batch_end])
        
        # Create jobs for this batch
        batch_jobs = []
        for scenario_name, scenario in batch_scenarios.items():
            for sim in range(n_simulations):
                batch_jobs.append((scenario_name, scenario, sim, agent_classes, agent_names, n_draws))
        
        # Print appropriate message based on batching
        print(f"Processing batch {batch_idx + 1}/{len(batch_ranges)}: scenarios {batch_start+1}-{batch_end} " +
                 f"({len(batch_jobs)} jobs)")
        
        # Run simulation jobs
        if n_jobs != 1:
            # Parallel execution
            parallel_kwargs = {'n_jobs': n_jobs}
            # Only add max_nbytes for batched execution (to save memory)
            if len(batch_ranges) > 1:
                parallel_kwargs['max_nbytes'] = '1K'
                
            batch_results = Parallel(**parallel_kwargs)(
                delayed(_run_scenario_simulation)(*job) for job in batch_jobs
            )
        else:
            # Sequential execution with progress bar
            batch_results = []
            for job in tqdm(batch_jobs, desc="Running simulations"):
                result = _run_scenario_simulation(*job)
                batch_results.append(result)
        
        # Process results
        for scenario_name, sim, cum_regrets, time_agents, mem_agents in batch_results:
            for i, name in enumerate(agent_names):
                results[scenario_name]['regrets'][name][sim, :] = cum_regrets[:, i]
                results[scenario_name]['times'][name][sim] = time_agents[i]
                results[scenario_name]['memory usage'][name][sim] = mem_agents[i]
    
    return results

def plot_comparison_results(results: Dict[str, Dict], n_draws: int, save_path: str = None, show_time = True):
    """
    Plot comparison results across all scenarios and agents.
    
    Args:
        results (Dict): Results from compare_agents_across_scenarios
        n_draws (int): Number of draws per simulation
        save_path (str, optional): Path to save the figure
    """
    n_scenarios = len(results)
    n_rows = (n_scenarios + 1) // 2  # Ceiling division for number of rows
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 6 * n_rows))
    
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
        
        quantile_95 = np.quantile([np.mean(agent_regrets, axis=0) for agent_regrets in regrets.values()], 0.95)

        # For each agent
        for agent_name, agent_regrets in regrets.items():
            # log_regrets = np.log(agent_regrets + 1e-10)
            y_values = agent_regrets
            mean_regret = np.mean(y_values, axis=0)
            lower_ci = np.percentile(y_values, 25, axis=0)
            upper_ci = np.percentile(y_values, 75, axis=0)

            # Plot mean regret
            x_draws = np.arange(n_draws) # np.log(np.arange(n_draws))
            ax.plot(x_draws, mean_regret, label=f"{agent_name}", 
                   color=agent_colors[agent_name], linewidth=2)
            
            # Plot credible interval
            # ax.fill_between(range(n_draws), lower_ci, upper_ci, 
            #               color=agent_colors[agent_name], alpha=0.2)
            ax.plot(x_draws, lower_ci, '--', color=agent_colors[agent_name], alpha=0.7, linewidth=1)
            ax.plot(x_draws, upper_ci, '--', color=agent_colors[agent_name], alpha=0.7, linewidth=1)

        ax.set_ylim(0, top = quantile_95)

        if show_time:
            # Add computation time annotations
            for agent_name, agent_times in times.items():
                mean_time = np.mean(agent_times)
                ax.annotate(f"{agent_name}: {mean_time:.2f}s", 
                           xy=(0.5, 0.02 + list(times.keys()).index(agent_name) * 0.05),
                           xycoords='axes fraction',
                           ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
        
        ax.set_title(f"{scenario_name} Scenario", fontsize=12)
        ax.set_xlabel("Draw", fontsize=10)
        ax.set_ylabel("Cumulative Regret", fontsize=10)
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
        mems = scenario_results['memory usage']
        
        # Print final regrets
        print("\nFinal cumulative regrets (mean ± std):")
        for agent_name, agent_regrets in regrets.items():
            final_regrets = agent_regrets[:, -1]  # Get the last column (final regrets)
            print(f"  {agent_name}: {np.mean(final_regrets):.2f} (±{np.std(final_regrets):.2f})")
        
        # Print computation times
        print("\nAverage computation times (seconds):")
        for agent_name, agent_times in times.items():
            print(f"  {agent_name}: {np.mean(agent_times):.4f} (±{np.std(agent_times):.4f})")

        # Print memory usage if exists (i.e. any agent has non-"zero" memory usage)
        has_memory_data = any(np.any(agent_memory > 0) for agent_memory in mems.values())
        if has_memory_data:
            print("\nAverage memory usage (bytes):")
            for agent_name, agent_memory in scenario_results['memory usage'].items():
                print(f"  {agent_name}: {np.mean(agent_memory):.2f} (±{np.std(agent_memory):.2f})")
        
        print("\n" + "=" * 40)
