from bart_playground.bandit.bart_agent import BARTAgent#, LogisticBARTAgent
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

def class_to_agents(sim, scenario, agent_classes: List[Any]) -> Tuple[List[BanditAgent], List[str]]:
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
            agent = BCFAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                nskip=100,
                ndpost=100,
                nadd=3,
                nbatch=1,
                random_state=1000 + sim
            )
        elif agent_cls == BARTAgent:
            agent = BARTAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                nskip=100,
                ndpost=200,
                nadd=5,
                nbatch=1,
                random_state=1000 + sim,
                encoding='multi'
            )
        # elif agent_cls == LogisticBARTAgent:
        #     agent = LogisticBARTAgent(
        #         n_arms=scenario.K,
        #         n_features=scenario.P,
        #         nskip=100,
        #         ndpost=200,
        #         nadd=20,
        #         nbatch=1,
        #         random_state=1000 + sim
        #     )
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
                nadd=2,
                nbatch=1,
                random_state=1000 + sim
            )
        elif agent_cls == LinearTSAgent:
            agent = LinearTSAgent(
                n_arms=scenario.K,
                n_features=scenario.P,
                v = 1
            )
        # elif agent_cls == RoMEAgent:
        #     agent = RoMEAgent(
        #         n_arms=scenario.K,
        #         n_features=scenario.P,
        #         featurize=_featurize,
        #         t_max=n_draws,
        #         pool_users=False
        #     )
        # elif agent_cls == StandardTSAgent:
        #     agent = StandardTSAgent(
        #         n_arms=scenario.K,
        #         n_features=scenario.P,
        #         featurize=_featurize
        #     )
        # elif agent_cls == ActionCenteredTSAgent:
        #     agent = ActionCenteredTSAgent(
        #         n_arms=scenario.K,
        #         n_features=scenario.P,
        #         featurize=_featurize
        #     )
        # elif agent_cls == IntelligentPoolingAgent:
        #     agent = IntelligentPoolingAgent(
        #         n_arms=scenario.K,
        #         n_features=scenario.P,
        #         featurize=_featurize,
        #         t_max=n_draws
        #     )
        else:
            # For other agents, just initialize with standard params
            agent = agent_cls(
                n_arms=scenario.K,
                n_features=scenario.P
            )
        sim_agents.append(agent)
    
    return sim_agents

def _run_single_simulation(sim, scenario, agent_classes, agent_names, n_draws):
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
    sim_agents = class_to_agents(sim=sim, scenario=scenario, agent_classes=agent_classes)
    
    # Run simulation
    cum_regrets, time_agents, mem_agents = simulate(scenario, sim_agents, n_draws=n_draws)
    
    # Return results for this simulation
    return sim, cum_regrets, time_agents, sim_agents

def generate_simulation_data_for_agents(scenario: Scenario, agents: List[BanditAgent], agent_names: List[str], n_simulations: int = 10, n_draws: int = 500, n_jobs=6):
    """
    Generate simulation data for multiple agents on a given scenario.
    
    Args:
        scenario (Scenario): The scenario instance to use for simulation
        agents (List[BanditAgent]): List of agent classes to instantiate and test
        agent_names (List[str]): Names for each agent for display purposes
        n_simulations (int): Number of simulation runs
        n_draws (int): Number of draws per simulation
        n_jobs (int): Number of parallel jobs to run. -1 means using all processors.
        
    Returns:
        Dict: Dictionary containing simulation results
    """
    n_agents = len(agents)
    all_regrets = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    all_times = {name: np.zeros(n_simulations) for name in agent_names}
    
    if n_jobs != 1:
        # Run simulations in parallel
        print(f"Running {n_simulations} simulations in parallel...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_simulation)(
                sim, scenario, agents, agent_names, n_draws
            ) for sim in range(n_simulations) # tqdm(range(n_simulations), desc="Simulating")
        )
    else:
        # Run simulations sequentially
        results = []
        if n_simulations > 1:
            for sim in tqdm(range(n_simulations), desc="Simulating sequentially"):
                result = _run_single_simulation(sim, scenario, agents, agent_names, n_draws)
                results.append(result)
        else:
            print("Running a single simulation...")
            result = _run_single_simulation(0, scenario, agents, agent_names, n_draws)
            results.append(result)
    
    all_agents = []
    # Process results from parallel jobs
    for sim, cum_regrets, time_agents, sim_agents in results:
        # Store results
        for i, name in enumerate(agent_names):
            all_regrets[name][sim, :] = cum_regrets[:, i]
            all_times[name][sim] = time_agents[i]
        all_agents.append(sim_agents)

    return {
        'regrets': all_regrets,
        'times': all_times,
        'temp_agent': all_agents  # Store the last agent instance for reference
    }


def compare_agents_across_scenarios(scenarios: Dict[str, Scenario], n_simulations: int = 10, n_draws: int = 500, 
    agent_classes = [SillyAgent, LinearTSAgent, BCFAgent], 
    agent_names = ["Random", "LinearTS", "BCF"],
    n_jobs=6):
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
    
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\nEvaluating {scenario_name} scenario...")
        scenario_results = generate_simulation_data_for_agents(
            scenario=scenario,
            agents=agent_classes,
            agent_names=agent_names,
            n_simulations=n_simulations,
            n_draws=n_draws,
            n_jobs=n_jobs
        )
        results[scenario_name] = scenario_results
    
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
            
            # Plot mean regret
            ax.plot(range(n_draws), mean_regret, label=f"{agent_name}", 
                   color=agent_colors[agent_name], linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(range(n_draws), lower_ci, upper_ci, 
                          color=agent_colors[agent_name], alpha=0.2)
        
        if show_time:
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
