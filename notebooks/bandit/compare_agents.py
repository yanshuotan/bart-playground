import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Tuple, Any, Union
from tqdm import tqdm
from datetime import datetime
import ray
import logging

from bart_playground.bandit.sim_util import simulate, Scenario

def setup_logging():
    """
    Set up logging configuration for simulation.
    """
    # Logging information
    logger = logging.getLogger("bandit_simulator")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()       # console
    c_handler.setLevel(logging.INFO)
    
    c_fmt = "%(levelname)s %(name)s — %(message)s"
    c_formatter = logging.Formatter(c_fmt)
    c_handler.setFormatter(c_formatter)

    logger.addHandler(c_handler)
    return logger

def add_logging_file(save_dir: str):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("bandit_simulator")
    f_handler = logging.FileHandler(f"{log_dir}/sim_{now}.log", mode="w", encoding="utf-8")

    f_handler.setLevel(logging.DEBUG)
    f_fmt = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    f_formatter = logging.Formatter(f_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    f_handler.setFormatter(f_formatter)
    logger.addHandler(f_handler)

_ca_logger = setup_logging()
AgentSpec = Tuple[str, Any, Dict[str, Any]]

def instantiate_agents(sim: int, scenario_K: int,
                       scenario_P: int, specs: List[AgentSpec]) -> List[Any]:
    """
    Instantiate agents from specs; offset 'random_state' when provided.

    Args:
        sim: current simulation index (to set random_state)
        scenario_K: number of arms
        scenario_P: number of features
        specs: list of (agent_name, AgentClass, default_kwargs)

    Returns:
        A list of instantiated BanditAgent objects.
    """
    agents = []
    for name, cls, base_kwargs in specs:
        kwargs = base_kwargs.copy()
        kwargs['n_arms'] = scenario_K
        kwargs['n_features'] = scenario_P

        # offset seed if provided
        if 'random_state' in base_kwargs:
            kwargs['random_state'] = sim

        agents.append(cls(**kwargs))
    return agents

@ray.remote
def _run_single_simulation_remote(sim, scenario, agent_specs, n_draws):
    """
    Remote function to run a single simulation with the given scenario and agents.
    """
    return _run_single_simulation(sim, scenario, agent_specs, n_draws)

def _run_single_simulation(sim, scenario, agent_specs, n_draws):
    """
    Run a single simulation with the given scenario and agents.
    
    Args:
        sim (int): Simulation number (used for random seed)
        scenario (Scenario): The scenario instance to use for simulation
        agent_specs (List[AgentSpec]): Specifications for each agent
        n_draws (int): Number of draws per simulation
        
    Returns:
        Tuple: (sim_index, regrets, computation_times)
    """
    # Create agents with different seeds for this simulation
    sim_agents = instantiate_agents(sim, scenario.K, scenario.P, agent_specs)

    _ca_logger.debug(f"Shuffling scenario for simulation {sim} with random state {sim}...")
    scenario.shuffle(random_state=sim)

    _ca_logger.debug(f"Scenario random generator state: {scenario.rng_state}")
    # Run simulation
    cum_regrets, time_agents = simulate(scenario, sim_agents, n_draws=n_draws)
    
    # Return results for this simulation
    return sim, cum_regrets, time_agents

def generate_simulation_data_for_agents(scenario_name: str, scenario: Scenario, agent_specs: List[AgentSpec], sim_indices: List[int], n_draws: int = 500, parallel=True, save_dir: str = "test_results") -> Dict[str, Any]:
    """
    Generate simulation data for multiple agents on a given scenario.
    
    Args:
        scenario_name (str): Name of the scenario
        scenario (Scenario): The scenario instance to use for simulation
        agent_specs (List[AgentSpec]): Specifications for each agent
        sim_indices (List[int]): List of simulation indices to run
        n_draws (int): Number of draws per simulation
        parallel (bool): Whether to run simulations in parallel using Ray
        save_dir (str): Directory to save intermediate simulation results
        
    Returns:
        Dict: Dictionary containing simulation results
    """
    agent_names = [name for name, _, _ in agent_specs]
    n_simulations = len(sim_indices)
    all_regrets = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    all_times = {name: np.zeros((n_simulations, n_draws)) for name in agent_names}
    
    _ca_logger.debug("Scenario information:")
    _ca_logger.debug(f"  Name: {scenario_name}. Internal name: {scenario.__class__.__name__}")
    if hasattr(scenario, 'dataset_name'):
        _ca_logger.debug(f"    Dataset: {scenario.dataset_name}")
    _ca_logger.debug(f"  K (arms): {scenario.K}")
    _ca_logger.debug(f"  P (features): {scenario.P}")
    _ca_logger.debug(f"  Max draws: {scenario.max_draws}")
    _ca_logger.debug("Agent names and specs:")
    for name, cls, kwargs in agent_specs:
        _ca_logger.debug(f"  {name}: {cls.__name__}, kwargs: {kwargs}")

    if parallel:
        # Run simulations in parallel
        _ca_logger.info(f"Running {n_simulations} simulations in parallel using Ray...")
        results_futures = [
            _run_single_simulation_remote.remote(sim, scenario, agent_specs, n_draws)
            for sim in sim_indices
        ]
        # Wait for all tasks to complete and retrieve the results.
        results = [ray.get(future) for future in tqdm(results_futures, desc="Simulating")]
    else:
        # Run simulations sequentially
        results = []
        _ca_logger.info(f"Sequentially running simulation.")
        for i, sim in enumerate(sim_indices):
            _ca_logger.info(f"Progress: {i}/{n_simulations}. Current sim index: {sim}. ")
            _ca_logger.debug(f"Set np.random state: {sim}")
            np.random.seed(sim)  # Set random seed for reproducibility
            result : Tuple[int, np.ndarray, np.ndarray] = _run_single_simulation(sim, scenario, agent_specs, n_draws)
                
            sim_id, cum_regrets, time_agents = result
            serializable_result = {
                "sim_id": sim_id,
                "cum_regrets": cum_regrets.tolist(),
                "time_agents": time_agents.tolist()
            }
            if save_dir is not None:
                file_path = os.path.join(save_dir, f"{scenario_name}_sim{sim}_{datetime.now().strftime('%m%d_%H%M')}.json")
                with open(file_path, "w") as f:
                    json.dump(serializable_result, f, indent=4) 
                _ca_logger.info(f"Saved results for simulation {sim} of {scenario_name} in {file_path}.")

            results.append(result)
    
    for result_idx, (sim, cum_regrets, time_agents) in enumerate(results):
        # Store results
        for i, name in enumerate(agent_names):
            all_regrets[name][result_idx, :] = cum_regrets[:, i]
            all_times[name][result_idx, :] = time_agents[:, i]

    _ca_logger.debug("Simulation data generation completed successfully.")

    return {
        'regrets': all_regrets,
        'times': all_times
    }

def compare_agents_across_scenarios(scenarios: Dict[str, Scenario],
    agent_specs: List[AgentSpec], sim_indices: Union[List[int], int, None] = None, 
    max_draws: int = 500, parallel: bool = False, save_dir: str = "test_results", log_to_file: bool = True) -> Dict[str, Dict]:
    """
    Compare multiple agents across different scenarios.
    
    Args:
        scenarios (Dict[str, Scenario]): Dictionary of scenario name to scenario instance
        sim_indices (List[int]): List of simulation indices to run (default: [0,1,2,...,7]). Note that this list will be used as random seeds.
        max_draws (int): Max number of draws per simulation
        parallel (bool): Whether to run simulations in parallel using Ray
        agent_specs (List[AgentSpec]): Specifications for each agent
        save_dir (str): Directory to save intermediate simulation results
        log_to_file (bool): Whether to log detailed information to a file
    Returns:
        Dict: Results for each scenario and agent
    """
    if sim_indices is None:
        sim_indices = 8  # Default to 8 simulations if not provided
    if isinstance(sim_indices, int):
        sim_indices = list(range(sim_indices))
    
    _ca_logger = setup_logging()
    if log_to_file:
        add_logging_file(save_dir)
        _ca_logger.info(f"Logging detailed information to {save_dir}/logs.")
    if parallel:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(ignore_reinit_error=True)
        _ca_logger.info("Ray initialized for parallel simulations.")
    
    results = {}
    
    try:
        for scenario_name, scenario in scenarios.items():
            _ca_logger.info(f"Evaluating {scenario_name} scenario...")
            n_draws = int(min(max_draws, scenario.max_draws))
            scenario_results = generate_simulation_data_for_agents(
                scenario_name=scenario_name,
                scenario=scenario,
                agent_specs=agent_specs,
                sim_indices=sim_indices,
                n_draws=n_draws,
                parallel=parallel,
                save_dir=save_dir
            )
            results[scenario_name] = scenario_results
    finally:
        # Shut down Ray when all work is finished.
        if parallel and ray.is_initialized():
            ray.shutdown()
            _ca_logger.info("Ray has been shut down.")
    
    return results

def plot_comparison_results(results: Dict[str, Dict], save_loc: str = None):
    """
    Plot comparison results across all scenarios and agents.
    
    Args:
        results (Dict): Results from compare_agents_across_scenarios
        save_loc (str): Location to save the plot (if None, will not save)
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
    
    if save_loc:
        plt.savefig(save_loc, dpi=300)
    
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
