from bart_playground.bandit.sim_util import *
from bart_playground.bandit.bcf_agent import BCFAgent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import datetime

def format_computation_times(all_results):
    """Format computation times for all experiment results
    
    Args:
        all_results: Dictionary with results for each parameter combination
        
    Returns:
        list: List of formatted strings with computation times
    """
    formatted_lines = []
    for params_key, result in all_results.items():
        param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        times = result['times']
        formatted_lines.append(f"{param_str}: {np.mean(times):.4f} (±{np.std(times):.4f})")
    return formatted_lines

def format_final_regrets(all_results):
    """Format final regrets for all experiment results
    
    Args:
        all_results: Dictionary with results for each parameter combination
        
    Returns:
        list: List of formatted strings with final regrets
    """
    formatted_lines = []
    for params_key, result in all_results.items():
        param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        final_regrets = result['regrets'][:, -1]  # Get the last column (final regrets)
        formatted_lines.append(f"{param_str}: {np.mean(final_regrets):.2f} (±{np.std(final_regrets):.2f})")
    return formatted_lines

def log_simulation_results(all_results, n_draws, n_repeats, param_list, base_dir='simulation_logs'):
    """
    Log simulation parameters and results to a timestamped file and save plot.
    
    Args:
        all_results (dict): Dictionary of simulation results
        n_draws (int): Number of simulation rounds
        n_repeats (int): Number of experiment repetitions
        param_list (list): List of parameter dictionaries used
        base_dir (str): Directory to save logs and figures
        
    Returns:
        str: Path to the log directory
    """
    # Create timestamp for unique directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure base log directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create log subdirectory with timestamp
    log_dir = os.path.join(base_dir, f"parameter_test_{timestamp}")
    os.makedirs(log_dir)
    
    # Define filenames
    log_filename = os.path.join(log_dir, f"simulation_log_{timestamp}.txt")
    plot_filename = os.path.join(log_dir, f"regret_plot_{timestamp}.png")
    
    # Get the varied parameters (parameters that have different values across experiments)
    varied_params = set()
    for param_dict in param_list:
        varied_params.update(param_dict.keys())
    
    # Write parameters to log file
    with open(log_filename, 'w') as log_file:
        log_file.write(f"BCFAgent Parameter Test - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=" * 50 + "\n\n")
        
        # Write general experiment parameters
        log_file.write("EXPERIMENT PARAMETERS:\n")
        log_file.write("-" * 30 + "\n")
        log_file.write(f"Number of draws: {n_draws}\n")
        log_file.write(f"Number of repetitions: {n_repeats}\n")
        log_file.write(f"Parameter combinations tested: {param_list}\n\n")
        
        # Write detailed results
        log_file.write("DETAILED RESULTS:\n")
        log_file.write("-" * 30 + "\n")
        
        # Computation times
        log_file.write("Average computation times (seconds):\n")
        for line in format_computation_times(all_results):
            log_file.write(f"{line}\n")
        
        # Final regrets
        log_file.write("\nFinal cumulative regrets (mean ± std):\n")
        for line in format_final_regrets(all_results):
            log_file.write(f"{line}\n")
            
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write(f"Log created at: {datetime.datetime.now()}\n")
    
    # Save current plot
    plt.figure(figsize=(12, 7))
    
    # Plot each experiment
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (params_key, result) in enumerate(all_results.items()):
        regrets = result['regrets']
        param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        
        # Calculate mean and 50% confidence interval
        mean_regret = np.mean(regrets, axis=0)
        lower_ci = np.percentile(regrets, 25, axis=0)
        upper_ci = np.percentile(regrets, 75, axis=0)
        
        # Plot mean regret
        plt.plot(range(n_draws), mean_regret, label=param_str, color=colors[idx % len(colors)], linewidth=2)
        
        # Plot confidence interval
        plt.fill_between(range(n_draws), lower_ci, upper_ci, color=colors[idx % len(colors)], alpha=0.2)
    
    plt.xlabel("Draw", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    
    # Create a title based on varied parameters
    varied_params_str = " & ".join(varied_params)
    plt.title(f"Effect of {varied_params_str} Parameters on BCFAgent Performance", fontsize=16)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    
    print(f"Simulation log saved to: {log_filename}")
    print(f"Plot saved to: {plot_filename}")
        
    return log_dir

def run_single_experiment(params, scenario, n_draws=500, agent_seed=None):
    """Run a single experiment with specific parameters
    
    Args:
        params: Dictionary of parameters to use (must include nadd and nbatch)
        scenario: The scenario to use
        n_draws: Number of simulation rounds
        agent_seed: Random seed for the agent
        
    Returns:
        cum_regrets: Cumulative regret over time
        time_agent: Total computation time
    """
    # Create agent with the specified parameters
    agent = BCFAgent(
        n_arms=scenario.K, 
        n_features=scenario.P, 
        nskip=100, 
        ndpost=10, 
        nadd=params.get('nadd', 1),
        nbatch=params.get('nbatch', 1),
        random_state=agent_seed
    )
    
    # Run simulation
    cum_regrets, time_agent = simulate(scenario, [agent], n_draws=n_draws)
    
    return cum_regrets[:, 0], time_agent[0]  # Return only the first agent's results

def run_multiple_experiments(param_list, n_repeats=8, n_draws=500):
    """Run multiple experiments for each parameter combination
    
    Args:
        param_list: List of parameter dictionaries to test
        n_repeats: Number of repetitions for each parameter combination
        n_draws: Number of simulation rounds
        
    Returns:
        all_results: Dictionary with results for each parameter combination
    """
    # Scenario parameters
    n_arms = 3
    n_features = 4
    sigma2 = 1.0
    
    # Create a single scenario with fixed seed to ensure consistent parameters
    np.random.seed(42)  # Fixed seed for reproducibility
    scenario = LinearOffsetScenario(P=n_features, K=n_arms, sigma2=sigma2)
    
    # Store the original mu_a and arm_offsets for reference
    original_mu_a = scenario.mu.copy()
    original_arm_offsets = scenario.arm_offsets.copy()
    
    # Reset random seed
    np.random.seed(None)
    
    # Store results for each parameter combination
    all_results = {}
    
    # Run experiments for each parameter combination
    for exp_idx, params in enumerate(param_list):
        # Create a hashable key from parameters
        params_key = tuple(sorted(params.items()))
        
        # Print parameter combination
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        print(f"Running experiment {exp_idx}: {params_str}")
        
        results = []
        times = []
        
        for i in tqdm(range(n_repeats), desc=f"exp={exp_idx}", position=0, leave=True):
            # Use a different seed for each agent
            agent_seed = 1000 + i
            
            # Run experiment with the fixed scenario
            cum_regrets, time_agent = run_single_experiment(
                params=params,
                scenario=scenario, 
                n_draws=n_draws, 
                agent_seed=agent_seed
            )
            
            results.append(cum_regrets)
            times.append(time_agent)
        
        # Store results with the parameters key
        all_results[params_key] = {
            'params': params,            # Store the parameters explicitly
            'regrets': np.array(results), # Shape: (n_repeats, n_draws)
            'times': np.array(times)      # Shape: (n_repeats,)
        }
    
    return all_results

def plot_results_with_confidence_intervals(all_results, n_draws):
    """Plot results with 50% confidence intervals"""
    plt.figure(figsize=(12, 7))
    
    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Get the varied parameters (parameters that have different values)
    param_keys = set()
    for params_key, result in all_results.items():
        param_keys.update(result['params'].keys())
    
    # Plot each experiment
    for idx, (params_key, result) in enumerate(all_results.items()):
        regrets = result['regrets']
        param_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        
        # Calculate mean and 50% confidence interval
        mean_regret = np.mean(regrets, axis=0)
        lower_ci = np.percentile(regrets, 25, axis=0)
        upper_ci = np.percentile(regrets, 75, axis=0)
        
        # Plot mean regret
        plt.plot(range(n_draws), mean_regret, label=param_str, color=colors[idx % len(colors)], linewidth=2)
        
        # Plot confidence interval
        plt.fill_between(range(n_draws), lower_ci, upper_ci, color=colors[idx % len(colors)], alpha=0.2)
    
    plt.xlabel("Draw", fontsize=14)
    plt.ylabel("Cumulative Regret", fontsize=14)
    
    # Create a title based on varied parameters
    varied_params_str = " & ".join(param_keys)
    plt.title(f"Effect of {varied_params_str} Parameters on BCFAgent Performance", fontsize=16)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_results(all_results):
    """Print a summary of the results for all experiments
    
    Args:
        all_results: Dictionary with results for each parameter combination
    """
    # Print average computation times
    print("Average computation times (seconds):")
    for line in format_computation_times(all_results):
        print(line)

    # Print final regrets
    print("\nFinal cumulative regrets (mean ± std):")
    for line in format_final_regrets(all_results):
        print(line)
