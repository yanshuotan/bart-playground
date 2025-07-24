import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_results_for_dgp(dgp_dir: str, n_train: int, result_prefix: str) -> Dict:
    """Load all results for a specific DGP, n_train, and result type across seeds."""
    results = []
    # Find all seed directories
    seed_dirs = glob.glob(os.path.join(dgp_dir, "seed_*"))
    LOGGER.info(f"Found {len(seed_dirs)} seed directories in {dgp_dir}")
    
    for seed_dir in seed_dirs:
        n_train_dir = os.path.join(seed_dir, f"ntrain_{n_train}")
        result_file = os.path.join(n_train_dir, f"results_{result_prefix}.pkl")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'rb') as f:
                    results.append(pickle.load(f))
            except (ModuleNotFoundError, ImportError) as e:
                LOGGER.warning(f"Error loading {result_file}: {str(e)}. Skipping this file.")
                continue
            except Exception as e:
                LOGGER.warning(f"Unexpected error loading {result_file}: {str(e)}. Skipping this file.")
                continue
        else:
            LOGGER.debug(f"Result file not found: {result_file}")
    
    return results

def aggregate_metrics(results: List[Dict], metric: str) -> Tuple[float, float]:
    """Calculate mean and standard error for a metric across seeds."""
    values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
    if not values:
        return np.nan, np.nan
    mean = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, std_err

def plot_aggregated_results(artifacts_dir: str, output_dir: str):
    """Create plots of aggregated results across seeds."""
    artifacts_dir = os.path.abspath(artifacts_dir)
    output_dir = os.path.abspath(output_dir)
    
    LOGGER.info(f"Reading results from: {artifacts_dir}")
    LOGGER.info(f"Saving plots to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all DGP directories
    dgp_dirs = [d for d in glob.glob(os.path.join(artifacts_dir, "*")) if os.path.isdir(d)]
    if not dgp_dirs:
        LOGGER.warning(f"No DGP directories found in {artifacts_dir}")
        return
        
    LOGGER.info(f"Found {len(dgp_dirs)} DGP directories: {[os.path.basename(d) for d in dgp_dirs]}")
    
    # Metrics to plot
    metrics = {
        'final_mse': 'RMSE',
        'coverage': 'Coverage',
        'gr_fx': 'Gelman-Rubin',
        'ess_fx': 'ESS (f(x))'
    }
    
    # For each DGP
    for dgp_dir in dgp_dirs:
        dgp_name = os.path.basename(dgp_dir)
        LOGGER.info(f"Processing DGP: {dgp_name}")
        
        # Find all n_train values
        n_train_values = []
        seed_dirs = glob.glob(os.path.join(dgp_dir, "seed_*"))
        for seed_dir in seed_dirs:
            n_train_dirs = glob.glob(os.path.join(seed_dir, "ntrain_*"))
            # Extract n_train values from directory names like "ntrain_100", "ntrain_200", etc.
            for d in n_train_dirs:
                try:
                    n_train = int(os.path.basename(d).replace("ntrain_", ""))
                    n_train_values.append(n_train)
                except ValueError:
                    LOGGER.warning(f"Could not parse n_train value from directory: {d}")
        
        n_train_values = sorted(list(set(n_train_values)))
        if not n_train_values:
            LOGGER.warning(f"No valid n_train values found in {dgp_dir}")
            continue
            
        LOGGER.info(f"Found n_train values: {n_train_values}")
        
        # Dynamically find result types
        result_types = []
        if seed_dirs:
            # Check the first seed/n_train directory for result files
            # Look into the first n_train directory of the first seed directory
            first_n_train_dir_path = sorted(glob.glob(os.path.join(seed_dirs[0], "ntrain_*")))
            if first_n_train_dir_path:
                result_files = glob.glob(os.path.join(first_n_train_dir_path[0], "results_*.pkl"))
                result_types = sorted([Path(f).stem.replace("results_", "") for f in result_files])
        
        if not result_types:
            LOGGER.warning(f"No result files found for DGP {dgp_name} to determine result types. Skipping.")
            continue
            
        LOGGER.info(f"Found result types: {result_types}")
        
        # For each metric
        for metric, metric_label in metrics.items():
            plt.figure(figsize=(10, 6))
            
            # For each result type (temperature, schedule, etc.)
            for result_type in result_types:
                means = []
                std_errs = []
                valid_n_trains_for_plot = []
                
                for n_train in n_train_values:
                    try:
                        results = load_results_for_dgp(dgp_dir, n_train, result_type)
                        if results:
                            mean, std_err = aggregate_metrics(results, metric)
                            if not np.isnan(mean):
                                if metric == 'final_mse':
                                    std_err = std_err / (2 * np.sqrt(mean)) if mean > 0 else 0
                                    mean = np.sqrt(mean)
                                
                                means.append(mean)
                                std_errs.append(std_err)
                                valid_n_trains_for_plot.append(n_train)
                    except Exception as e:
                        LOGGER.warning(f"Error processing n_train={n_train}, result_type={result_type}: {str(e)}")
                        continue
                
                if means:
                    plt.errorbar(valid_n_trains_for_plot, means, yerr=std_errs, 
                               label=result_type.replace("_", " ").capitalize(),
                               marker='o', capsize=5)
            
            if not plt.gca().has_data():
                plt.close()
                continue

            plt.xlabel('Number of Training Samples')
            plt.ylabel(metric_label)
            plt.title(f'{metric_label} vs Training Samples for {dgp_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"{dgp_name}_{metric}_aggregated.png")
            plt.savefig(plot_path)
            LOGGER.info(f"Saved plot to: {plot_path}")
            plt.close()

def main():
    # Get the absolute path to the experiments directory
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths for each experiment type
    experiment_dirs = {
        'temperature': os.path.join(experiments_dir, "outputs", "temperature_runs"),
        'schedule': os.path.join(experiments_dir, "outputs", "schedule_runs"),
        'initialization': os.path.join(experiments_dir, "outputs", "initialization_runs")
    }
    
    # Create output directory for plots
    output_dir = os.path.join(experiments_dir, "aggregated_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    LOGGER.info("Starting plot generation...")
    LOGGER.info(f"Experiments directory: {experiments_dir}")
    for exp_type, exp_dir in experiment_dirs.items():
        LOGGER.info(f"Processing {exp_type} results from: {exp_dir}")
        if os.path.exists(exp_dir):
            plot_aggregated_results(exp_dir, os.path.join(output_dir, exp_type))
        else:
            LOGGER.warning(f"Directory not found: {exp_dir}")
    
    LOGGER.info("Plot generation complete!")

if __name__ == "__main__":
    main() 