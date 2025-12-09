import numpy as np
import matplotlib.pyplot as plt
from bart_playground.bandit.experiment_utils.sim_util import simulate, Scenario
from bart_playground.bandit.bcf_agent import BCFAgent
from tqdm import tqdm
from typing import Dict
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import warnings

def generate_simulation_data_for_scenario(scenario: Scenario, n_simulations=10, n_draws=500):
    """
    Generate simulation data for a given scenario.
    
    Args:
        scenario (Scenario): The scenario instance to use for simulation
        n_simulations (int): Number of simulation runs
        n_draws (int): Number of draws per simulation
        
    Returns:
        numpy.ndarray: Array of shape (n_simulations, n_draws) containing cumulative regret curves
    """
    # Store results
    all_regrets = np.zeros((n_simulations, n_draws))
    
    for sim in tqdm(range(n_simulations), desc=f"Simulating"):
        # Create agent with a different seed for each simulation
        agent = BCFAgent(
            n_arms=scenario.K,
            n_features=scenario.P,
            nskip=100,
            ndpost=10,
            nadd=2,
            random_state=1000 + sim
        )
        
        # Run simulation
        cum_regrets, _ = simulate(scenario, [agent], n_draws=n_draws)
        all_regrets[sim, :] = cum_regrets[:, 0]
    
    return all_regrets

def generate_simulation_data(scenarios: Dict[str, Scenario], n_simulations=10, n_draws=500):
    """
    Generate simulation data for multiple scenarios.
    
    Args:
        scenarios (Dict[str, Scenario]): Dictionary mapping scenario names to scenario instances
        n_simulations (int): Number of simulation runs
        n_draws (int): Number of draws per simulation
        
    Returns:
        Dict[str, numpy.ndarray]: Dictionary mapping scenario names to arrays of shape
                                  (n_simulations, n_draws) containing cumulative regret curves
    """
    results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\nGenerating data for {scenario_name} scenario...")
        scenario_regrets = generate_simulation_data_for_scenario(scenario, n_simulations, n_draws)
        results[scenario_name] = scenario_regrets
    
    return results

def compute_pointwise_average(curves):
    """
    Compute the pointwise average of a set of curves.
    """
    return np.mean(curves, axis=0)

def plot_scenario_results(all_scenarios_regrets, scenario_means, n_simulations, save_path=None):
    """
    Plot the results for all scenarios.
    
    Args:
        all_scenarios_regrets (Dict[str, numpy.ndarray]): Dictionary mapping scenario names to regret arrays
        scenario_means (Dict[str, numpy.ndarray]): Dictionary mapping scenario names to mean regret curves
        n_simulations (int): Number of simulation runs
        save_path (str, optional): Path to save the figure
    """
    n_scenarios = len(all_scenarios_regrets)
    fig_width = min(15, 7.5 * n_scenarios)
    
    plt.figure(figsize=(fig_width, 6))
    
    for i, (scenario_name, regrets) in enumerate(all_scenarios_regrets.items()):
        plt.subplot(1, n_scenarios, i+1)
        
        # Plot individual simulation curves
        for sim in range(n_simulations):
            plt.plot(regrets[sim], alpha=0.3, color='blue')
        
        # Plot mean curve
        plt.plot(scenario_means[scenario_name], linewidth=2, color='red', label='Mean')
        
        plt.title(f'{scenario_name} Scenario - Cumulative Regret')
        plt.xlabel('Draw')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def smooth_curve(curve, window_size=None, method='moving_average', sigma=2):
    """
    Apply smoothing to a curve.
    
    Args:
        curve (numpy.ndarray): The curve to smooth
        window_size (int, optional): Window size for moving average. If None, computes automatically.
        method (str): Smoothing method ('moving_average' or 'gaussian')
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        numpy.ndarray: Smoothed curve
    """
    n = len(curve)
    
    # Automatically determine window size if not provided
    if (window_size is None):
        window_size = max(int(n / 20), 5)  # Default to 5% of points or at least 5
    
    if method == 'moving_average':
        # Ensure window_size is odd for symmetry
        if window_size % 2 == 0:
            window_size += 1
        
        half_window = window_size // 2
        smoothed = np.zeros(n)
        
        # Handle the boundaries
        for i in range(half_window):
            smoothed[i] = np.mean(curve[:i + half_window + 1])
            smoothed[n - i - 1] = np.mean(curve[n - i - half_window - 1:])
        
        # Main smoothing using a sliding window
        for i in range(half_window, n - half_window):
            smoothed[i] = np.mean(curve[i - half_window:i + half_window + 1])
    
    elif method == 'gaussian':
        smoothed = gaussian_filter1d(curve, sigma=sigma)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    return smoothed

def fit_growth_model(x, y, model_type='power', min_points=100):
    """
    Fit a growth model to the data and estimate the asymptotic growth rate.
    
    Args:
        x (numpy.ndarray): Time points (e.g., draw numbers)
        y (numpy.ndarray): Corresponding values (e.g., cumulative regret)
        model_type (str): Type of growth model ('power', 'log', 'auto', or 'both')
        min_points (int): Minimum number of points to use (from the end of the curve)
        
    Returns:
        dict: Results containing model parameters and metrics
    """
    # Ensure we have enough points
    n = len(x)
    if n < min_points:
        min_points = n
    
    # Use the last min_points for asymptotic behavior
    x_fit = x[-min_points:]
    y_fit = y[-min_points:]
    
    results = {}
    
    # Remove zeros and negative values for log transformation
    valid_indices = np.where((x_fit > 0) & (y_fit > 0))[0]
    x_valid = x_fit[valid_indices]
    y_valid = y_fit[valid_indices]
    
    if len(x_valid) < 5:
        return {"error": "Not enough valid data points after filtering zeros/negatives"}
    
    n_valid = len(x_valid)
    
    # Power-law model: y = a * x^b (log(y) = log(a) + b*log(x))
    if model_type in ['power', 'auto', 'both']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_x = np.log(x_valid)
            log_y = np.log(y_valid)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
            
            # Back-transform parameters: a = exp(intercept), b = slope.
            a = np.exp(intercept)
            b = slope
            
            # Compute predictions on the original y scale
            y_pred_power = a * (x_valid ** b)
            rss_power = np.sum((y_valid - y_pred_power)**2)
            aic_power = n_valid * np.log(rss_power / n_valid) + 2 * 2  # 2 parameters
            
            power_results = {
                'model': 'power',
                'exponent': slope,
                'coefficient': np.exp(intercept),
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'aic': aic_power,
                'big_o': f"O(n^{slope:.3f})"
            }
            
            if model_type == 'power':
                return power_results
            results['power'] = power_results
    
    # Logarithmic model: y = a + b*log(x)
    if model_type in ['log', 'auto', 'both']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_x = np.log(x_valid)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, y_valid)
            
            # Compute RSS on the original scale for the log model
            rss_log = np.sum((y_valid - (intercept + slope * log_x))**2)
            aic_log = n_valid * np.log(rss_log / n_valid) + 2 * 2  # 2 parameters
            
            log_results = {
                'model': 'log',
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'aic': aic_log,
                'big_o': f"O(log(n))"
            }
            
            if model_type == 'log':
                return log_results
            results['log'] = log_results
    
    # For auto or both, use AIC to choose the best model (lower AIC is better)
    if model_type in ['auto', 'both']:
        if results.get('power', {}).get('aic', np.inf) < results.get('log', {}).get('aic', np.inf):
            best = 'power'
        else:
            best = 'log'
        results['best_model'] = best
        if model_type == 'auto':
            return results[best]
    
    return results

def analyze_asymptotic_complexity(all_scenarios_regrets, scenario_means, smoothing_method='gaussian', 
                                 growth_model='both', plot=True, save_path=None, min_points=100):
    """
    Perform asymptotic complexity analysis on the simulation data.
    
    Args:
        all_scenarios_regrets (Dict[str, numpy.ndarray]): Dictionary mapping scenario names to regret arrays
        scenario_means (Dict[str, numpy.ndarray]): Dictionary mapping scenario names to mean regret curves
        smoothing_method (str): Method to use for smoothing ('moving_average' or 'gaussian')
        growth_model (str): Growth model to fit ('power', 'log', 'auto', or 'both')
        plot (bool): Whether to generate plots
        save_path (str, optional): Path to save the figure
        min_points (int): Minimum number of points to use for regression analysis
    
    Returns:
        Dict: Results of the asymptotic complexity analysis
    """
    results = {}
    
    for scenario_name, mean_curve in scenario_means.items():
        print(f"\nAnalyzing asymptotic complexity for {scenario_name} scenario...")
        
        # Step 3: Smoothing the Mean Curve
        x = np.arange(1, len(mean_curve) + 1)  # Time points (1-indexed)
        smoothed_curve = smooth_curve(mean_curve, method=smoothing_method)
        
        # Step 4: Regression Analysis
        growth_results = fit_growth_model(x, smoothed_curve, model_type=growth_model, min_points=min_points)
        
        results[scenario_name] = {
            'smoothed_curve': smoothed_curve,
            'growth_results': growth_results,
            'x': x
        }
    
    return results

def plot_complexity_analysis(complexity_results, all_scenarios_regrets, scenario_means, n_simulations, save_path=None, min_points=100):
    """
    Plot smoothed curves and regression lines for asymptotic complexity analysis.
    
    Args:
        complexity_results (dict): Results from analyze_asymptotic_complexity function
        all_scenarios_regrets (dict): Dictionary mapping scenario names to regret arrays
        scenario_means (dict): Dictionary mapping scenario names to mean regret curves
        n_simulations (int): Number of simulation runs
        save_path (str, optional): Path to save the figure
        min_points (int): Minimum number of points used in regression analysis
    """
    n_scenarios = len(complexity_results)
    fig_width = 6 * n_scenarios
    
    # Create a figure with two rows - original+smoothed curves on top, regression analysis on bottom
    fig, axes = plt.subplots(2, n_scenarios, figsize=(fig_width, 12))
    
    # If only one scenario, wrap the axes in a list for consistent indexing
    if n_scenarios == 1:
        axes = axes.reshape(2, 1)
    
    for i, (scenario_name, results) in enumerate(complexity_results.items()):
        x = results['x']
        smoothed_curve = results['smoothed_curve']
        growth_results = results['growth_results']
        
        # Original data and smoothed curve (top row)
        ax_top = axes[0, i]
        
        # Plot individual simulation curves
        raw_regrets = all_scenarios_regrets[scenario_name]
        for sim in range(n_simulations):
            ax_top.plot(x, raw_regrets[sim], alpha=0.2, color='gray', linewidth=0.5)
        
        # Plot mean curve
        ax_top.plot(x, scenario_means[scenario_name], alpha=0.5, color='blue', label='Mean')
        
        # Plot smoothed curve
        ax_top.plot(x, smoothed_curve, linewidth=2.5, color='red', label='Smoothed')
        
        ax_top.set_title(f"{scenario_name} Scenario: Raw and Smoothed Curves")
        ax_top.set_xlabel("Draw")
        ax_top.set_ylabel("Cumulative Regret")
        ax_top.legend()
        ax_top.grid(alpha=0.3)
        
        # Regression analysis (bottom row)
        ax_bottom = axes[1, i]
        
        # Get best model from results
        best_model = growth_results.get('best_model', 'power')
        
        # Use only the last min_points for regression analysis plots
        n_total = len(x)
        min_points_actual = min(min_points, n_total)
        x_fit = x[-min_points_actual:]
        smoothed_fit = smoothed_curve[-min_points_actual:]
        
        # For power-law model, use log-log plot
        if 'power' in growth_results:
            power_model = growth_results['power']
            log_x = np.log(x_fit)  # Only use points considered for regression
            log_y = np.log(smoothed_fit)
            
            # Remove any inf or nan values
            valid_indices = np.isfinite(log_x) & np.isfinite(log_y)
            log_x_valid = log_x[valid_indices]
            log_y_valid = log_y[valid_indices]
            
            # Plot the log-log data
            ax_bottom.scatter(log_x_valid, log_y_valid, alpha=0.3, s=5, label='Log-Log Data (Fitted Points)', color='blue')
            
            # Plot the regression line
            exponent = power_model['exponent']
            coefficient = power_model['coefficient']
            r_squared = power_model['r_squared']
            
            regression_y = exponent * log_x_valid + np.log(coefficient)
            ax_bottom.plot(log_x_valid, regression_y, 'r-', linewidth=2, 
                         label=f'Power Model: O(n^{exponent:.3f}), R²={r_squared:.3f}')
            
            ax_bottom.set_title(f"{scenario_name} Scenario: Power-Law Analysis") # (Last {min_points_actual} points)")
            ax_bottom.set_xlabel("ln(n)")
            ax_bottom.set_ylabel("ln(Regret)")
        
        # For log model, use linear-log plot
        if 'log' in growth_results:
            log_model = growth_results['log']
            log_x = np.log(x_fit)  # Only use points considered for regression
            y_valid = smoothed_fit
            
            # Remove any inf or nan values
            valid_indices = np.isfinite(log_x) & np.isfinite(y_valid)
            log_x_valid = log_x[valid_indices]
            y_valid = y_valid[valid_indices]

            slope = log_model['slope']
            intercept = log_model['intercept']
            r_squared = log_model['r_squared']
            
            # If both models are available and log model is best, add to the plot
            if 'power' in growth_results and best_model == 'log':
                # Create a second y-axis for the log model
                ax_log = ax_bottom.twinx()
                ax_log.scatter(log_x_valid, y_valid, alpha=0.3, s=5, label='Linear-Log Data (Fitted Points)', color='purple')
                
                regression_y = slope * log_x_valid + intercept
                ax_log.plot(log_x_valid, regression_y, 'g-', linewidth=2, 
                          label=f'Log Model: O(log n), R²={r_squared:.3f}')
                
                ax_log.set_ylabel("Regret")
                ax_log.legend(loc='upper right')
            
            # If only log model available or we want separate plots for clarity
            elif 'power' not in growth_results:
                ax_bottom.scatter(log_x_valid, y_valid, alpha=0.3, s=5, label='Linear-Log Data (Fitted Points)', color='purple')
                
                regression_y = slope * log_x_valid + intercept
                ax_bottom.plot(log_x_valid, regression_y, 'g-', linewidth=2, 
                             label=f'Log Model: O(log n), R²={r_squared:.3f}')
                
                ax_bottom.set_title(f"{scenario_name} Scenario: Logarithmic Analysis") # (Last {min_points_actual} points)")
                ax_bottom.set_xlabel("ln(n)")
                ax_bottom.set_ylabel("Regret")
        
        ax_bottom.grid(alpha=0.3)
        ax_bottom.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
