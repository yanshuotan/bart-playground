"""
Profile BART agents with feel_good_lambda > 0 enabled.

This script tests the performance impact of feel_good_lambda parameter
and visualizes the time consumption using graphviz.

Usage:
    python profile_feel_good.py
"""

import cProfile
import pstats
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bart_playground.bandit.agents.bart_ts_agents import DefaultBARTTSAgent
from bart_playground.bandit.experiment_utils.simulation import simulate, Scenario


# =============================================================================
# FIXED PARAMETERS - Modify here to change test configuration
# =============================================================================
N_DRAWS = 100
FEEL_GOOD_LAMBDA = 1/30
N_FEATURES = 10
N_ARMS = 3
RANDOM_SEED = 0
# =============================================================================


class LinearScenario(Scenario):
    """
    Simple linear scenario for testing.
    Reward = linear function of covariates with different coefficients per arm.
    """
    def __init__(self, P=10, K=3, sigma2=1.0, random_generator=None):
        super().__init__(P, K, sigma2, random_generator)
    
    def init_params(self):
        """Initialize arm coefficients."""
        # Each arm has different coefficients
        self.arm_coeffs = self.rng.normal(0, 1, size=(self.K, self.P))
    
    def reward_function(self, x):
        """
        Compute rewards as linear function of covariates.
        
        Args:
            x: Feature vector of shape (P,)
        
        Returns:
            Dictionary with 'outcome_mean' and 'reward'
        """
        # Compute mean reward for each arm
        outcome_mean = np.dot(self.arm_coeffs, x)  # (K,)
        
        # Add noise
        noise = self.rng.normal(0, np.sqrt(self.sigma2), size=self.K)
        reward = outcome_mean + noise
        
        return {
            'outcome_mean': outcome_mean,
            'reward': reward
        }



def run_simulation_with_feel_good():
    """
    Run simulation with feel_good_lambda enabled using fixed parameters.
    
    Returns:
        Tuple of (cumulative_regrets, agent_times, agent_names)
    """
    print(f"\n{'='*60}")
    print(f"Running simulation with feel_good_lambda={FEEL_GOOD_LAMBDA}")
    print(f"n_draws={N_DRAWS}, n_features={N_FEATURES}, n_arms={N_ARMS}")
    print(f"{'='*60}\n")
    
    # Create scenario
    scenario = LinearScenario(
        P=N_FEATURES,
        K=N_ARMS,
        sigma2=1.0,
        random_generator=np.random.default_rng(RANDOM_SEED)
    )
    
    # Create agents with feel_good_lambda enabled
    # Test different configurations
    agents = [
        # Multi encoding with feel-good
        DefaultBARTTSAgent(
            n_arms=N_ARMS,
            n_features=N_FEATURES,
            bart_kwargs={'nskip': 20, 'ndpost': 20, 'n_trees': 50},
            encoding='multi',
            feel_good_lambda=FEEL_GOOD_LAMBDA
        ),
        # Separate encoding with feel-good
        DefaultBARTTSAgent(
            n_arms=N_ARMS,
            n_features=N_FEATURES,
            bart_kwargs={'nskip': 20, 'ndpost': 20, 'n_trees': 50},
            encoding='separate',
            feel_good_lambda=FEEL_GOOD_LAMBDA
        ),
    ]
    
    agent_names = [
        f'DefaultBARTTS-Multi (λ={FEEL_GOOD_LAMBDA})',
        f'DefaultBARTTS-Separate (λ={FEEL_GOOD_LAMBDA})',
    ]
    
    # Run simulation
    cum_regrets, time_agent = simulate(
        scenario,
        agents,
        n_draws=N_DRAWS,
        agent_names=agent_names
    )
    
    return cum_regrets, time_agent, agent_names


def plot_results(cum_regrets, time_agent, agent_names):
    """
    Plot cumulative regret and computation time.
    
    Args:
        cum_regrets: Cumulative regrets array (n_draws, n_agents)
        time_agent: Agent times array (n_draws, n_agents)
        agent_names: List of agent names
    """
    # Plot cumulative regret
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Cumulative regret
    ax = axes[0]
    for i in range(cum_regrets.shape[1]):
        ax.plot(cum_regrets[:, i], label=agent_names[i], linewidth=2)
    
    ax.set_xlabel("Draw", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title(f"Cumulative Regret (feel_good_lambda={FEEL_GOOD_LAMBDA})", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Computation time
    ax = axes[1]
    avg_times = np.mean(time_agent, axis=0)
    x = np.arange(len(agent_names))
    bars = ax.bar(x, avg_times, width=0.6, color=['#1f77b4', '#ff7f0e'])
    
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Average Time per Draw (seconds)", fontsize=12)
    ax.set_title("Computation Time Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Multi', 'Separate'], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feel_good_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults plot saved to: feel_good_results.png")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"\nfeel_good_lambda = {FEEL_GOOD_LAMBDA}")
    print("-" * 40)
    for i, name in enumerate(agent_names):
        avg_time = np.mean(time_agent[:, i])
        final_regret = cum_regrets[-1, i]
        print(f"{name:40s}: {avg_time:.6f}s/draw, regret={final_regret:.2f}")



def profile_with_graphviz():
    """
    Profile the simulation and generate graphviz visualization.
    """
    output_dir = Path('profile_output')
    output_dir.mkdir(exist_ok=True)
    
    prof_file = output_dir / f'feel_good_lambda_{FEEL_GOOD_LAMBDA}.prof'
    dot_file = output_dir / f'feel_good_lambda_{FEEL_GOOD_LAMBDA}.dot'
    png_file = output_dir / f'feel_good_lambda_{FEEL_GOOD_LAMBDA}.png'
    
    print(f"\n{'='*60}")
    print(f"Profiling with graphviz visualization")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Run profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    cum_regrets, time_agent, agent_names = run_simulation_with_feel_good()
    
    profiler.disable()
    
    # Save profile data
    profiler.dump_stats(str(prof_file))
    print(f"\nProfile data saved to: {prof_file}")
    
    # Print top functions by cumulative time
    print(f"\n{'='*60}")
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print(f"{'='*60}\n")
    
    stats = pstats.Stats(str(prof_file))
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    # Generate graphviz visualization using gprof2dot
    print(f"\n{'='*60}")
    print("Generating graphviz visualization...")
    print(f"{'='*60}\n")
    
    try:
        # Convert profile to dot format
        with open(dot_file, 'w') as f:
            subprocess.run(
                ['gprof2dot', '-f', 'pstats', str(prof_file)],
                stdout=f,
                check=True
            )
        print(f"DOT file generated: {dot_file}")
        
        # Convert dot to PNG
        subprocess.run(
            ['dot', '-Tpng', str(dot_file), '-o', str(png_file)],
            check=True
        )
        print(f"✓ Graphviz visualization saved to: {png_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error generating graphviz: {e}")
        print("  Make sure gprof2dot and graphviz are installed:")
        print("    pip install gprof2dot")
        print("    apt-get install graphviz  # or brew install graphviz")
    except FileNotFoundError:
        print("✗ gprof2dot or dot command not found")
        print("  Make sure gprof2dot and graphviz are installed:")
        print("    pip install gprof2dot")
        print("    apt-get install graphviz  # or brew install graphviz")
    
    # Plot results
    plot_results(cum_regrets, time_agent, agent_names)
    
    return cum_regrets, time_agent, agent_names


def main():
    """Main entry point."""
    print("="*60)
    print("Feel-Good Lambda Profiling")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  n_draws: {N_DRAWS}")
    print(f"  feel_good_lambda: {FEEL_GOOD_LAMBDA}")
    print(f"  n_features: {N_FEATURES}")
    print(f"  n_arms: {N_ARMS}")
    print(f"  random_seed: {RANDOM_SEED}")
    
    # Profile with graphviz
    profile_with_graphviz()
    
    print(f"\n{'='*60}")
    print("✓ DONE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

