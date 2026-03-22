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
N_DRAWS = 5000
FEEL_GOOD_LAMBDA = 0.01
N_FEATURES = 5
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



def warmup_numba(agents, n_features, n_arms):
    """
    Warm up numba JIT compilation by running a simple fit.
    This avoids profiling JIT compilation overhead.
    """
    print("Warming up numba JIT compilation...")
    
    # Create a simple scenario for warmup
    scenario = LinearScenario(
        P=n_features,
        K=n_arms,
        sigma2=1.0,
        random_generator=np.random.default_rng(RANDOM_SEED + 9999)  # Different seed
    )
    
    # Run a few updates to trigger fit (need enough to trigger refresh)
    # Use minimal parameters for speed
    for agent in agents:
        for _ in range(20):  # Enough to trigger at least one fit
            x = scenario.generate_covariates()
            u = scenario.reward_function(x)
            arm = agent.choose_arm(x)
            agent.update_state(arm, x, u["reward"][arm])
    
    print("✓ Numba warmup complete\n")


def create_agents():
    """Create agents for simulation."""
    return [
        DefaultBARTTSAgent(
            n_arms=N_ARMS,
            n_features=N_FEATURES,
            bart_kwargs={'nskip': 200, 'ndpost': 200, 'n_trees': 50},
            encoding='separate',
            feel_good_lambda=0.0
        ),
        DefaultBARTTSAgent(
            n_arms=N_ARMS,
            n_features=N_FEATURES,
            bart_kwargs={'nskip': 200, 'ndpost': 200, 'n_trees': 50},
            encoding='separate',
            feel_good_lambda=FEEL_GOOD_LAMBDA
        ),
    ]


def run_simulation_with_feel_good(agents=None):
    """
    Run simulation comparing lambda=0 (baseline) with feel_good_lambda enabled.
    
    Args:
        agents: Optional list of agents. If None, creates new agents.
    
    Returns:
        Tuple of (cumulative_regrets, agent_times, agent_names)
    """
    print(f"\n{'='*60}")
    print(f"Running simulation comparing λ=0 vs λ={FEEL_GOOD_LAMBDA}")
    print(f"n_draws={N_DRAWS}, n_features={N_FEATURES}, n_arms={N_ARMS}")
    print(f"{'='*60}\n")
    
    # Create scenario
    scenario = LinearScenario(
        P=N_FEATURES,
        K=N_ARMS,
        sigma2=1.0,
        random_generator=np.random.default_rng(RANDOM_SEED)
    )
    
    # Create agents if not provided
    if agents is None:
        agents = create_agents()
    
    agent_names = [
        f'Baseline (λ=0)',
        f'Feel-Good (λ={FEEL_GOOD_LAMBDA})',
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
    Plot cumulative regret and computation time comparison.
    
    Args:
        cum_regrets: Cumulative regrets array (n_draws, n_agents)
        time_agent: Agent times array (n_draws, n_agents)
        agent_names: List of agent names
    """
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left: Cumulative regret
    ax = axes[0]
    colors = ['#2E86AB', '#A23B72']  # Blue for baseline, Purple for feel-good
    for i in range(cum_regrets.shape[1]):
        ax.plot(cum_regrets[:, i], label=agent_names[i], linewidth=2.5, color=colors[i])
    
    ax.set_xlabel("Draw", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title(f"Cumulative Regret: λ=0 vs λ={FEEL_GOOD_LAMBDA}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Middle: Average computation time
    ax = axes[1]
    avg_times = np.mean(time_agent, axis=0)
    x = np.arange(len(agent_names))
    bars = ax.bar(x, avg_times, width=0.6, color=colors)
    
    ax.set_xlabel("Agent", fontsize=12)
    ax.set_ylabel("Average Time per Draw (seconds)", fontsize=12)
    ax.set_title("Computation Time", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['λ=0', f'λ={FEEL_GOOD_LAMBDA}'], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}s',
                ha='center', va='bottom', fontsize=10)
    
    # Right: Performance metrics comparison
    ax = axes[2]
    final_regrets = cum_regrets[-1, :]
    time_overhead = (avg_times[1] - avg_times[0]) / avg_times[0] * 100  # Percentage
    regret_change = (final_regrets[0] - final_regrets[1]) / final_regrets[0] * 100  # Percentage
    
    # Use more descriptive label based on sign
    if regret_change >= 0:
        regret_label = 'Regret Reduction\n(%)'
    else:
        regret_label = 'Regret Increase\n(%)'
    
    metrics = ['Time Overhead\n(%)', regret_label]
    values = [time_overhead, regret_change]
    bar_colors = ['#E63946' if v > 0 else '#06A77D' for v in values]
    
    bars = ax.barh(metrics, values, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel("Percentage Change", fontsize=12)
    ax.set_title("Feel-Good Impact", fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        label_x = width + (5 if width > 0 else -5)
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2.,
                f'{val:+.1f}%',
                ha=ha, va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feel_good_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults plot saved to: feel_good_results.png")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"\nComparison: λ=0 (Baseline) vs λ={FEEL_GOOD_LAMBDA} (Feel-Good)")
    print("-" * 60)
    
    for i, name in enumerate(agent_names):
        avg_time = np.mean(time_agent[:, i])
        final_regret = cum_regrets[-1, i]
        print(f"{name:30s}: {avg_time:.6f}s/draw, regret={final_regret:.2f}")
    
    print("\n" + "-" * 60)
    print(f"Time Overhead:      {time_overhead:+.2f}%")
    if regret_change >= 0:
        print(f"Regret Reduction:   {regret_change:+.2f}%")
    else:
        print(f"Regret Increase:    {regret_change:+.2f}%")
    
    if regret_change > 0 and time_overhead > 0:
        efficiency = regret_change / time_overhead
        print(f"Efficiency Ratio:   {efficiency:.3f} (regret reduction per % time cost)")
        if efficiency > 1.0:
            print("✓ Feel-good is EFFICIENT (good regret reduction for time cost)")
        else:
            print("⚠ Feel-good has LOW EFFICIENCY (high time cost for regret reduction)")
    elif regret_change < 0 and time_overhead > 0:
        efficiency = regret_change / time_overhead
        print(f"Efficiency Ratio:   {efficiency:.3f} (NEGATIVE - regret increase per % time cost)")
        print("✗ Feel-good is COUNTERPRODUCTIVE (increases regret with time cost)")
    elif regret_change > 0 and time_overhead <= 0:
        print("✓ Feel-good improves regret with no time overhead")
    else:
        print("⚠ Feel-good increases regret but no time cost")




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
    
    # Warm up numba JIT before profiling (use separate agents for warmup)
    warmup_agents = create_agents()
    warmup_numba(warmup_agents, N_FEATURES, N_ARMS)
    
    # Run profiling (create fresh agents for actual test)
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

