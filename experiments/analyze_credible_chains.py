#!/usr/bin/env python3
"""
Analyze credible interval chains and compute Gelman-Rubin statistics on quantiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def gelman_rubin_statistic(chains):
    """
    Calculate Gelman-Rubin statistic for MCMC convergence.
    
    Args:
        chains: array of shape (n_chains, n_samples) or (n_test_points, n_chains, n_samples)
    
    Returns:
        float or array: Gelman-Rubin R-hat statistic(s)
    """
    if chains.ndim == 2:
        # Single set of chains
        chains = chains[np.newaxis, :, :]
    
    n_test, n_chains, n_samples = chains.shape
    
    # Calculate within-chain variance
    W = np.mean(np.var(chains, axis=2, ddof=1), axis=1)
    
    # Calculate between-chain variance
    chain_means = np.mean(chains, axis=2)
    overall_mean = np.mean(chain_means, axis=1, keepdims=True)
    B = n_samples * np.var(chain_means, axis=1, ddof=1)
    
    # Pooled variance estimate
    var_plus = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
    
    # Gelman-Rubin statistic
    R_hat = np.sqrt(var_plus / W)
    
    return R_hat if n_test > 1 else R_hat[0]


def compute_quantile_gelman_rubin(credible_chains, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Compute Gelman-Rubin statistics on quantiles of predictions.
    
    For each quantile q and each chain, we compute the q-th quantile of predictions 
    across all test points and all posterior samples within that chain.
    This gives us one value per chain per quantile.
    Then we compute GR using these quantile values across posterior samples within each chain.
    
    Args:
        credible_chains: array of shape (n_test_samples, n_chains, n_posterior_samples)
        quantiles: list of quantiles to compute (default: [0.05, 0.25, 0.5, 0.75, 0.95])
    
    Returns:
        dict: Gelman-Rubin statistics for each quantile
    """
    n_test, n_chains, n_posterior = credible_chains.shape
    
    results = {}
    
    for q in quantiles:
        # For each chain and each posterior sample, compute the q-th quantile across test points
        # This gives us: (n_chains, n_posterior)
        quantile_per_chain_sample = np.quantile(credible_chains, q, axis=0)  # shape: (n_chains, n_posterior)
        
        # Now compute Gelman-Rubin on these quantile values
        # Transpose to (n_chains, n_posterior) which is what gelman_rubin expects with single metric
        gr_stat = gelman_rubin_statistic(quantile_per_chain_sample[np.newaxis, :, :])  # Add test dimension
        
        results[f'q{int(q*100)}'] = {
            'quantile': q,
            'gr_value': float(gr_stat),
        }
        
        LOGGER.debug(f"Quantile {q:.2f}: GR = {gr_stat:.4f}")
    
    return results


def analyze_experiment_run(npz_path, pkl_path=None):
    """
    Analyze a single experimental run.
    
    Args:
        npz_path: Path to credible chains NPZ file
        pkl_path: Optional path to results PKL file for metadata
    
    Returns:
        dict: Analysis results
    """
    # Load credible chains
    data = np.load(npz_path)
    credible_chains = data['credible_chains']
    data.close()
    
    n_test, n_chains, n_posterior = credible_chains.shape
    LOGGER.info(f"Loaded chains: {n_test} test points, {n_chains} chains, {n_posterior} posterior samples")
    
    # Compute quantile-based Gelman-Rubin statistics
    quantile_results = compute_quantile_gelman_rubin(credible_chains)
    
    # Also compute GR on the posterior mean across test points for each posterior sample
    # For each chain and posterior sample, compute mean across test points
    mean_per_chain_sample = np.mean(credible_chains, axis=0)  # shape: (n_chains, n_posterior)
    gr_posterior_mean = gelman_rubin_statistic(mean_per_chain_sample[np.newaxis, :, :])
    
    results = {
        'quantile_gr': quantile_results,
        'posterior_mean_gr': {
            'gr_value': float(gr_posterior_mean),
        },
        'n_test': n_test,
        'n_chains': n_chains,
        'n_posterior': n_posterior
    }
    
    # Load metadata if available
    if pkl_path and Path(pkl_path).exists():
        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
            results['metadata'] = {
                'rmse_credible': metadata.get('rmse_credible', np.nan),
                'coverage_credible': metadata.get('coverage_credible', np.nan),
            }
    
    return results


def scan_and_analyze_all_runs(base_dir, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Scan all credible prediction runs and compute GR statistics.
    
    Args:
        base_dir: Base directory containing experiment results
        quantiles: List of quantiles to analyze
    
    Returns:
        DataFrame: Results for all runs
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        LOGGER.error(f"Base directory does not exist: {base_dir}")
        return pd.DataFrame()
    
    results = []
    
    # Find all NPZ files
    npz_files = list(base_path.glob("**/credible_chains_*.npz"))
    LOGGER.info(f"Found {len(npz_files)} credible chain files to analyze")
    
    for npz_file in tqdm(npz_files, desc="Analyzing runs"):
        # Parse path: .../dgp/seed_X/ntrain_Y/credible_chains_temperature_Z.npz
        parts = npz_file.parts
        
        try:
            dgp_idx = -4
            seed_idx = -3
            ntrain_idx = -2
            
            dgp = parts[dgp_idx]
            seed = int(parts[seed_idx].replace('seed_', ''))
            ntrain = int(parts[ntrain_idx].replace('ntrain_', ''))
            
            # Extract temperature from filename
            filename = npz_file.stem  # credible_chains_temperature_1.0
            temp_str = filename.replace('credible_chains_temperature_', '')
            temperature = float(temp_str)
            
            # Find corresponding PKL file
            pkl_file = npz_file.parent / f"results_temperature_{temp_str}.pkl"
            
            # Analyze this run
            analysis = analyze_experiment_run(npz_file, pkl_file)
            
            # Extract GR statistics for each quantile
            row = {
                'dgp': dgp,
                'seed': seed,
                'ntrain': ntrain,
                'temperature': temperature,
                'n_test': analysis['n_test'],
                'n_chains': analysis['n_chains'],
                'n_posterior': analysis['n_posterior'],
            }
            
            # Add posterior mean GR
            row['gr_posterior_mean'] = analysis['posterior_mean_gr']['gr_value']
            
            # Add quantile GR statistics
            for q_key, q_data in analysis['quantile_gr'].items():
                row[f'gr_{q_key}'] = q_data['gr_value']
            
            # Add metadata if available
            if 'metadata' in analysis:
                row['rmse_credible'] = analysis['metadata']['rmse_credible']
                row['coverage_credible'] = analysis['metadata']['coverage_credible']
            
            results.append(row)
            
        except Exception as e:
            LOGGER.warning(f"Failed to process {npz_file}: {e}")
            continue
    
    df = pd.DataFrame(results)
    LOGGER.info(f"Successfully analyzed {len(df)} runs")
    
    return df


def plot_gr_by_quantile(df, output_dir='plots_gr_quantiles'):
    """
    Create plots showing GR statistics by quantile using grouped bar plots.
    
    Args:
        df: DataFrame with GR statistics
        output_dir: Directory to save plots
    """
    # Set publication-quality rcParams
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = [7.8, 5.8]
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['font.size'] = 14.0
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract quantile columns
    quantile_cols = [col for col in df.columns if col.startswith('gr_q') and not col.startswith('gr_q_')]
    quantiles = [int(col.replace('gr_q', '')) / 100 for col in quantile_cols]
    
    # Define color palette for temperatures (consistent with Fig 4)
    temp_colors = {
        1.0: '#000000',        # Black for T=1.0 (baseline)
        2.0: '#8B0000',        # Dark red for T=2.0
        3.0: '#FF6B6B',        # Light red for T=3.0
        'linear': '#FF8C00',   # Orange for linear schedule
    }
    
    # Temperature labels mapping
    temp_labels = {
        1.0: r'$T=1.0$',
        2.0: r'$T=2.0$',
        3.0: r'$T=3.0$',
        'linear': 'Linear Schedule',
    }
    
    # Filter: only use n_train = 10000
    df_filtered = df[df['ntrain'] == 10000].copy()
    
    # Plot: Grouped bar plot - one subplot per DGP
    for dgp in df_filtered['dgp'].unique():
        dgp_df = df_filtered[df_filtered['dgp'] == dgp]
        
        # Get unique temperatures and filter out cosine schedule
        temps = sorted([t for t in dgp_df['temperature'].unique() if t != 'cosine'])
        
        # If 'linear' schedule exists, include it
        if 'linear' in dgp_df['temperature'].unique():
            temps = [t for t in temps if t != 'linear']  # Remove if numeric
            temps.append('linear')
        
        if len(temps) == 0:
            LOGGER.warning(f"No valid temperatures found for {dgp}")
            continue
        
        # Create figure (using rcParams figsize)
        fig, ax = plt.subplots(1, 1)
        
        # Set up bar positions
        n_quantiles = len(quantiles)
        n_temps = len(temps)
        bar_width = 0.8 / n_temps
        x_positions = np.arange(n_quantiles)
        
        # Plot bars for each temperature
        for i, temp in enumerate(temps):
            subset_df = dgp_df[dgp_df['temperature'] == temp]
            
            if len(subset_df) > 0:
                # Get GR values for each quantile (mean across seeds)
                gr_means = [subset_df[col].mean() for col in quantile_cols]
                
                # Get color for this temperature
                color = temp_colors.get(temp, '#888888')
                label = temp_labels.get(temp, f'T={temp}')
                
                # Calculate bar positions for this temperature
                bar_positions = x_positions + (i - n_temps/2 + 0.5) * bar_width
                
                # Plot bars
                ax.bar(bar_positions, gr_means, bar_width, 
                      color=color, label=label, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add reference line at R̂ = 1.1 (without adding to legend)
        ax.axhline(y=1.1, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # Customize plot
        ax.set_xlabel('Quantile', fontweight='bold')
        ax.set_ylabel(r'Gelman-Rubin $\hat{R}$', fontweight='bold')
        ax.set_title(dgp.replace("_", " ").title(), fontweight='bold')
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'{q:.2f}' for q in quantiles])
        
        # Add grid and legend (only for Echo Months)
        ax.grid(True, alpha=0.3, axis='y')
        if dgp == '1199_BNG_echoMonths':
            ax.legend(frameon=True, loc='upper right')
        ax.set_ylim(0.98, None)  # Let matplotlib auto-scale the upper limit
        
        plt.tight_layout()
        plot_path = output_path / f'gr_quantiles_{dgp}_ntrain10k.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"Saved plot to {plot_path}")
    
    # Summary plot: All datasets
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for dgp in sorted(df['dgp'].unique()):
        dgp_df = df[df['dgp'] == dgp]
        gr_means = [dgp_df[col].mean() for col in quantile_cols]
        ax.plot(quantiles, gr_means, marker='o', label=dgp.replace('_', ' '), linewidth=2, markersize=8)
    
    ax.axhline(y=1.1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='R̂ = 1.1 threshold')
    ax.set_xlabel('Quantile', fontsize=14)
    ax.set_ylabel('Mean Gelman-Rubin R̂', fontsize=14)
    ax.set_title('Gelman-Rubin Statistics by Quantile - All Datasets', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / 'gr_quantiles_all_datasets.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved summary plot to {plot_path}")


def main():
    """Main analysis pipeline."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Analyze credible chains and compute Gelman-Rubin statistics')
    parser.add_argument('--base_dir', type=str, 
                       default='/accounts/projects/sekhon/theo_s/scratch/credible_predictions_runs',
                       help='Base directory containing experiment results')
    parser.add_argument('--output_csv', type=str, default='gr_quantile_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--output_plots', type=str, default='plots_gr_quantiles',
                       help='Output directory for plots')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.05, 0.25, 0.5, 0.75, 0.95],
                       help='Quantiles to analyze')
    parser.add_argument('--force_reanalyze', action='store_true',
                       help='Force re-analysis even if cached CSV exists')
    parser.add_argument('--use_cache', type=str, default=None,
                       help='Use specific cached CSV file instead of analyzing')
    
    args = parser.parse_args()
    
    # Check if we should use cached results
    if args.use_cache:
        cache_file = args.use_cache
    elif not args.force_reanalyze and os.path.exists(args.output_csv):
        cache_file = args.output_csv
    else:
        cache_file = None
    
    if cache_file and os.path.exists(cache_file):
        LOGGER.info(f"Using cached results from {cache_file}")
        df = pd.read_csv(cache_file)
        LOGGER.info(f"Loaded {len(df)} cached results")
    else:
        # Analyze all runs
        LOGGER.info(f"Scanning {args.base_dir} for credible chain files...")
        df = scan_and_analyze_all_runs(args.base_dir, quantiles=args.quantiles)
        
        if df.empty:
            LOGGER.error("No runs found to analyze")
            return
        
        # Save results
        df.to_csv(args.output_csv, index=False)
        LOGGER.info(f"Saved results to {args.output_csv}")
    
    # Create plots
    LOGGER.info("Creating plots...")
    plot_gr_by_quantile(df, output_dir=args.output_plots)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("GELMAN-RUBIN QUANTILE ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nTotal runs analyzed: {len(df)}")
    print(f"Datasets: {', '.join(sorted(df['dgp'].unique()))}")
    print(f"Sample sizes: {sorted(df['ntrain'].unique())}")
    print(f"Temperatures: {sorted(df['temperature'].unique())}")
    
    print("\n" + "-"*80)
    print("Mean Gelman-Rubin Statistics Across All Runs:")
    print("-"*80)
    
    quantile_cols = [col for col in df.columns if col.startswith('gr_q') and not col.startswith('gr_q_')]
    for col in quantile_cols:
        q_val = int(col.replace('gr_q', ''))
        mean_gr = df[col].mean()
        max_gr = df[col].max()
        min_gr = df[col].min()
        print(f"  Quantile {q_val:>2}%: Mean R̂ = {mean_gr:.4f}, Min R̂ = {min_gr:.4f}, Max R̂ = {max_gr:.4f}")
    
    print(f"\n  Posterior Mean: Mean R̂ = {df['gr_posterior_mean'].mean():.4f}, Min R̂ = {df['gr_posterior_mean'].min():.4f}, Max R̂ = {df['gr_posterior_mean'].max():.4f}")
    print("="*80)


if __name__ == '__main__':
    main()

