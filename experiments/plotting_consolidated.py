import os
import pickle
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Publication-quality metric labels
METRIC_LABELS = {
    'rmse': 'RMSE',
    'coverage': 'Coverage',
    'gr_rmse': r'Gelman-Rubin $\hat{R}$',
    'ess_rmse': 'Effective Sample Size',
}

# Standardized color palette for common parameters
COLOR_PALETTES = {
    'temperature': {
        'temperature_1.0': '#000000',  # black (baseline)
        'temperature_2.0': '#8B0000',  # dark red
        'temperature_3.0': '#FF6B6B',  # light red
        'temperature_5.0': '#d62728',  # red
    },
    'schedule': {
        'Cosine_3.0_to_0.1': '#9467bd',          # purple (will be dropped)
        'Linear_3.0_to_1.0': '#FF8C00',          # orange for linear schedule  
        'cosine': '#9467bd',                      # fallback (will be dropped)
        'linear': '#FF8C00',                      # orange for linear schedule
        # exponential deliberately omitted - will be filtered out
    },
    'initialization': {
        'FromScratch': '#000000',     # black (baseline, solid)
        'XGBoostInit': '#000000',     # black (dashed)
    },
    'trees': {
        # Gradient from very dark to very light blue based on number of trees
        'n_trees_10': '#08306b',    # very dark blue
        'n_trees_20': '#08519c',    # dark blue
        'n_trees_50': '#3182bd',    # medium blue
        'n_trees_100': '#6baed6',   # light blue
        'n_trees_200': '#9ecae1',   # lighter blue
        'n_trees_500': '#c6dbef',   # lightest blue
    },
    'burnin': {
        # burnin_1000 from temperature=1 is baseline (black), longer burnins are green gradient
        'burnin_1000': '#000000',    # black (baseline, from temperature=1)
        'nskip_5000': '#238b45',     # dark green
        'nskip_10000': '#74c476',    # light green
    },
    'prior': {
        'Standard': '#000000',     # black (baseline, solid)
        'Dirichlet': '#000000',    # black (dashed)
    },
    'dirichlet': {
        'Standard': '#000000',     # black (baseline, solid)
        'Dirichlet': '#000000',    # black (dashed)
    },
    'proposalmoves': {
        'AllMoves': '#000000',         # black (baseline, solid)
        'GrowPruneOnly': '#000000',    # black (dashed)
    },
    'marginalized': {
        'Marginalized': '#000000',     # black (dashed)
        'Standard': '#000000',         # black (solid, baseline)
    },
    'thresholds': {
        # max_bins_ntrain is baseline (black), restricted bins are green
        'max_bins_100': '#74c476',         # light green (most restricted)
        'max_bins_200': '#238b45',         # dark green (moderately restricted)
        'max_bins_ntrain': '#000000',      # black (baseline, unrestricted)
    }
}

# Clean variation names for display
VARIATION_LABELS = {
    'temperature_1.0': 'T = 1.0',
    'temperature_2.0': 'T = 2.0',
    'temperature_3.0': 'T = 3.0',
    'temperature_5.0': 'T = 5.0',
    'Cosine_3.0_to_0.1': 'Cosine',
    'Linear_3.0_to_1.0': 'Linear',
    'cosine': 'Cosine',
    'linear': 'Linear',
    'exponential': 'Exponential',
    'constant_1.0': 'Constant (T=1.0)',
    'constant_5.0': 'Constant (T=5.0)',
    'FromScratch': 'From Scratch',
    'XGBoostInit': 'XGBoost Init',
    'n_trees_10': '10 Trees',
    'n_trees_20': '20 Trees',
    'n_trees_50': '50 Trees',
    'n_trees_100': '100 Trees',
    'n_trees_200': '200 Trees',
    'n_trees_500': '500 Trees',
    'burnin_1000': 'Burn-in: 1,000',
    'nskip_5000': 'Burn-in: 5,000',
    'nskip_10000': 'Burn-in: 10,000',
    'Standard': 'Standard Prior',
    'Uniform': 'Uniform Prior',
    'Dirichlet': 'Dirichlet Prior',
    'AllMoves': 'All Moves',
    'GrowPruneOnly': 'Grow/Prune Only',
    'Marginalized': 'Marginalized',
    'max_bins_100': 'Max Bins: 100',
    'max_bins_200': 'Max Bins: 200',
    'max_bins_ntrain': r'Max Bins: $n_{\mathrm{train}}$',
    'ntrain_thresholds': r'Max Bins: $n_{\mathrm{train}}$',
}

def load_results_from_cache(cache_path: Path, experiment_name: str) -> pd.DataFrame:
    """
    Load results for a specific experiment from the master CSV cache.
    
    Args:
        cache_path: Path to the master_results_cache.csv file
        experiment_name: Name of the experiment to filter for
    
    Returns:
        DataFrame containing results for the specified experiment
    """
    try:
        LOGGER.info(f"Loading results from cache: {cache_path}")
        df = pd.read_csv(cache_path)
        
        # Filter for this experiment
        df_exp = df[df['experiment_name'] == experiment_name].copy()
        
        if df_exp.empty:
            LOGGER.warning(f"No results found in cache for experiment: {experiment_name}")
            return pd.DataFrame()
        
        # Drop the experiment_name column as it's no longer needed
        df_exp = df_exp.drop(columns=['experiment_name'])
        
        LOGGER.info(f"Loaded {len(df_exp)} results from cache for {experiment_name}")
        return df_exp
        
    except Exception as e:
        LOGGER.error(f"Error loading cache file {cache_path}: {e}")
        return pd.DataFrame()


def load_results_to_dataframe(base_artifact_dir: Path, experiment_name: str = None, 
                              use_cache: bool = True) -> pd.DataFrame:
    """
    Load results either from cache (if available) or by scanning pickle files.
    
    Args:
        base_artifact_dir: Path to the artifact directory with pickle files
        experiment_name: Name of experiment (for cache lookup)
        use_cache: Whether to try loading from cache first
    
    Returns:
        DataFrame containing experiment results
    """
    # Try loading from cache first
    if use_cache and experiment_name:
        script_dir = Path(__file__).parent
        cache_path = script_dir / "master_results_cache.csv"
        
        if cache_path.exists():
            df = load_results_from_cache(cache_path, experiment_name)
            if not df.empty:
                return df
            else:
                LOGGER.info(f"Cache miss for {experiment_name}, falling back to pickle loading")
        else:
            LOGGER.info(f"Cache file not found at {cache_path}, using pickle files")
    
    # Fallback: scan pickle files (original implementation)
    LOGGER.info(f"Loading results from pickle files in: {base_artifact_dir}")
    results = []
    
    if not base_artifact_dir.exists():
        LOGGER.error(f"Artifact directory does not exist: {base_artifact_dir}")
        return pd.DataFrame()

    path_pattern = re.compile(
        r".*/(.*?)/seed_(\d+)/ntrain_(\d+)/(.*?)\.pkl"
    )

    for pkl_file in base_artifact_dir.glob("**/*.pkl"):
        match = path_pattern.match(str(pkl_file))
        if not match:
            LOGGER.warning(f"Could not parse path: {pkl_file}")
            continue
            
        dgp, seed, n_train, result_key = match.groups()
        seed, n_train = int(seed), int(n_train)
        
        variation = result_key.replace("results_", "")

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
                
                results.append({
                    "dgp": dgp, "seed": seed, "n_train": n_train, "variation": variation,
                    "rmse_credible": data.get("rmse_credible", np.nan),
                    "coverage_credible": data.get("coverage_credible", np.nan),
                    "gr_rmse_credible": data.get("gr_rmse_credible", np.nan),
                    "ess_rmse_credible": data.get("ess_rmse_credible", np.nan),
                    "rmse_predictive": data.get("rmse_predictive", np.nan),
                    "coverage_predictive": data.get("coverage_predictive", np.nan),
                    "gr_rmse_predictive": data.get("gr_rmse_predictive", np.nan),
                    "ess_rmse_predictive": data.get("ess_rmse_predictive", np.nan),
                    "rmse_pred_vs_true": data.get("rmse_pred_vs_true", np.nan),
                    "coverage_pred_vs_true": data.get("coverage_pred_vs_true", np.nan),
                    "gr_rmse_pred_vs_true": data.get("gr_rmse_pred_vs_true", np.nan),
                    "ess_rmse_pred_vs_true": data.get("ess_rmse_pred_vs_true", np.nan),
                })
        except Exception as e:
            LOGGER.error(f"Could not load or process {pkl_file}: {e}")

    if not results:
        LOGGER.warning("No results found to plot.")
        
    return pd.DataFrame(results)

def get_color_palette(exp_name: str, variations: list) -> dict:
    """
    Get standardized color palette for an experiment.
    """
    exp_key = exp_name.lower().replace('-', '').replace(' ', '')
    
    if exp_key in COLOR_PALETTES:
        return COLOR_PALETTES[exp_key]
    else:
        # Fallback to default seaborn palette
        colors = sns.color_palette("husl", n_colors=len(variations))
        return {var: colors[i] for i, var in enumerate(sorted(variations))}

def get_variation_label(variation: str) -> str:
    """Get clean label for variation."""
    # Special handling for nskip variations (burnin)
    if 'nskip_' in variation:
        num = variation.split('_')[-1]
        return f'Burn-in: {int(num):,}'
    return VARIATION_LABELS.get(variation, variation.replace('_', ' ').title())

def get_linestyle_for_variation(exp_name: str, variation: str) -> str:
    """
    Get line style for a variation based on experiment type.
    Returns '-' for solid or '--' for dashed.
    """
    exp_key = exp_name.lower().replace('-', '').replace(' ', '')
    
    # Initialization: FromScratch is solid (baseline), XGBoostInit is dashed
    if exp_key == 'initialization':
        return '-' if variation == 'FromScratch' else '--'
    
    # Dirichlet/Prior: Standard is solid (baseline), Dirichlet is dashed
    if exp_key == 'dirichlet':
        return '-' if variation == 'Standard' else '--'
    
    # Marginalized: Standard is solid (baseline), Marginalized is dashed
    if exp_key == 'marginalized':
        return '-' if variation == 'Standard' else '--'
    
    # ProposalMoves: AllMoves is solid (baseline), GrowPruneOnly is dashed
    if exp_key == 'proposalmoves':
        return '-' if variation == 'AllMoves' else '--'
    
    # All others use solid lines
    return '-'

def sort_variations_for_legend(exp_name: str, variations: list) -> list:
    """
    Sort variations for consistent legend ordering.
    For trees experiment, sort by number of trees.
    For burnin experiment, sort by burnin value.
    """
    exp_key = exp_name.lower().replace('-', '').replace(' ', '')
    
    # Trees: sort by number of trees (ascending)
    if exp_key == 'trees':
        def tree_count(var):
            # Extract number from 'n_trees_X'
            if 'n_trees_' in var:
                return int(var.split('_')[-1])
            return 0
        return sorted(variations, key=tree_count)
    
    # Burnin: sort by burnin value (ascending)
    if exp_key == 'burnin':
        def burnin_value(var):
            # Extract number from 'burnin_X' or 'nskip_X'
            if 'burnin_' in var:
                return int(var.split('_')[-1])
            elif 'nskip_' in var:
                return int(var.split('_')[-1])
            return 0
        return sorted(variations, key=burnin_value)
    
    # Thresholds: sort by bin count (ascending), ntrain as largest
    if exp_key == 'thresholds':
        def threshold_value(var):
            if 'ntrain' in var:
                return 99999  # Treat ntrain as largest (infinity)
            elif 'max_bins_' in var:
                try:
                    return int(var.split('_')[-1])
                except ValueError:
                    return 0
            return 0
        return sorted(variations, key=threshold_value)
    
    # Default: alphabetical sort
    return sorted(variations)

def set_publication_params():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['mathtext.rm'] = 'arial'
    plt.rcParams['mathtext.it'] = 'arial:italic'
    plt.rcParams['mathtext.bf'] = 'arial:bold'
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = [7.8, 5.8]
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['font.size'] = 16.0
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['lines.linewidth'] = 4

def create_plots(df: pd.DataFrame, output_dir: Path, exp_name: str):
    """
    Generates and saves line plots for each DGP, grouped by metric type.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib parameters for publication-quality figures
    set_publication_params()
    
    # Special handling for Burn-in: include temperature=1.0 as burnin_1000
    if exp_name == 'Burn-in':
        script_dir = Path(__file__).parent
        temp_config_path = script_dir / "consolidated_configs" / "temperature.yaml"
        if temp_config_path.exists():
            temp_cfg = OmegaConf.load(temp_config_path)
            temp_df = load_results_to_dataframe(Path(temp_cfg.artifacts_dir), experiment_name='Temperature')
            # Filter for temperature=1.0 only and rename to burnin_1000
            temp_df = temp_df[temp_df['variation'] == 'temperature_1.0'].copy()
            temp_df['variation'] = 'burnin_1000'
            # Combine with burnin data
            df = pd.concat([df, temp_df], ignore_index=True)
            LOGGER.info(f"Added temperature=1.0 data as burnin_1000: {len(temp_df)} rows")
    
    # Special handling for ProposalMoves: include temperature=1.0 as AllMoves baseline
    if exp_name == 'ProposalMoves':
        script_dir = Path(__file__).parent
        temp_config_path = script_dir / "consolidated_configs" / "temperature.yaml"
        if temp_config_path.exists():
            temp_cfg = OmegaConf.load(temp_config_path)
            temp_df = load_results_to_dataframe(Path(temp_cfg.artifacts_dir), experiment_name='Temperature')
            # Filter for temperature=1.0 only and rename to AllMoves
            temp_df = temp_df[temp_df['variation'] == 'temperature_1.0'].copy()
            temp_df['variation'] = 'AllMoves'
            # Combine with proposal moves data
            df = pd.concat([df, temp_df], ignore_index=True)
            LOGGER.info(f"Added temperature=1.0 data as AllMoves baseline: {len(temp_df)} rows")
    
    # Special handling for Thresholds: include temperature=1.0 as max_bins_ntrain baseline
    if exp_name == 'Thresholds':
        script_dir = Path(__file__).parent
        temp_config_path = script_dir / "consolidated_configs" / "temperature.yaml"
        if temp_config_path.exists():
            temp_cfg = OmegaConf.load(temp_config_path)
            temp_df = load_results_to_dataframe(Path(temp_cfg.artifacts_dir), experiment_name='Temperature')
            # Filter for temperature=1.0 only and rename to max_bins_ntrain
            temp_df = temp_df[temp_df['variation'] == 'temperature_1.0'].copy()
            temp_df['variation'] = 'max_bins_ntrain'
            # Combine with thresholds data
            df = pd.concat([df, temp_df], ignore_index=True)
            LOGGER.info(f"Added temperature=1.0 data as max_bins_ntrain baseline: {len(temp_df)} rows")
    
    # Filter out exponential schedule if present
    initial_variations = df['variation'].unique()
    df = df[df['variation'] != 'exponential'].copy()
    df = df[~df['variation'].str.contains('exp', case=False, na=False)].copy()
    
    filtered_variations = df['variation'].unique()
    if len(initial_variations) != len(filtered_variations):
        LOGGER.info(f"Filtered out exponential schedules: {set(initial_variations) - set(filtered_variations)}")
    
    # Only generate plots for Credible Intervals
    # Reorder metrics: GR on left, RMSE in middle, Coverage on right
    metric_groups = {
        "Credible_Intervals": {
            "gr_rmse": "gr_rmse_credible",
            "rmse": "rmse_credible", 
            "coverage": "coverage_credible", 
        },
    }
    
    for dgp in df['dgp'].unique():
        dgp_df = df[df['dgp'] == dgp].copy()
        
        # Add display labels
        dgp_df['variation_label'] = dgp_df['variation'].apply(get_variation_label)
        
        # Get color palette and sort variations for this experiment
        variations = dgp_df['variation'].unique()
        variations = sort_variations_for_legend(exp_name, variations)
        color_palette = get_color_palette(exp_name, variations)
        
        for group_name, metrics_map in metric_groups.items():
            # Generate each metric as a separate PDF
            dgp_title = dgp.replace('_', ' ').title()
            dgp_filename = "".join(c if c.isalnum() else "_" for c in dgp)
            
            for metric_short_name, metric_full_name in metrics_map.items():
                # Create single subplot for this metric
                fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8))
                
                # Title: dataset name (only for coverage plots)
                if 'coverage' in metric_short_name:
                    fig.suptitle(dgp_title, fontweight='bold')
                
                # Plot with standardized colors and line styles
                for variation in variations:
                    var_df = dgp_df[dgp_df['variation'] == variation]
                    color = color_palette.get(variation, None)
                    label = get_variation_label(variation)
                    linestyle = get_linestyle_for_variation(exp_name, variation)
                    
                    # Calculate mean and SE across seeds
                    grouped = var_df.groupby('n_train')[metric_full_name].agg(['mean', 'sem']).reset_index()
                    
                    ax.plot(grouped['n_train'], grouped['mean'], 
                           marker='o', color=color, label=label, linestyle=linestyle, markersize=9)
                    ax.fill_between(grouped['n_train'], 
                                   grouped['mean'] - 1.96 * grouped['sem'],
                                   grouped['mean'] + 1.96 * grouped['sem'],
                                   alpha=0.2, color=color)
                
                # Change x-axis label to n_train
                ax.set_xlabel(r"$n_{\mathrm{train}}$")
                ax.set_ylabel(METRIC_LABELS.get(metric_short_name, metric_short_name))
                ax.set_xscale("log")
                
                # Put RMSE on log scale
                if 'rmse' in metric_short_name and 'gr_rmse' not in metric_short_name:
                    ax.set_yscale("log")
                
                ax.grid(True, alpha=0.3)
                
                # Legend only for RMSE plots (not GR or coverage)
                if 'rmse' in metric_short_name and 'gr_rmse' not in metric_short_name:
                    ax.legend(title="", frameon=True, ncol=1, loc='best')
                
                # Add reference lines
                if 'coverage' in metric_short_name:
                    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
                if 'gr_rmse' in metric_short_name:
                    # Only show 1.1 line if data is reasonably close to it
                    current_ylim = ax.get_ylim()
                    if current_ylim[1] > 1.05:
                        ax.axhline(y=1.1, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

                plt.tight_layout()
                
                # Save each metric as separate PDF (include exp_name in filename)
                plot_path = output_dir / exp_name / group_name / f"{exp_name}_{dgp_filename}_{metric_short_name}.pdf"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                
                plt.savefig(plot_path, format='pdf', bbox_inches='tight')
                plt.close()
                LOGGER.info(f"Saved plot to {plot_path}")

def create_combined_temp_schedule_plots(temp_df: pd.DataFrame, schedule_df: pd.DataFrame, 
                                       output_dir: Path):
    """
    Creates combined temperature + schedule plots on the same axes.
    Each plot is saved as a single PDF file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set matplotlib parameters for publication-quality figures
    set_publication_params()
    
    # Filter exponential and cosine from schedule - CRITICAL
    LOGGER.info(f"Schedule variations before filtering: {schedule_df['variation'].unique()}")
    schedule_df = schedule_df[schedule_df['variation'] != 'exponential'].copy()
    schedule_df = schedule_df[schedule_df['variation'] != 'cosine'].copy()
    schedule_df = schedule_df[~schedule_df['variation'].str.contains('exp', case=False, na=False)].copy()
    schedule_df = schedule_df[~schedule_df['variation'].str.contains('cosine', case=False, na=False)].copy()
    schedule_df = schedule_df[schedule_df['variation'] != 'Cosine_3.0_to_0.1'].copy()
    LOGGER.info(f"Schedule variations after filtering: {schedule_df['variation'].unique()}")
    
    # Combine dataframes
    temp_df = temp_df.copy()
    schedule_df = schedule_df.copy()
    temp_df['source'] = 'Temperature'
    schedule_df['source'] = 'Schedule'
    combined_df = pd.concat([temp_df, schedule_df], ignore_index=True)
    
    # Get all unique variations for color mapping
    all_variations = list(temp_df['variation'].unique()) + list(schedule_df['variation'].unique())
    color_palette_temp = get_color_palette('temperature', temp_df['variation'].unique())
    color_palette_sched = get_color_palette('schedule', schedule_df['variation'].unique())
    color_palette = {**color_palette_temp, **color_palette_sched}
    
    # Line style mapping: solid for fixed temperature, dashed for schedules
    linestyle_map = {}
    for var in temp_df['variation'].unique():
        linestyle_map[var] = '-'  # solid for fixed temperature
    for var in schedule_df['variation'].unique():
        linestyle_map[var] = '--'  # dashed for schedules
    
    # Only generate plots for Credible Intervals
    # Reorder metrics: GR on left, RMSE in middle, Coverage on right
    metric_groups = {
        "Credible_Intervals": {
            "gr_rmse": "gr_rmse_credible",
            "rmse": "rmse_credible", 
            "coverage": "coverage_credible", 
        },
    }
    
    for dgp in combined_df['dgp'].unique():
        dgp_df = combined_df[combined_df['dgp'] == dgp].copy()
        dgp_df['variation_label'] = dgp_df['variation'].apply(get_variation_label)
        
        for group_name, metrics_map in metric_groups.items():
            # Generate each metric as a separate PDF
            dgp_title = dgp.replace('_', ' ').title()
            dgp_filename = "".join(c if c.isalnum() else "_" for c in dgp)
            
            for metric_short_name, metric_full_name in metrics_map.items():
                # Create single subplot for this metric
                fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8))
                
                # Title: dataset name (only for coverage plots)
                if 'coverage' in metric_short_name:
                    fig.suptitle(dgp_title, fontweight='bold')
                
                # Plot each variation
                for variation in sorted(dgp_df['variation'].unique()):
                    var_df = dgp_df[dgp_df['variation'] == variation]
                    color = color_palette.get(variation, None)
                    label = get_variation_label(variation)
                    linestyle = linestyle_map.get(variation, '-')
                    
                    # Calculate mean and SE
                    grouped = var_df.groupby('n_train')[metric_full_name].agg(['mean', 'sem']).reset_index()
                    
                    ax.plot(grouped['n_train'], grouped['mean'], 
                           marker='o', color=color, label=label, linestyle=linestyle, 
                           markersize=9)
                    ax.fill_between(grouped['n_train'], 
                                   grouped['mean'] - 1.96 * grouped['sem'],
                                   grouped['mean'] + 1.96 * grouped['sem'],
                                   alpha=0.2, color=color)
                
                # Change x-axis label to n_train
                ax.set_xlabel(r"$n_{\mathrm{train}}$")
                ax.set_ylabel(METRIC_LABELS.get(metric_short_name, metric_short_name))
                ax.set_xscale("log")
                
                # Put RMSE on log scale
                if 'rmse' in metric_short_name and 'gr_rmse' not in metric_short_name:
                    ax.set_yscale("log")
                
                ax.grid(True, alpha=0.3)
                
                # Legend only for RMSE plots (not GR or coverage)
                if 'rmse' in metric_short_name and 'gr_rmse' not in metric_short_name:
                    ax.legend(title="", frameon=True, ncol=1, loc='best')
                
                # Add reference lines
                if 'coverage' in metric_short_name:
                    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
                if 'gr_rmse' in metric_short_name:
                    # Only show 1.1 line if data is reasonably close to it
                    current_ylim = ax.get_ylim()
                    if current_ylim[1] > 1.05:  # Only add line if y-axis extends reasonably high
                        ax.axhline(y=1.1, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

                plt.tight_layout()
                
                # Save each metric as separate PDF (include exp_name in filename)
                plot_path = output_dir / "Temperature_Schedule" / group_name / f"Temperature_Schedule_{dgp_filename}_{metric_short_name}.pdf"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                
                plt.savefig(plot_path, format='pdf', bbox_inches='tight')
                plt.close()
                LOGGER.info(f"Saved combined plot to {plot_path}")

@hydra.main(config_path="configs", config_name="plotting_consolidated", version_base=None)
def main(cfg: DictConfig):
    exp_name = cfg.experiment_name
    output_dir = Path(cfg.output_dir)

    experiment_configs = {
        "Marginalized": "marginalized.yaml",
        "Dirichlet": "dirichlet.yaml",
        "Initialization": "initialization.yaml",
        "Temperature": "temperature.yaml",
        "Schedule": "schedule.yaml",
        "Trees": "trees.yaml",
        "Burn-in": "burnin.yaml",
        "Thresholds": "thresholds.yaml",
        "ProposalMoves": "proposal_moves.yaml"
    }
    
    exp_config_name = experiment_configs.get(exp_name)
    if not exp_config_name:
        LOGGER.error(f"Unknown experiment name: {exp_name}")
        return

    script_dir = Path(__file__).parent
    exp_config_path = script_dir / "consolidated_configs" / exp_config_name
    
    if not exp_config_path.exists():
        LOGGER.error(f"Could not find experiment config at: {exp_config_path}")
        return
        
    exp_cfg = OmegaConf.load(exp_config_path)
    base_artifact_dir = Path(exp_cfg.artifacts_dir)
    
    LOGGER.info(f"Loading results for '{exp_name}' from: {base_artifact_dir}")
    
    df = load_results_to_dataframe(base_artifact_dir, experiment_name=exp_name)
    
    if not df.empty:
        LOGGER.info(f"Found {len(df)} results. Generating plots...")
        create_plots(df, output_dir, exp_name)
        
        # If this is Temperature or Schedule, also check for combined plot
        if exp_name in ["Temperature", "Schedule"] and cfg.get("create_combined_temp_schedule", False):
            LOGGER.info("Creating combined Temperature + Schedule plots...")
            
            # Load both temperature and schedule data
            temp_config_path = script_dir / "consolidated_configs" / "temperature.yaml"
            sched_config_path = script_dir / "consolidated_configs" / "schedule.yaml"
            
            temp_cfg = OmegaConf.load(temp_config_path)
            sched_cfg = OmegaConf.load(sched_config_path)
            
            temp_df = load_results_to_dataframe(Path(temp_cfg.artifacts_dir), experiment_name='Temperature')
            sched_df = load_results_to_dataframe(Path(sched_cfg.artifacts_dir), experiment_name='Schedule')
            
            if not temp_df.empty and not sched_df.empty:
                create_combined_temp_schedule_plots(temp_df, sched_df, output_dir)
        
        LOGGER.info("Plotting complete.")
    else:
        LOGGER.info("No data found to plot.")

if __name__ == "__main__":
    main()

