import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logger
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_results(results_dir):
    """
    Finds all 'results_dgp_*.csv' files in the given directory,
    loads them into a single pandas DataFrame.
    """
    search_pattern = os.path.join(results_dir, "results_dgp_*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        LOGGER.warning(f"No CSV files found matching the pattern: {search_pattern}")
        return pd.DataFrame()
        
    df_list = [pd.read_csv(file) for file in csv_files]
    aggregated_df = pd.concat(df_list, ignore_index=True)
    LOGGER.info(f"Aggregated {len(csv_files)} result files into a single DataFrame.")
    return aggregated_df

def create_dot_plot(df, n_train, plot_dir):
    """
    Generates and saves a dot plot for a specific n_train value.
    
    X-axis: R-squared at temperature = 1.0
    Y-axis: Difference in Gelman-Rubin (credible) between temp 1.0 and 2.0
    """
    LOGGER.info(f"Generating plot for n_train = {n_train}...")
    
    df_n = df[df['n_train'] == n_train].copy()
    
    if df_n.empty:
        LOGGER.warning(f"No data available for n_train = {n_train}. Skipping plot.")
        return

    # Pivot the table to get temperatures as columns
    df_pivot = df_n.pivot_table(
        index='dataset_name', 
        columns='temperature', 
        values=['r_squared', 'gr_rmse_credible']
    )
    
    # Flatten the multi-level column index
    df_pivot.columns = [f'{val}_{int(col)}' for val, col in df_pivot.columns]
    df_pivot.reset_index(inplace=True)

    # Check if required columns exist
    required_cols = ['r_squared_1', 'gr_rmse_credible_1', 'gr_rmse_credible_2']
    if not all(col in df_pivot.columns for col in required_cols):
        LOGGER.error(f"Missing required data for n_train = {n_train}. "
                     f"Make sure experiments for temperatures 1.0 and 2.0 have completed.")
        return

    # Calculate the difference in Gelman-Rubin
    df_pivot['gr_diff'] = df_pivot['gr_rmse_credible_1'] - df_pivot['gr_rmse_credible_2']
    
    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use regplot to add a scatter plot and a linear regression trend line
    sns.regplot(
        data=df_pivot,
        x='r_squared_1',
        y='gr_diff',
        ax=ax,
        scatter_kws={'s': 100, 'alpha': 0.8},
        line_kws={'color': 'red', 'linestyle': '--'}  # Style for the trend line
    )
    
    ax.set_title(f'Performance vs. Mixing Improvement (N_train = {n_train})', fontsize=16)
    ax.set_xlabel('R-squared (Temp = 1.0)', fontsize=12)
    ax.set_ylabel('Gelman-Rubin Difference (GR_Temp1 - GR_Temp2)', fontsize=12)
    
    # Add labels to each point
    for i, point in df_pivot.iterrows():
        ax.text(point['r_squared_1'] + 0.005, point['gr_diff'], str(point['dataset_name']), fontsize=8)

    # Save the figure
    plot_filename = f"snr_r2_vs_gr_diff_ntrain_{n_train}.png"
    output_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    LOGGER.info(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot results from the SNR hypothesis experiment.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments/consolidated_outputs/snr_hypothesis_runs",
        help="Directory containing the CSV result files."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="experiments/consolidated_outputs/snr_hypothesis_runs",
        help="Directory where plots will be saved."
    )
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    
    # 1. Aggregate all results
    full_df = aggregate_results(args.results_dir)
    
    if full_df.empty:
        LOGGER.error("No data to plot. Exiting.")
        return

    # 2. Generate plots for specified n_train values
    n_train_values_to_plot = [1000, 10000]
    for n_train in n_train_values_to_plot:
        create_dot_plot(full_df, n_train, args.plot_dir)

if __name__ == "__main__":
    main()
