import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmlb import fetch_data
from sklearn.datasets import fetch_california_housing

from bart_playground import DefaultBART

# --- Plot generation ---
# For each dataset, create a plot showing runtime vs. number of trees, with separate lines for each sample size.
results_df = pd.read_csv("runtime_results.csv")
datasets = results_df["dataset"].unique()
for dataset in datasets:
    df_dataset = results_df[results_df["dataset"] == dataset]
    # Create one figure with two subplots: one for non-marginalized and one for marginalized.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for sample_size in sorted(df_dataset["sample_size"].unique()):
        df_subset = df_dataset[df_dataset["sample_size"] == sample_size]
        # Ensure the data is sorted by number of trees.
        df_subset = df_subset.sort_values("n_trees")
        axes[0].plot(df_subset["n_trees"], df_subset["non_marg_runtime"],
                     marker='o', label=f"Sample size {sample_size}")
        axes[1].plot(df_subset["n_trees"], df_subset["marg_runtime"],
                     marker='o', label=f"Sample size {sample_size}")

    axes[0].set_title(f"{dataset}: Non-Marginalized")
    axes[0].set_xlabel("Number of Trees")
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title(f"{dataset}: Marginalized")
    axes[1].set_xlabel("Number of Trees")
    axes[1].grid(True)
    axes[1].legend()

    plt.suptitle(f"Runtime vs. Number of Trees for {dataset}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"runtime_{dataset}.png")
    plt.show()

print("Plot generation completed.")
