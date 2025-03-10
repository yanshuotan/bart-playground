import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmlb import fetch_data
from sklearn.datasets import fetch_california_housing

from bart_playground import DefaultBART

# --- Define datasets ---
DATASETS = [
    {"name": "1201_BNG_breastTumor", "n_features": 9, "source": "pmlb"},
    {"name": "california_housing", "n_features": 8, "source": "sklearn"},
    {"name": "1199_BNG_echoMonths", "n_features": 9, "source": "pmlb"},
    {"name": "294_satellite_image", "n_features": 36, "source": "pmlb"},
]

# --- Experiment parameters ---
trees_list = [5, 10, 20, 40, 100, 200]  # Vary number of trees.
sample_sizes = [100, 500]  # Different sample sizes from the data.
runs_per_setting = 3  # Independent runs per configuration.
ndpost = 100  # Posterior iterations.
nskip = 20  # Burn-in iterations.

# List to cache all results
results_list = []

# Loop over datasets.
for dataset_info in DATASETS:
    dataset_name = dataset_info["name"]
    print(f"Processing dataset {dataset_name}")
    # Load dataset.
    if dataset_info["source"] == "sklearn" and dataset_info["name"] == "california_housing":
        data_bunch = fetch_california_housing(return_X_y=False, as_frame=True)
        df = data_bunch.frame
    else:
        df = fetch_data(dataset_info["name"], return_X_y=False)

    # Check features.
    assert df.shape[1] - 1 == dataset_info["n_features"], (
        f"Dataset {dataset_name} expected {dataset_info['n_features']} features but got {df.shape[1] - 1}"
    )

    # Loop over different sample sizes.
    for sample_size in sample_sizes:
        if df.shape[0] > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()
        # Assume the target is the last column.
        X = df_sample.iloc[:, :-1].to_numpy()
        y = df_sample.iloc[:, -1].to_numpy()

        # Loop over different numbers of trees.
        for n_trees in trees_list:
            times_non = []
            times_marg = []
            print(f"Dataset: {dataset_name}, Sample size: {sample_size}, Trees: {n_trees}")
            for run in range(runs_per_setting):
                # --- Non-Marginalized run ---
                model = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                                    random_state=42 + run, marginalize=False)
                start = time.perf_counter()
                model.fit(X, y, quietly=True)
                end = time.perf_counter()
                times_non.append(end - start)

                # --- Marginalized run ---
                model = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                                    random_state=42 + run, marginalize=True)
                start = time.perf_counter()
                model.fit(X, y, quietly=True)
                end = time.perf_counter()
                times_marg.append(end - start)

            avg_time_non = np.mean(times_non)
            avg_time_marg = np.mean(times_marg)

            # Cache results.
            results_list.append({
                "dataset": dataset_name,
                "sample_size": sample_size,
                "n_trees": n_trees,
                "non_marg_runtime": avg_time_non,
                "marg_runtime": avg_time_marg
            })

# Save all results to CSV.
results_df = pd.DataFrame(results_list)
results_df.to_csv("runtime_results.csv", index=False)
print("\nRuntime results saved to runtime_results.csv")

# --- Plot generation ---
# For each dataset, create a plot showing runtime vs. number of trees, with separate lines for each sample size.

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
