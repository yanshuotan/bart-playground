import time
import numpy as np
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

# --- Define experiment parameters ---
# Vary the number of trees over these values.
trees_list = [5, 10, 20, 40]
# Number of independent runs per setting (to average runtime)
runs_per_setting = 3
# Use a small number of iterations for fast runtime experiments.
ndpost = 50
nskip = 100

results = {}  # Will store runtime results for each dataset

for dataset_info in DATASETS:
    dataset_name = dataset_info["name"]
    print(f"Processing dataset {dataset_name}")
    # Load dataset based on its source.
    if dataset_info["source"] == "sklearn" and dataset_info["name"] == "california_housing":
        data_bunch = fetch_california_housing(return_X_y=False, as_frame=True)
        df = data_bunch.frame
    else:
        df = fetch_data(dataset_info["name"], return_X_y=False)

    # Subsample to at most 100 samples for fast runs.
    if df.shape[0] > 100:
        df = df.sample(n=100, random_state=42)

    # Assume the target is the last column.
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    # Check the number of features.
    assert X.shape[1] == dataset_info["n_features"], (
        f"Dataset {dataset_name} expected {dataset_info['n_features']} features but got {X.shape[1]}"
    )

    non_marg_times = []
    marg_times = []

    for n_trees in trees_list:
        times_non = []
        times_marg = []
        print(f"Running dataset {dataset_name} for ntree {n_trees}")
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
        non_marg_times.append(avg_time_non)
        marg_times.append(avg_time_marg)

    # Save results for this dataset.
    results[dataset_name] = {
        "trees": trees_list,
        "non_marg": non_marg_times,
        "marg": marg_times
    }

    # --- Plot runtime vs number of trees for this dataset ---
    plt.figure(figsize=(8, 6))
    plt.plot(trees_list, non_marg_times, marker='o', label='Non-Marginalized')
    plt.plot(trees_list, marg_times, marker='o', label='Marginalized')
    plt.xlabel("Number of Trees")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime vs Number of Trees for {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"runtime_{dataset_name}.png")
    plt.show()

# --- Summary Printout ---
print("\nSummary of Runtime Results:")
for dataset_name, data in results.items():
    print(f"\nDataset: {dataset_name}")
    print("Trees\tNon-Marg (s)\tMarg (s)")
    for trees, non, marg in zip(data["trees"], data["non_marg"], data["marg"]):
        print(f"{trees}\t{non:.3f}\t\t{marg:.3f}")
