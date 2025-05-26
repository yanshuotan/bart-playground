# Complete script comparing BART with vs. without XGBoost initialization

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from pmlb import fetch_data
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from bart_playground import DefaultBART, DefaultPreprocessor


# --- Mixing Diagnostic Functions ---
def gelman_rubin(chains):
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = np.mean(np.var(chains, axis=1, ddof=1))
    V = ((n - 1) / n) * W + B / n
    return np.sqrt(V / W)


def autocorrelation(chain, lag):
    n = len(chain)
    if lag >= n:
        return np.nan
    return np.corrcoef(chain[:-lag], chain[lag:])[0, 1]


def effective_sample_size(chains, step=1):
    m, n = chains.shape
    total_ess = 0.0
    for chain in chains:
        ac_sum = 0.0
        for lag in range(1, n, step):
            ac = autocorrelation(chain, lag)
            if ac < 0:
                break
            ac_sum += step * ac
        total_ess += n / (1 + 2 * ac_sum)
    return total_ess


def geweke(chain, first=0.1, last=0.5):
    n = len(chain)
    n_first = int(first * n)
    n_last = int(last * n)
    mean_first = np.mean(chain[:n_first])
    mean_last = np.mean(chain[-n_last:])
    var_first = np.var(chain[:n_first], ddof=1)
    var_last = np.var(chain[-n_last:], ddof=1)
    z = (mean_first - mean_last) / np.sqrt(var_first / n_first + var_last / n_last)
    return z


def average_geweke(chains):
    return np.mean([abs(geweke(chain)) for chain in chains])


# --- One Chain ---
# Updated run_chain_bart to use sampler.run directly, avoiding model.fit and shape mismatches
def run_chain_bart(X, y, ndpost=100, n_trees=50, seed=0, init_from_xgb=False):
    """
    Runs one chain by directly sampling from the initialized BART sampler.
    """
    # Split raw data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Preprocess training data
    preprocessor = DefaultPreprocessor(max_bins=10)
    train_data = preprocessor.fit_transform(X_train, y_train)

    # Initialize BART model state
    model = DefaultBART(
        ndpost=ndpost,
        nskip=200,
        n_trees=n_trees,
        random_state=seed,
        proposal_probs={'grow': 0.5, 'prune': 0.5}
    )
    model.preprocessor = preprocessor
    model.data = train_data
    # Ensure sampler has data and thresholds
    model.sampler.add_data(train_data)
    model.sampler.add_thresholds(preprocessor.thresholds)
    model.is_fitted = True

    # Optional XGBoost initialization
    if init_from_xgb:
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_trees,
            max_depth=3,
            learning_rate=.2,
            random_state=seed,
            tree_method="exact",
            grow_policy="depthwise",
            base_score=0.0  # disable default mean-centering
        )
        xgb_model.fit(train_data.X, train_data.y)
        model.init_from_xgboost(xgb_model, train_data.X, train_data.y)

    # Run sampler directly
    total_iters = ndpost + model.nskip
    start_time = time.perf_counter()
    trace = model.sampler.run(total_iters, quietly=True, n_skip=model.nskip)
    # Store trace in BART object for posterior_f
    model.trace = trace

    # Posterior predictions on raw test X
    post = model.posterior_f(X_test)
    # Ensure shape (n_test, ndpost)
    if post.shape[0] != X_test.shape[0]:
        post = post.T

    preds = np.mean(post, axis=1)
    lower = np.percentile(post, 2.5, axis=1)
    upper = np.percentile(post, 97.5, axis=1)
    mse = mean_squared_error(y_test, preds)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    # Chain samples for mixing diagnostics
    chain_sample = post[0, :]
    runtime = time.perf_counter() - start_time  # you can measure time externally if needed

    return {
        'predictions': preds,
        'mse': mse,
        'coverage': coverage,
        'chain_sample': chain_sample,
        'runtime': runtime
    }






# --- Run Multi-chain Experiment ---
def run_experiment(X, y, ndpost=100, n_trees=20, n_chains=4, init_from_xgb=False):
    results = [run_chain_bart(X, y, ndpost, n_trees, seed, init_from_xgb) for seed in range(n_chains)]
    return {
        'mse': np.mean([r['mse'] for r in results]),
        'coverage': np.mean([r['coverage'] for r in results]),
        'runtime': np.mean([r['runtime'] for r in results]),
        'chain_samples': np.array([r['chain_sample'] for r in results]),
        'predictions': [r['predictions'] for r in results]
    }


# --- Main ---
def main():
    cache_file = "xgb_init_results_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
        print("Loaded cached results.")
    else:
        results = {}

    datasets = {
        '1201_BNG_breastTumor': fetch_data("1201_BNG_breastTumor", return_X_y=True),
        '1199_BNG_echoMonths': fetch_data("1199_BNG_echoMonths", return_X_y=True),
        '294_satellite_image': fetch_data("294_satellite_image", return_X_y=True),
        'california_housing': fetch_california_housing(return_X_y=True),
    }

    config_flags = {'FromScratch': False, 'XGBoostInit': True}
    subsample_sizes = [100]

    for ds_name, (X, y) in datasets.items():
        for size in subsample_sizes:
            idx = np.random.choice(X.shape[0], size=min(size, X.shape[0]), replace=False)
            X_sub, y_sub = X[idx], y[idx]
            for label, flag in config_flags.items():
                key = (ds_name, size, label)
                if key not in results:
                    print(f"Running {ds_name} (n={size}) | {label}")
                    res = run_experiment(X_sub, y_sub, ndpost=100, n_trees=20, n_chains=4, init_from_xgb=flag)
                    results[key] = res
                    with open(cache_file, "wb") as f:
                        pickle.dump(results, f)

    # --- Plots ---
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    def make_barplot(metric_name, values):
        labels = sorted(set(k[0] + f"_{k[1]}" for k in results))
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        init_vals = [values.get((label, 'XGBoostInit'), np.nan) for label in labels]
        scratch_vals = [values.get((label, 'FromScratch'), np.nan) for label in labels]
        ax.bar(x - width / 2, init_vals, width, label='XGBoostInit', color='blue')
        ax.bar(x + width / 2, scratch_vals, width, label='FromScratch', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(metric_name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{metric_name.lower().replace(' ', '_')}.png"))

    def collect_metric(metric):
        vals = {}
        for (ds, n, cfg), res in results.items():
            key = (f"{ds}_{n}", cfg)
            vals[key] = res[metric]
        return vals

    make_barplot("MSE", collect_metric("mse"))
    make_barplot("Coverage", collect_metric("coverage"))
    make_barplot("Runtime (sec)", collect_metric("runtime"))

    print("Simulation complete.")


if __name__ == "__main__":
    main()
