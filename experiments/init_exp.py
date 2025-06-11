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
    if lag == 0:
        return 1.0
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

def run_chain_bart(X, y, ndpost=300, n_trees=20, seed=0, init_from_xgb=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    preprocessor = DefaultPreprocessor(max_bins=100)
    train_data = preprocessor.fit_transform(X_train, y_train)

    # If initializing from XGBoost, we'll focus on a single tree for debugging.
    bart_n_trees = 1 if init_from_xgb else n_trees

    model = DefaultBART(
        ndpost=ndpost,
        nskip=0,
        n_trees=bart_n_trees,
        random_state=seed,
        proposal_probs={'grow': 0.25, 'prune': 0.25, 'change': .4, 'swap': .1}
    )
    model.preprocessor = preprocessor
    model.data = train_data
    model.sampler.add_data(train_data)
    model.sampler.add_thresholds(preprocessor.thresholds)
    model.is_fitted = True

    if init_from_xgb:
        xgb_model = xgb.XGBRegressor(
            n_estimators=1, # Use 1 tree from XGBoost
            max_depth=3,    # With a single split
            learning_rate=0.05,
            random_state=seed,
            tree_method="exact",
            grow_policy="depthwise",
            base_score=0.0
        )
        xgb_model.fit(train_data.X, train_data.y)
        # Ensure BART itself expects the same number of trees it's being initialized with
        model.init_from_xgboost(xgb_model, train_data.X, train_data.y, debug=True)

    init_params = model.sampler.get_init_state()
    if not init_from_xgb and bart_n_trees == 1: # Added debug for FromScratch single tree
        print(f"[DEBUG FromScratch] Initial state from get_init_state():")
        print(f"[DEBUG FromScratch]   Tree 0 Leaf Vals (stump): {init_params.trees[0].leaf_vals[init_params.trees[0].leaves]}")
        print(f"[DEBUG FromScratch]   Global eps_sigma2: {init_params.global_params['eps_sigma2']}")

    init_pred_scaled = init_params.evaluate(X_test)
    init_pred = preprocessor.backtransform_y(init_pred_scaled)
    init_mse = mean_squared_error(y_test, init_pred)

    trace = model.sampler.run(ndpost, quietly=True, n_skip=0)
    model.trace = trace
    post = model.posterior_f(X_test)
    if post.shape[1] != X_test.shape[0]:
        post = post.T

    preds = np.mean(post, axis=0)
    mse = mean_squared_error(y_test, preds)
    mse_per_iter = [mean_squared_error(y_test, post[i]) for i in range(post.shape[0])]

    return {
        'init_mse': init_mse,
        'mse': mse,
        'coverage': np.mean((y_test >= np.percentile(post,2.5,axis=0)) & (y_test <= np.percentile(post,97.5,axis=0))),
        'chain_sample': post,
        'mse_per_iter': mse_per_iter,
        'runtime': time.perf_counter(),
        'y_test': y_test
    }

def run_experiment(X, y, ndpost=1000, n_trees=20, n_chains=5, init_from_xgb=False):
    results = [run_chain_bart(X, y, ndpost, n_trees, seed, init_from_xgb) for seed in range(n_chains)]
    posterior_stacks = np.array([r['chain_sample'] for r in results])  # (n_chains, n_iters, n_test)
    
    # Ensure posterior_stacks is (n_chains, n_test_samples, n_iters) for Gelman-Rubin if necessary,
    # or adjust Gelman-Rubin. For now, it's (n_chains, n_iters, n_test)
    # If chain_sample is (n_iters, n_test), then posterior_stacks is (n_chains, n_iters, n_test)
    # For Gelman-Rubin on a specific test point prediction, it might need reshaping/selection.
    
    y_test = results[0]['y_test']
    # Calculate mean predictions across chains for each iteration
    # posterior_stacks shape: (n_chains, n_iters, n_test_samples)
    # mean_preds_per_iter should be (n_iters, n_test_samples)
    mean_preds_per_iter = np.mean(posterior_stacks, axis=0) 

    mse_per_iter = [mean_squared_error(y_test, mean_preds_per_iter[i]) for i in range(mean_preds_per_iter.shape[0])]

    return {
        'init_mse': np.mean([r['init_mse'] for r in results]),
        'mse': np.mean([r['mse'] for r in results]),
        'coverage': np.mean([r['coverage'] for r in results]),
        'chain_samples': posterior_stacks,
        'runtime': np.mean([r['runtime'] for r in results]),
        'mse_per_iter': mse_per_iter
    }

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
    }

    config_flags = {'FromScratch': False, 'XGBoostInit': True}
    subsample_sizes = [500]
    # Use n_trees=1 for easier debugging when XGBoostInit is True
    experiment_n_trees = 20

    for ds_name, (X, y) in datasets.items():
        for size in subsample_sizes:
            idx = np.random.choice(X.shape[0], size=min(size, X.shape[0]), replace=False)
            X_sub, y_sub = X[idx], y[idx]
            for label, flag in config_flags.items():
                key = (ds_name, size, label)
                if key not in results:
                    current_n_trees = experiment_n_trees if flag else 20 # Use 1 tree for XGB init, 20 for scratch
                    print(f"Running {ds_name} (n={size}) | {label} with n_trees={current_n_trees}")
                    res = run_experiment(X_sub, y_sub, ndpost=300, n_trees=current_n_trees, n_chains=5, init_from_xgb=flag)
                    results[key] = res
                    mse_per_iter_preview = res['mse_per_iter'][:5] # Get first 5 MSEs
                    print(f"  Finished {ds_name} (n={size}) | {label} - Init MSE: {res['init_mse']:.4f}, MSE per Iter (first 5): {['{:.4f}'.format(m) for m in mse_per_iter_preview]}, Final MSE: {res['mse']:.4f}")
                    with open(cache_file, "wb") as f:
                        pickle.dump(results, f)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    for (ds, n, cfg), res in results.items():
        mse_series = res.get("mse_per_iter")
        if mse_series is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(mse_series)
            plt.xlabel("Iteration")
            plt.ylabel("Test MSE")
            plt.title(f"MSE per Iteration: {ds}_{n} | {cfg}")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{ds}_{n}_{cfg}_mse_per_iter.png"))
            plt.close()

if __name__ == "__main__":
    main()
