from bart_playground import *
import numpy as np
import pandas as pd
# Add logging configuration before importing arviz
import logging
logging.getLogger('arviz.preview').setLevel(logging.WARNING)
import arviz as az
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from joblib import Parallel, delayed

def _single_run_burnin(seed, X, y, n_chains, burnin_values, window, n_trees, proposal_probs_mtmh, multi_tries):
    """Single run for multi-tries comparison with different burnin values"""
    max_burnin = max(burnin_values)
    ndpost = max_burnin + window
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    
    results = {}
    
    # Collect full traces for all chains
    chains_sigma2 = []
    chains_rmse = []
    test_mse = None
    coverage = {}
    train_time = None
    
    for i in range(n_chains):
        t0 = time.time()
        bart = MultiBART(
            ndpost=ndpost, nskip=0, n_trees=n_trees,
            proposal_probs=proposal_probs_mtmh,
            multi_tries=multi_tries,
            random_state=seed*100+i
        )
        bart.fit(X_train, y_train, quietly=True)
        
        if i == 0:
            train_time = time.time() - t0
        
        sigmas = [trace.global_params['eps_sigma2'] for trace in bart.sampler.trace]
        preds = bart.posterior_f(X_test, backtransform=True)
        rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]
        chains_sigma2.append(sigmas)
        chains_rmse.append(rmses)
        
        # Calculate coverage for first chain only (for performance)
        if i == 0:
            pred_all = bart.posterior_predict(X_test)
            # Store predictions for different burnin values
            for burnin in burnin_values:
                start_idx = burnin
                end_idx = burnin + window
                pred_subset = pred_all[:, start_idx:end_idx]
                lower = np.percentile(pred_subset, 2.5, axis=1)
                upper = np.percentile(pred_subset, 97.5, axis=1)
                coverage[burnin] = ((y_test >= lower) & (y_test <= upper)).mean()
    
    # Calculate R-hat for different burnin values
    for burnin in burnin_values:
        start_idx = burnin
        end_idx = burnin + window
        
        # R-hat for sigma2
        chains_subset = np.array([chain[start_idx:end_idx] for chain in chains_sigma2])
        idata = az.from_dict(posterior={"eps_sigma2": chains_subset})
        rhat = az.rhat(idata, var_names=["eps_sigma2"])
        rhat_sigma2 = float(rhat["eps_sigma2"])
        
        # R-hat for RMSE
        rmse_subset = np.array([chain[start_idx:end_idx] for chain in chains_rmse])
        idata_rmse = az.from_dict(posterior={"test_rmse": rmse_subset})
        rhat_rmse = az.rhat(idata_rmse, var_names=["test_rmse"])
        rhat_rmse_val = float(rhat_rmse["test_rmse"])
        
        # Test MSE using subset
        f_subset = bart.posterior_f(X_test, backtransform=True)[:, start_idx:end_idx]
        test_pred = np.mean(f_subset, axis=1)
        test_mse = mean_squared_error(y_test, test_pred)
        
        results[burnin] = {
            'rhat_sigma2': rhat_sigma2,
            'rhat_rmse': rhat_rmse_val,
            'test_mse': test_mse,
            'coverage': coverage[burnin],
            'train_time': train_time
        }
    
    return results


def multi_tries_burnin_comparison(
    X, y,
    multi_tries_list=[1, 5, 10, 50, 100],
    burnin_values=[0, 50, 100, 200, 500, 1000],
    window=1000,
    n_runs=5, n_chains=4,
    n_trees=100, n_jobs=4
):
    """Compare multi-tries performance across different burnin values"""
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    
    all_results = []
    
    for multi_tries in multi_tries_list:
        print(f"Processing multi_tries = {multi_tries}...")
        
        # Run parallel experiments
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_single_run_burnin)(
                seed, X, y, n_chains, burnin_values, window, n_trees, proposal_probs_mtmh, multi_tries
            ) for seed in range(n_runs)
        )
        
        # Organize results by burnin
        for run_idx, run_result in enumerate(parallel_results):
            for burnin in burnin_values:
                result_row = {
                    'run': run_idx,
                    'multi_tries': multi_tries,
                    'burnin': burnin,
                    'Rhat_Sigma2': run_result[burnin]['rhat_sigma2'],
                    'Rhat_RMSE': run_result[burnin]['rhat_rmse'],
                    'Test_MSE': run_result[burnin]['test_mse'],
                    'Coverage': run_result[burnin]['coverage'],
                    'Train_Time': run_result[burnin]['train_time']
                }
                all_results.append(result_row)
    
    df = pd.DataFrame(all_results)
    return df