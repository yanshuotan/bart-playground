import numpy as np
import pandas as pd
import bartz
from stochtree import BARTModel
from bart_playground import *
# Add logging configuration before importing arviz
import logging
logging.getLogger('arviz.preview').setLevel(logging.WARNING)
import arviz as az
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from joblib import Parallel, delayed

def _gelman_rubin_single_run(seed, X, y, n_chains, burnin_values, window, n_trees, proposal_probs_mtmh, proposal_probs_default):
    """Single run for Gelman-Rubin comparison with different burnin values"""
    max_burnin = max(burnin_values)
    ndpost = max_burnin + window
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    
    results = {}
    
    # MultiBART - collect full trace
    chains_mtmh = []
    rmse_chains_mtmh = []
    for i in range(n_chains):
        bart = MultiBART(ndpost=ndpost, nskip=0, n_trees=n_trees,
                         proposal_probs=proposal_probs_mtmh, multi_tries=10, random_state=seed*100+i)
        bart.fit(X_train, y_train, quietly=True)
        sigmas = [trace.global_params['eps_sigma2'] for trace in bart.sampler.trace]
        preds = bart.posterior_f(X_test, backtransform=True)
        rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]
        chains_mtmh.append(sigmas)
        rmse_chains_mtmh.append(rmses)
    
    # DefaultBART - collect full trace
    chains_default = []
    rmse_chains_default = []
    for i in range(n_chains):
        bart_default = DefaultBART(ndpost=ndpost, nskip=0, n_trees=n_trees,
                                  proposal_probs=proposal_probs_default, random_state=seed*100+i)
        bart_default.fit(X_train, y_train, quietly=True)
        sigmas = [trace.global_params['eps_sigma2'] for trace in bart_default.sampler.trace]
        preds = bart_default.posterior_f(X_test, backtransform=True)
        rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]
        chains_default.append(sigmas)
        rmse_chains_default.append(rmses)
    
    # Calculate R-hat for different burnin values
    for burnin in burnin_values:
        start_idx = burnin
        end_idx = burnin + window
        
        # MultiBART R-hat for sigma2
        chains_subset = np.array([chain[start_idx:end_idx] for chain in chains_mtmh])
        idata = az.from_dict(posterior={"eps_sigma2": chains_subset})
        rhat = az.rhat(idata, var_names=["eps_sigma2"])
        rhat_mtmh_sigma = float(rhat["eps_sigma2"])
        
        # MultiBART R-hat for RMSE
        rmse_subset = np.array([chain[start_idx:end_idx] for chain in rmse_chains_mtmh])
        idata_rmse = az.from_dict(posterior={"test_rmse": rmse_subset})
        rhat_rmse = az.rhat(idata_rmse, var_names=["test_rmse"])
        rhat_mtmh_rmse = float(rhat_rmse["test_rmse"])
        
        # DefaultBART R-hat for sigma2
        chains_subset = np.array([chain[start_idx:end_idx] for chain in chains_default])
        idata = az.from_dict(posterior={"eps_sigma2": chains_subset})
        rhat = az.rhat(idata, var_names=["eps_sigma2"])
        rhat_default_sigma = float(rhat["eps_sigma2"])
        
        # DefaultBART R-hat for RMSE
        rmse_subset = np.array([chain[start_idx:end_idx] for chain in rmse_chains_default])
        idata_rmse = az.from_dict(posterior={"test_rmse": rmse_subset})
        rhat_rmse = az.rhat(idata_rmse, var_names=["test_rmse"])
        rhat_default_rmse = float(rhat_rmse["test_rmse"])
        
        results[burnin] = {
            'rhat_mtmh_sigma': rhat_mtmh_sigma,
            'rhat_default_sigma': rhat_default_sigma,
            'rhat_mtmh_rmse': rhat_mtmh_rmse,
            'rhat_default_rmse': rhat_default_rmse
        }
    
    return results

def gelman_rubin_r_compare(
    X, y,
    burnin_values=[0, 50, 100, 200, 500, 1000],
    window=1000,
    n_runs=5, n_chains=4,
    n_trees=100,
    n_jobs=1
):
    """Compare Gelman-Rubin R-hat across different burnin values"""
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    proposal_probs_default = {"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_gelman_rubin_single_run)(
            seed, X, y, n_chains, burnin_values, window, n_trees, proposal_probs_mtmh, proposal_probs_default
        ) for seed in range(n_runs)
    )
    
    # Organize results into DataFrame
    data_rows = []
    for run_idx, run_result in enumerate(results):
        for burnin in burnin_values:
            row = {
                'run': run_idx,
                'burnin': burnin,
                'MultiBART_Rhat_Sigma2': run_result[burnin]['rhat_mtmh_sigma'],
                'DefaultBART_Rhat_Sigma2': run_result[burnin]['rhat_default_sigma'],
                'MultiBART_Rhat_RMSE': run_result[burnin]['rhat_mtmh_rmse'],
                'DefaultBART_Rhat_RMSE': run_result[burnin]['rhat_default_rmse']
            }
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    return df


def _bart_mse_single_run(seed, X, y, burnin_values, window, n_trees):
    """Single run for MSE comparison with different burnin values"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import time

    max_burnin = max(burnin_values)
    ndpost = max_burnin + window
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    
    results = {}
    
    # Initialize containers for each burnin value
    for burnin in burnin_values:
        results[burnin] = {
            "test_mse": {},
            #"train_mse": {},
            "pi_length": {},
            "coverage": {},
            #"train_time": {}
        }

    # bart_mtmh - collect full trace
    t0 = time.time()
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    bart_mtmh = MultiBART(ndpost=ndpost, nskip=0, n_trees=n_trees, proposal_probs=proposal_probs_mtmh, multi_tries=10, random_state=seed)
    bart_mtmh.fit(X_train, y_train, quietly=True)
    train_time_mtmh = time.time() - t0

    # bart_default - collect full trace
    t0 = time.time()
    proposal_probs_default = {"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}
    bart = DefaultBART(ndpost=ndpost, nskip=0, n_trees=n_trees, proposal_probs=proposal_probs_default, random_state=seed)
    bart.fit(X_train, y_train, quietly=True)
    train_time_default = time.time() - t0

    # btz
    t0 = time.time()
    btz_model = bartz.BART.gbart(np.transpose(X_train), y_train, ntree=n_trees, ndpost=ndpost, nskip=0, seed=seed, printevery=None)
    btpred_all_test = np.array(btz_model.predict(np.transpose(X_test)))
    btpred_all_train = np.array(btz_model.predict(np.transpose(X_train)))
    train_time_btz = time.time() - t0

    # rf
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
    rf.fit(X_train, y_train)
    train_time_rf = time.time() - t0

    # xgb
    t0 = time.time()
    xgbr = xgb.XGBRegressor(n_estimators=n_trees, random_state=seed, verbosity=0)
    xgbr.fit(X_train, y_train)
    train_time_xgb = time.time() - t0

    # Calculate metrics for each burnin value
    for burnin in burnin_values:
        start_idx = burnin
        end_idx = burnin + window
        
        # Get subset predictions for BART models
        bart_mtmh_subset_f = bart_mtmh.posterior_f(X_test, backtransform=True)[:, start_idx:end_idx]
        bart_mtmh_subset_pred = np.mean(bart_mtmh_subset_f, axis=1)
        bart_mtmh_subset_f_train = bart_mtmh.posterior_f(X_train, backtransform=True)[:, start_idx:end_idx]
        bart_mtmh_subset_pred_train = np.mean(bart_mtmh_subset_f_train, axis=1)
        
        bart_subset_f = bart.posterior_f(X_test, backtransform=True)[:, start_idx:end_idx]
        bart_subset_pred = np.mean(bart_subset_f, axis=1)
        bart_subset_f_train = bart.posterior_f(X_train, backtransform=True)[:, start_idx:end_idx]
        bart_subset_pred_train = np.mean(bart_subset_f_train, axis=1)
        
        # Get subset predictions for bartz
        btpred_subset_test = np.mean(btpred_all_test[start_idx:end_idx], axis=0)
        btpred_subset_train = np.mean(btpred_all_train[start_idx:end_idx], axis=0)
        
        # Test MSE
        results[burnin]["test_mse"]["bart_mtmh"] = mean_squared_error(y_test, bart_mtmh_subset_pred)
        results[burnin]["test_mse"]["bart"] = mean_squared_error(y_test, bart_subset_pred)
        results[burnin]["test_mse"]["btz"] = mean_squared_error(y_test, btpred_subset_test)
        results[burnin]["test_mse"]["rf"] = mean_squared_error(y_test, rf.predict(X_test))
        results[burnin]["test_mse"]["xgb"] = mean_squared_error(y_test, xgbr.predict(X_test))

        # # Train MSE
        # results[burnin]["train_mse"]["bart_mtmh"] = mean_squared_error(y_train, bart_mtmh_subset_pred_train)
        # results[burnin]["train_mse"]["bart"] = mean_squared_error(y_train, bart_subset_pred_train)
        # results[burnin]["train_mse"]["btz"] = mean_squared_error(y_train, btpred_subset_train)
        # results[burnin]["train_mse"]["rf"] = mean_squared_error(y_train, rf.predict(X_train))
        # results[burnin]["train_mse"]["xgb"] = mean_squared_error(y_train, xgbr.predict(X_train))

        # Prediction intervals
        # btz
        eps = np.zeros_like(btpred_all_test[start_idx:end_idx])
        sigma_subset = btz_model.sigma[start_idx:end_idx]
        for i in range(window):
            eps[i] = np.random.normal(0, sigma_subset[i], size=btpred_all_test.shape[1])
        btpred_with_eps = btpred_all_test[start_idx:end_idx] + eps
        btz_lower = np.percentile(btpred_with_eps, 2.5, axis=0)
        btz_upper = np.percentile(btpred_with_eps, 97.5, axis=0)
        results[burnin]["pi_length"]["btz"] = np.mean(btz_upper - btz_lower)
        btz_covered = ((y_test >= btz_lower) & (y_test <= btz_upper)).mean()
        results[burnin]["coverage"]["btz"] = btz_covered
        
        # bart
        bart_pred_all_test = bart.posterior_predict(X_test)[:, start_idx:end_idx]
        bart_lower = np.percentile(bart_pred_all_test, 2.5, axis=1)
        bart_upper = np.percentile(bart_pred_all_test, 97.5, axis=1)
        results[burnin]["pi_length"]["bart"] = np.mean(bart_upper - bart_lower)
        bart_covered = ((y_test >= bart_lower) & (y_test <= bart_upper)).mean()
        results[burnin]["coverage"]["bart"] = bart_covered

        # bart_mtmh
        bart_mtmh_pred_all_test = bart_mtmh.posterior_predict(X_test)[:, start_idx:end_idx]
        bart_mtmh_lower = np.percentile(bart_mtmh_pred_all_test, 2.5, axis=1)
        bart_mtmh_upper = np.percentile(bart_mtmh_pred_all_test, 97.5, axis=1)
        results[burnin]["pi_length"]["bart_mtmh"] = np.mean(bart_mtmh_upper - bart_mtmh_lower)
        bart_mtmh_covered = ((y_test >= bart_mtmh_lower) & (y_test <= bart_mtmh_upper)).mean()
        results[burnin]["coverage"]["bart_mtmh"] = bart_mtmh_covered

        # # Training times (same for all burnin values since we use full trace)
        # results[burnin]["train_time"]["bart_mtmh"] = train_time_mtmh
        # results[burnin]["train_time"]["bart"] = train_time_default
        # results[burnin]["train_time"]["btz"] = train_time_btz
        # results[burnin]["train_time"]["rf"] = train_time_rf
        # results[burnin]["train_time"]["xgb"] = train_time_xgb

    return results

def bart_mse_comparison(X, y, burnin_values=[0, 50, 100, 200, 500, 1000], window=1000, n_runs=10, n_trees=100, n_jobs=1):
    """Compare MSE across different burnin values"""
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_bart_mse_single_run)(seed, X, y, burnin_values, window, n_trees) for seed in range(n_runs)
    )

    # Organize results
    organized_results = {}
    for burnin in burnin_values:
        organized_results[burnin] = {
            "test_mse": {name: [] for name in ["bart_mtmh", "bart", "btz", "rf", "xgb"]},
            #"train_mse": {name: [] for name in ["bart_mtmh", "bart", "btz", "rf", "xgb"]},
            "pi_length": {name: [] for name in ["bart_mtmh", "bart", "btz"]},
            "coverage": {name: [] for name in ["bart_mtmh", "bart", "btz"]},
            #"train_time": {name: [] for name in ["bart_mtmh", "bart", "btz", "rf", "xgb"]}
        }

    for res in results_list:
        for burnin in burnin_values:
            for metric in ["test_mse", 
                           #"train_mse", 
                           "pi_length", 
                           "coverage", 
                           #"train_time"
                           ]:
                for method in organized_results[burnin][metric]:
                    organized_results[burnin][metric][method].append(res[burnin][metric][method])

    # Convert to DataFrames
    final_results = {}
    for burnin in burnin_values:
        final_results[burnin] = {
            "test_mse": pd.DataFrame(organized_results[burnin]["test_mse"]),
            #"train_mse": pd.DataFrame(organized_results[burnin]["train_mse"]),
            "pi_length": pd.DataFrame(organized_results[burnin]["pi_length"]),
            "coverage": pd.DataFrame(organized_results[burnin]["coverage"]),
            #"train_time": pd.DataFrame(organized_results[burnin]["train_time"])
        }
    
    return final_results