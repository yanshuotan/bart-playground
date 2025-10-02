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


def _gelman_rubin_single_run(seed, X, y, n_chains, ndpost, nskip, n_trees, proposal_probs_mtmh, proposal_probs_default):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    # MultiBART
    chains_mtmh = []
    rmse_chains_mtmh = []
    for i in range(n_chains):
        bart = MultiBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                         proposal_probs=proposal_probs_mtmh, multi_tries=10, random_state=seed*100+i)
        bart.fit(X_train, y_train, quietly=True)
        sigmas = [trace.global_params['eps_sigma2'] for trace in bart.sampler.trace]
        preds = bart.posterior_f(X_test, backtransform=True)
        rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]
        chains_mtmh.append(sigmas)
        rmse_chains_mtmh.append(rmses)
    chains_array = np.array(chains_mtmh)
    idata = az.from_dict(posterior={"eps_sigma2": chains_array})
    rhat = az.rhat(idata, var_names=["eps_sigma2"])
    rhat_mtmh = float(rhat["eps_sigma2"])
    # RMSE Rhat
    rmse_array = np.array(rmse_chains_mtmh)
    idata_rmse = az.from_dict(posterior={"test_rmse": rmse_array})
    rhat_rmse = az.rhat(idata_rmse, var_names=["test_rmse"])
    rhat_mtmh_rmse = float(rhat_rmse["test_rmse"])

    # DefaultBART
    chains_default = []
    rmse_chains_default = []
    for i in range(n_chains):
        bart_default = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                                  proposal_probs=proposal_probs_default, random_state=seed*100+i)
        bart_default.fit(X_train, y_train, quietly=True)
        sigmas = [trace.global_params['eps_sigma2'] for trace in bart_default.sampler.trace]
        preds = bart_default.posterior_f(X_test, backtransform=True)
        rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]
        chains_default.append(sigmas)
        rmse_chains_default.append(rmses)
    chains_array = np.array(chains_default)
    idata = az.from_dict(posterior={"eps_sigma2": chains_array})
    rhat = az.rhat(idata, var_names=["eps_sigma2"])
    rhat_default = float(rhat["eps_sigma2"])
    # RMSE Rhat
    rmse_array = np.array(rmse_chains_default)
    idata_rmse = az.from_dict(posterior={"test_rmse": rmse_array})
    rhat_rmse = az.rhat(idata_rmse, var_names=["test_rmse"])
    rhat_default_rmse = float(rhat_rmse["test_rmse"])

    return rhat_mtmh, rhat_default, rhat_mtmh_rmse, rhat_default_rmse

def gelman_rubin_r_compare(
    X, y,
    n_runs=5, n_chains=4,
    ndpost=1000, nskip=200, n_trees=100,
    proposal_probs_mtmh={"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1},
    proposal_probs_default={"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1},
    n_jobs=1
):
    results = Parallel(n_jobs=n_jobs)(
        delayed(_gelman_rubin_single_run)(
            seed, X, y, n_chains, ndpost, nskip, n_trees, proposal_probs_mtmh, proposal_probs_default
        ) for seed in range(n_runs)
    )
    rhat_mtmh, rhat_default, rhat_mtmh_rmse, rhat_default_rmse = zip(*results)
    df = pd.DataFrame({
        "MultiBART_Rhat_Sigma2": rhat_mtmh,
        "DefaultBART_Rhat_Sigma2": rhat_default,
        "MultiBART_Rhat_RMSE": rhat_mtmh_rmse,
        "DefaultBART_Rhat_RMSE": rhat_default_rmse
    })
    return df


def _bart_mse_single_run(seed, X, y, n_skip, n_post, n_trees, proposal_probs_mtmh, proposal_probs_default):
    test_mse = {}
    train_mse = {}
    pi_length = {}
    coverage = {}
    train_time = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # bart_mtmh
    t0 = time.time()
    bart_mtmh = MultiBART(ndpost=n_post, nskip=n_skip, n_trees=n_trees, proposal_probs=proposal_probs_mtmh, multi_tries=10, random_state=seed)
    bart_mtmh.fit(X_train, y_train, quietly=True)
    train_time["bart_mtmh"] = time.time() - t0

    # bart_default
    t0 = time.time()
    bart = DefaultBART(ndpost=n_post, nskip=n_skip, n_trees=n_trees, proposal_probs=proposal_probs_default, random_state=seed)
    bart.fit(X_train, y_train, quietly=True)
    train_time["bart"] = time.time() - t0


    # Test MSE
    test_mse["bart_mtmh"] = mean_squared_error(y_test, bart_mtmh.predict(X_test))
    test_mse["bart"] = mean_squared_error(y_test, bart.predict(X_test))
    
    # bart: axis=1, shape (n_test, n_mcmc)
    bart_pred_all_test = bart.posterior_predict(X_test)
    bart_lower = np.percentile(bart_pred_all_test, 2.5, axis=1)
    bart_upper = np.percentile(bart_pred_all_test, 97.5, axis=1)
    pi_length["bart"] = np.mean(bart_upper - bart_lower)
    bart_covered = ((y_test >= bart_lower) & (y_test <= bart_upper)).mean()
    coverage["bart"] = bart_covered

    # bart_mtmh: axis=1, shape (n_test, n_mcmc)
    bart_pred_all_test = bart_mtmh.posterior_predict(X_test)
    bart_lower = np.percentile(bart_pred_all_test, 2.5, axis=1)
    bart_upper = np.percentile(bart_pred_all_test, 97.5, axis=1)
    pi_length["bart_mtmh"] = np.mean(bart_upper - bart_lower)
    bart_covered = ((y_test >= bart_lower) & (y_test <= bart_upper)).mean()
    coverage["bart_mtmh"] = bart_covered

    return {
        "test_mse": test_mse,
        "pi_length": pi_length,
        "coverage": coverage,
        "train_time": train_time
    }

def bart_mse_comparison(X, y, n_runs=10, n_skip=100, n_post=100, n_trees=100, n_jobs=1, 
                        proposal_probs_mtmh={"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1},
                        proposal_probs_default={"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}):
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_bart_mse_single_run)(seed, X, y, n_skip, n_post, n_trees, proposal_probs_mtmh, proposal_probs_default) for seed in range(n_runs)
    )

    test_mse_results = {name: [] for name in ["bart_mtmh", "bart"]}
    pi_length_results = {name: [] for name in ["bart_mtmh", "bart"]}
    coverage_results = {name: [] for name in ["bart_mtmh", "bart"]}
    time_results = {name: [] for name in ["bart_mtmh", "bart"]}

    for res in results_list:
        for k in test_mse_results: test_mse_results[k].append(res["test_mse"][k])
        for k in pi_length_results: pi_length_results[k].append(res["pi_length"][k])
        for k in coverage_results: coverage_results[k].append(res["coverage"][k])
        for k in time_results: time_results[k].append(res["train_time"][k])

    results = {
        "test_mse": pd.DataFrame(test_mse_results),
        "pi_length": pd.DataFrame(pi_length_results),
        "coverage": pd.DataFrame(coverage_results),
        "train_time": pd.DataFrame(time_results)
    }
    return results