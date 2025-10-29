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

def _single_run(seed, X, y, n_chains, ndpost, nskip, n_trees, proposal_probs_mtmh, multi_tries):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    chains_sigma2 = []
    chains_rmse = []
    test_mse = None
    coverage = None
    train_time = None
    for i in range(n_chains):
        t0 = time.time()
        bart = MultiBART(
            ndpost=ndpost, nskip=nskip, n_trees=n_trees,
            proposal_probs=proposal_probs_mtmh,
            tree_alpha=0.8, tree_beta=3.0,  
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
        if i == 0:
            test_preds = bart.predict(X_test)
            test_mse = mean_squared_error(y_test, test_preds)
            bart_pred_all_test = bart.posterior_predict(X_test)
            bart_lower = np.percentile(bart_pred_all_test, 2.5, axis=1)
            bart_upper = np.percentile(bart_pred_all_test, 97.5, axis=1)
            coverage = ((y_test >= bart_lower) & (y_test <= bart_upper)).mean()
    # Rhat sigma2
    chains_array = np.array(chains_sigma2)
    idata = az.from_dict(posterior={"eps_sigma2": chains_array})
    rhat = az.rhat(idata, var_names=["eps_sigma2"])
    rhat_sigma2 = float(rhat["eps_sigma2"])
    # Rhat RMSE
    rmse_array = np.array(chains_rmse)
    idata_rmse = az.from_dict(posterior={"test_rmse": rmse_array})
    rhat_rmse = az.rhat(idata_rmse, var_names=["test_rmse"])
    rhat_rmse_val = float(rhat_rmse["test_rmse"])
    return rhat_sigma2, rhat_rmse_val, test_mse, coverage, train_time

def multi_tries_performance(
    X, y,
    multi_tries_list=[1, 5, 10, 50, 100],
    n_runs=5, n_chains=4,
    ndpost=500, nskip=200, n_trees=100, n_jobs=4
):
    results = {
        "multi_tries": [],
        "Rhat_Sigma2": [],
        "Rhat_RMSE": [],
        "Test_MSE": [],
        "Coverage": [],
        "Train_Time": []
    }
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    for multi_tries in multi_tries_list:
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_single_run)(
                seed, X, y, n_chains, ndpost, nskip, n_trees, proposal_probs_mtmh, multi_tries
            ) for seed in range(n_runs)
        )
        rhat_sigma2_list, rhat_rmse_list, test_mse_list, coverage_list, train_time_list = zip(*parallel_results)
        results["multi_tries"].append(multi_tries)
        results["Rhat_Sigma2"].append(np.mean(rhat_sigma2_list))
        results["Rhat_RMSE"].append(np.mean(rhat_rmse_list))
        results["Test_MSE"].append(np.mean(test_mse_list))
        results["Coverage"].append(np.mean(coverage_list))
        results["Train_Time"].append(np.mean(train_time_list))
    df = pd.DataFrame(results)
    return df