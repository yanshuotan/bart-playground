import numpy as np
import pandas as pd
import bartz
from stochtree import BARTModel
from bart_playground import *
import arviz as az
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def gelman_rubin_r_compare(
    X, y,
    n_runs=5, n_chains=4,
    ndpost=1000, nskip=100, n_trees=100
):
    rhat_mtmh = []
    rhat_default = []
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    proposal_probs_default = {"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}
    for seed in range(n_runs):
        X_train, _, y_train, _ = train_test_split(X, y, random_state=seed)
        # MultiBART
        chains_mtmh = []
        for i in range(n_chains):
            bart = MultiBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                             proposal_probs=proposal_probs_mtmh, multi_tries=10, random_state=seed*100+i)
            bart.fit(X_train, y_train, quietly = True)
            sigmas = [trace.global_params['eps_sigma2'] for trace in bart.sampler.trace]
            chains_mtmh.append(sigmas)
        chains_array = np.array(chains_mtmh)
        idata = az.from_dict(posterior={"eps_sigma2": chains_array})
        rhat = az.rhat(idata, var_names=["eps_sigma2"])
        rhat_mtmh.append(float(rhat["eps_sigma2"]))
        # DefaultBART
        chains_default = []
        for i in range(n_chains):
            bart_default = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                                      proposal_probs=proposal_probs_default, random_state=seed*100+i)
            bart_default.fit(X_train, y_train, quietly = True)
            sigmas = [trace.global_params['eps_sigma2'] for trace in bart_default.sampler.trace]
            chains_default.append(sigmas)
        chains_array = np.array(chains_default)
        idata = az.from_dict(posterior={"eps_sigma2": chains_array})
        rhat = az.rhat(idata, var_names=["eps_sigma2"])
        rhat_default.append(float(rhat["eps_sigma2"]))
    # Results DataFrame
    df = pd.DataFrame({
        "MultiBART_Rhat": rhat_mtmh,
        "DefaultBART_Rhat": rhat_default
    })
    return df

def bart_mse_comparison(X, y, n_runs=10, n_skip=100, n_post=100, n_trees=100):
    test_mse_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "sto", "rf", "xgb"]}
    train_mse_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "sto", "rf", "xgb"]}
    pi_length_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "sto"]}
    coverage_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "sto"]}
    time_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "sto", "rf", "xgb"]}

    for seed in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

        # bart_mtmh
        t0 = time.time()
        proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
        bart_mtmh = MultiBART(ndpost=n_post, nskip=n_skip, n_trees=n_trees, proposal_probs=proposal_probs_mtmh, multi_tries=10, random_state=seed)
        bart_mtmh.fit(X_train, y_train, quietly = True)
        time_results["bart_mtmh"].append(time.time() - t0)

        # bart_default
        t0 = time.time()
        proposal_probs_default = {"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}
        bart = DefaultBART(ndpost=n_post, nskip=n_skip, n_trees=n_trees, proposal_probs=proposal_probs_default, random_state=seed)
        bart.fit(X_train, y_train, quietly = True)
        time_results["bart"].append(time.time() - t0)

        # btz
        t0 = time.time()
        btz_model = bartz.BART.gbart(np.transpose(X_train), y_train, ntree=n_trees, ndpost=n_post, nskip=n_skip, seed=seed, printevery=None)
        btpred_all_test = np.array(btz_model.predict(np.transpose(X_test)))
        btpred_test = np.mean(btpred_all_test, axis=0)
        btpred_all_train = np.array(btz_model.predict(np.transpose(X_train)))
        btpred_train = np.mean(btpred_all_train, axis=0)
        time_results["btz"].append(time.time() - t0)

        # sto
        t0 = time.time()
        sto = BARTModel()
        sto.sample(X_train=X_train, y_train=y_train,
                   num_gfr=0, num_burnin=n_skip, num_mcmc=n_post, 
                   mean_forest_params={"num_trees": n_trees}, 
                   general_params = {"random_seed": seed},
                   variance_forest_params={"num_trees": n_trees // 2}) 
        sto_pred_all_test, sto_all_sigma = sto.predict(X_test)
        sto_pred_test = np.mean(sto_pred_all_test, axis=1)
        sto_pred_train = np.mean(sto.predict(X_train)[0], axis=1)
        time_results["sto"].append(time.time() - t0)

        # rf
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
        rf.fit(X_train, y_train)
        time_results["rf"].append(time.time() - t0)

        # xgb
        t0 = time.time()
        xgbr = xgb.XGBRegressor(n_estimators=n_trees, random_state=seed, verbosity=0)
        xgbr.fit(X_train, y_train)
        time_results["xgb"].append(time.time() - t0)

        # Test MSE
        test_mse_results["bart_mtmh"].append(mean_squared_error(y_test, bart_mtmh.predict(X_test)))
        test_mse_results["bart"].append(mean_squared_error(y_test, bart.predict(X_test)))
        test_mse_results["btz"].append(mean_squared_error(y_test, btpred_test))
        test_mse_results["sto"].append(mean_squared_error(y_test, sto_pred_test))
        test_mse_results["rf"].append(mean_squared_error(y_test, rf.predict(X_test)))
        test_mse_results["xgb"].append(mean_squared_error(y_test, xgbr.predict(X_test)))

        # Train MSE
        train_mse_results["bart_mtmh"].append(mean_squared_error(y_train, bart_mtmh.predict(X_train)))
        train_mse_results["bart"].append(mean_squared_error(y_train, bart.predict(X_train)))
        train_mse_results["btz"].append(mean_squared_error(y_train, btpred_train))
        train_mse_results["sto"].append(mean_squared_error(y_train, sto_pred_train))
        train_mse_results["rf"].append(mean_squared_error(y_train, rf.predict(X_train)))
        train_mse_results["xgb"].append(mean_squared_error(y_train, xgbr.predict(X_train)))

        # Prediction intervals
        # btz: axis=0, shape (n_mcmc, n_test)
        eps = np.zeros_like(btpred_all_test)
        for i in range(n_post):
            eps[i] = np.random.normal(0, btz_model.sigma[i], size=btpred_all_test.shape[1])
            btpred_all_test[i, :] += eps[i]    
        btz_lower = np.percentile(btpred_all_test, 2.5, axis=0)
        btz_upper = np.percentile(btpred_all_test, 97.5, axis=0)
        pi_length_results["btz"].append(np.mean(btz_upper - btz_lower))
        btz_covered = ((y_test >= btz_lower) & (y_test <= btz_upper)).mean()
        coverage_results["btz"].append(btz_covered)
        
        # sto: axis=1, shape (n_test, n_mcmc)
        sto_eps = np.zeros_like(sto_pred_all_test)
        for i in range(n_post):
            sto_eps[:, i] = np.random.normal(np.zeros(sto_pred_all_test.shape[0]), sto_all_sigma[:, i])
            sto_pred_all_test[:, i] += sto_eps[:, i]
        sto_lower = np.percentile(sto_pred_all_test, 2.5, axis=1)
        sto_upper = np.percentile(sto_pred_all_test, 97.5, axis=1)
        pi_length_results["sto"].append(np.mean(sto_upper - sto_lower))
        sto_covered = ((y_test >= sto_lower) & (y_test <= sto_upper)).mean()
        coverage_results["sto"].append(sto_covered)
        
        # bart: axis=1, shape (n_test, n_mcmc)
        bart_pred_all_test = bart.posterior_predict(X_test)
        bart_lower = np.percentile(bart_pred_all_test, 2.5, axis=1)
        bart_upper = np.percentile(bart_pred_all_test, 97.5, axis=1)
        pi_length_results["bart"].append(np.mean(bart_upper - bart_lower))
        bart_covered = ((y_test >= bart_lower) & (y_test <= bart_upper)).mean()
        coverage_results["bart"].append(bart_covered)

        # bart_mtmh: axis=1, shape (n_test, n_mcmc)
        bart_pred_all_test = bart_mtmh.posterior_predict(X_test)
        bart_lower = np.percentile(bart_pred_all_test, 2.5, axis=1)
        bart_upper = np.percentile(bart_pred_all_test, 97.5, axis=1)
        pi_length_results["bart_mtmh"].append(np.mean(bart_upper - bart_lower))
        bart_covered = ((y_test >= bart_lower) & (y_test <= bart_upper)).mean()
        coverage_results["bart_mtmh"].append(bart_covered)

    results = {
        "test_mse": pd.DataFrame(test_mse_results),
        "train_mse": pd.DataFrame(train_mse_results),
        "pi_length": pd.DataFrame(pi_length_results),
        "coverage": pd.DataFrame(coverage_results),
        "train_time": pd.DataFrame(time_results)
    }
    return results