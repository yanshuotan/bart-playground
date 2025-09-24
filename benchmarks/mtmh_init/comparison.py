import numpy as np
import pandas as pd
import bartz
from stochtree import BARTModel
from bart_playground import *
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
        preprocessor = DefaultPreprocessor()
        data = preprocessor.fit_transform(X_train, y_train)
        rng = np.random.default_rng(seed*1000+i)
        random_trees_uniform = create_random_init_trees(
            n_trees=n_trees,
            dataX=data.X,
            possible_thresholds=preprocessor.thresholds,
            generator=rng
        )
        bart = MultiBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                         proposal_probs=proposal_probs_mtmh, multi_tries=10, 
                         random_state=seed*100+i, init_trees=random_trees_uniform)
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
        preprocessor = DefaultPreprocessor()
        data = preprocessor.fit_transform(X_train, y_train)
        rng = np.random.default_rng(seed*1000+i)
        random_trees_uniform = create_random_init_trees(
            n_trees=n_trees,
            dataX=data.X,
            possible_thresholds=preprocessor.thresholds,
            generator=rng
        )
        bart_default = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                                  proposal_probs=proposal_probs_default, 
                                  random_state=seed*100+i, init_trees=random_trees_uniform)
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
    n_jobs=1
):
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    proposal_probs_default = {"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}
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


def _bart_mse_single_run(seed, X, y, n_skip, n_post, n_trees):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import time

    test_mse = {}
    train_mse = {}
    pi_length = {}
    coverage = {}
    train_time = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    preprocessor = DefaultPreprocessor()
    data = preprocessor.fit_transform(X_train, y_train)
    rng = np.random.default_rng(seed*100)
    random_trees_uniform = create_random_init_trees(
        n_trees=n_trees,
        dataX=data.X,
        possible_thresholds=preprocessor.thresholds,
        generator=rng
    )

    # bart_mtmh
    t0 = time.time()
    proposal_probs_mtmh = {"multi_grow": 0.25, "multi_prune": 0.25, "multi_change": 0.4, "multi_swap": 0.1}
    bart_mtmh = MultiBART(ndpost=n_post, nskip=n_skip, n_trees=n_trees, proposal_probs=proposal_probs_mtmh, 
                          multi_tries=10, random_state=seed, init_trees=random_trees_uniform)
    bart_mtmh.fit(X_train, y_train, quietly=True)
    train_time["bart_mtmh"] = time.time() - t0

    # bart_default
    t0 = time.time()
    proposal_probs_default = {"grow": 0.25, "prune": 0.25, "change": 0.4, "swap": 0.1}
    bart = DefaultBART(ndpost=n_post, nskip=n_skip, n_trees=n_trees, proposal_probs=proposal_probs_default, 
                       random_state=seed, init_trees=random_trees_uniform)
    bart.fit(X_train, y_train, quietly=True)
    train_time["bart"] = time.time() - t0

    # btz
    t0 = time.time()
    btz_model = bartz.BART.gbart(np.transpose(X_train), y_train, ntree=n_trees, ndpost=n_post, nskip=n_skip, seed=seed, printevery=None)
    btpred_all_test = np.array(btz_model.predict(np.transpose(X_test)))
    btpred_test = np.mean(btpred_all_test, axis=0)
    btpred_all_train = np.array(btz_model.predict(np.transpose(X_train)))
    btpred_train = np.mean(btpred_all_train, axis=0)
    train_time["btz"] = time.time() - t0

    # # sto
    # t0 = time.time()
    # sto = BARTModel()
    # sto.sample(X_train=X_train, y_train=y_train,
    #            num_gfr=0, num_burnin=n_skip, num_mcmc=n_post, 
    #            mean_forest_params={"num_trees": n_trees}, 
    #            general_params = {"random_seed": seed}) 
    # sto_pred_all_test, sto_all_sigma = sto.predict(X_test)
    # sto_pred_test = np.mean(sto_pred_all_test, axis=1)
    # sto_pred_train = np.mean(sto.predict(X_train)[0], axis=1)
    # train_time["sto"] = time.time() - t0

    # rf
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=seed)
    rf.fit(X_train, y_train)
    train_time["rf"] = time.time() - t0

    # xgb
    t0 = time.time()
    xgbr = xgb.XGBRegressor(n_estimators=n_trees, random_state=seed, verbosity=0)
    xgbr.fit(X_train, y_train)
    train_time["xgb"] = time.time() - t0

    # Test MSE
    test_mse["bart_mtmh"] = mean_squared_error(y_test, bart_mtmh.predict(X_test))
    test_mse["bart"] = mean_squared_error(y_test, bart.predict(X_test))
    test_mse["btz"] = mean_squared_error(y_test, btpred_test)
    # test_mse["sto"] = mean_squared_error(y_test, sto_pred_test)
    test_mse["rf"] = mean_squared_error(y_test, rf.predict(X_test))
    test_mse["xgb"] = mean_squared_error(y_test, xgbr.predict(X_test))

    # Train MSE
    train_mse["bart_mtmh"] = mean_squared_error(y_train, bart_mtmh.predict(X_train))
    train_mse["bart"] = mean_squared_error(y_train, bart.predict(X_train))
    train_mse["btz"] = mean_squared_error(y_train, btpred_train)
    # train_mse["sto"] = mean_squared_error(y_train, sto_pred_train)
    train_mse["rf"] = mean_squared_error(y_train, rf.predict(X_train))
    train_mse["xgb"] = mean_squared_error(y_train, xgbr.predict(X_train))

    # Prediction intervals
    # btz: axis=0, shape (n_mcmc, n_test)
    eps = np.zeros_like(btpred_all_test)
    for i in range(n_post):
        eps[i] = np.random.normal(0, btz_model.sigma[i], size=btpred_all_test.shape[1])
        btpred_all_test[i, :] += eps[i]    
    btz_lower = np.percentile(btpred_all_test, 2.5, axis=0)
    btz_upper = np.percentile(btpred_all_test, 97.5, axis=0)
    pi_length["btz"] = np.mean(btz_upper - btz_lower)
    btz_covered = ((y_test >= btz_lower) & (y_test <= btz_upper)).mean()
    coverage["btz"] = btz_covered
    
    # # sto: axis=1, shape (n_test, n_mcmc)
    # sto_eps = np.zeros_like(sto_pred_all_test)
    # for i in range(n_post):
    #     sto_eps[:, i] = np.random.normal(np.zeros(sto_pred_all_test.shape[0]), sto_all_sigma[:, i])
    #     sto_pred_all_test[:, i] += sto_eps[:, i]
    # sto_lower = np.percentile(sto_pred_all_test, 2.5, axis=1)
    # sto_upper = np.percentile(sto_pred_all_test, 97.5, axis=1)
    # pi_length["sto"] = np.mean(sto_upper - sto_lower)
    # sto_covered = ((y_test >= sto_lower) & (y_test <= sto_upper)).mean()
    # coverage["sto"] = sto_covered
    
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
        "train_mse": train_mse,
        "pi_length": pi_length,
        "coverage": coverage,
        "train_time": train_time
    }

def bart_mse_comparison(X, y, n_runs=10, n_skip=100, n_post=100, n_trees=100, n_jobs=1):
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_bart_mse_single_run)(seed, X, y, n_skip, n_post, n_trees) for seed in range(n_runs)
    )

    test_mse_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "rf", "xgb"]}
    train_mse_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "rf", "xgb"]}
    pi_length_results = {name: [] for name in ["bart_mtmh", "bart", "btz"]}
    coverage_results = {name: [] for name in ["bart_mtmh", "bart", "btz"]}
    time_results = {name: [] for name in ["bart_mtmh", "bart", "btz", "rf", "xgb"]}

    for res in results_list:
        for k in test_mse_results: test_mse_results[k].append(res["test_mse"][k])
        for k in train_mse_results: train_mse_results[k].append(res["train_mse"][k])
        for k in pi_length_results: pi_length_results[k].append(res["pi_length"][k])
        for k in coverage_results: coverage_results[k].append(res["coverage"][k])
        for k in time_results: time_results[k].append(res["train_time"][k])

    results = {
        "test_mse": pd.DataFrame(test_mse_results),
        "train_mse": pd.DataFrame(train_mse_results),
        "pi_length": pd.DataFrame(pi_length_results),
        "coverage": pd.DataFrame(coverage_results),
        "train_time": pd.DataFrame(time_results)
    }
    return results