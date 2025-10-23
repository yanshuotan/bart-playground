from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import pickle
import numpy as np
from bart_playground import *

def run_experiment(run_id, X, y, ndpost, nskip, n_trees, m_tries, notebook):
    """Run a single experiment with different train-test split"""
    
    # Use different random_state for train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=run_id)
    
    # Train default BART model
    proposal_probs_default = {
        'grow': 0.25,
        'prune': 0.25,
        'change': 0.4,
        'swap': 0.1
    }
    bart_default = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                    proposal_probs=proposal_probs_default, random_state=0)
    bart_default.fit(X_train, y_train)
    
    # Extract and save default BART results instead of the model
    sigmas_default = [trace.global_params['eps_sigma2'] for trace in bart_default.sampler.trace]
    preds_default = bart_default.posterior_f(X_test, backtransform=True)
    rmses_default = [root_mean_squared_error(y_test, preds_default[:, k]) for k in range(preds_default.shape[1])]
    
    # Save default BART results
    np.save(f'store/{notebook}_sigmas_default_run{run_id}.npy', np.array(sigmas_default))
    np.save(f'store/{notebook}_rmses_default_run{run_id}.npy', np.array(rmses_default))
    
    # Train MTMH BART model
    proposal_probs_mtmh = {
        'multi_grow': 0.25,
        'multi_prune': 0.25,
        'multi_change': 0.4,
        'multi_swap': 0.1
    }
    bart_mtmh = MultiBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                    proposal_probs=proposal_probs_mtmh, multi_tries=m_tries, random_state=0)
    bart_mtmh.fit(X_train, y_train)
    
    # Extract and save MTMH BART results
    sigmas_mtmh = [trace.global_params['eps_sigma2'] for trace in bart_mtmh.sampler.trace]
    preds_mtmh = bart_mtmh.posterior_f(X_test, backtransform=True)
    rmses_mtmh = [root_mean_squared_error(y_test, preds_mtmh[:, k]) for k in range(preds_mtmh.shape[1])]
    
    # Save MTMH BART results
    np.save(f'store/{notebook}_sigmas_mtmh_run{run_id}.npy', np.array(sigmas_mtmh))
    np.save(f'store/{notebook}_rmses_mtmh_run{run_id}.npy', np.array(rmses_mtmh))
    
    return run_id

def run_parallel_experiments(X, y, ndpost, nskip, n_trees, notebook, m_tries=10, n_runs=5, n_jobs=-1):
    """Run parallel experiments with different train-test splits"""
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_experiment)(run_id, X, y, ndpost, nskip, n_trees, m_tries, notebook)
        for run_id in range(n_runs)
    )
    
    return results