from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
from bart_playground import *
import gc

def count_leaves_in_trees(trace_record):
    """Count leaves (vars == -1) in all trees of a single trace record and return average"""
    total_leaves = 0
    total_trees = len(trace_record.trees)
    
    for tree in trace_record.trees:
        # Count number of -1s in vars (leaf nodes)
        leaves_count = np.sum(np.array(tree.vars) == -1)
        total_leaves += leaves_count
    
    # Return average number of leaves per tree
    return total_leaves / total_trees

def calculate_tree_depth(tree):
    """Calculate tree depth as ceil(log2(position_of_last_-1)) - 1"""
    vars_array = np.array(tree.vars)
    
    # Find positions of all -1s (leaf nodes)
    leaf_positions = np.where(vars_array == -1)[0]
    
    # Get the last position of -1
    last_leaf_position = leaf_positions[-1]
    
    # Calculate depth
    depth = int(np.ceil(np.log2(last_leaf_position + 2))) - 1
    
    return depth

def calculate_avg_depth_per_trace(trace_record):
    """Calculate average tree depth for all trees in a trace record"""
    total_depth = 0
    total_trees = len(trace_record.trees)
    
    for tree in trace_record.trees:
        tree_depth = calculate_tree_depth(tree)
        total_depth += tree_depth
    
    return total_depth / total_trees

def get_feature_split_ratios(trace_list, n_features):
    ratios_per_trace = []
    for trace in trace_list:
        feature_counts = np.zeros(n_features)
        total_splits = 0
        for tree in trace.trees:
            for var in tree.vars:
                if var >= 0:
                    feature_counts[var] += 1
                    total_splits += 1
        if total_splits > 0:
            ratios = feature_counts / total_splits
        else:
            ratios = np.zeros(n_features)
        ratios_per_trace.append(ratios)
    return np.array(ratios_per_trace)


def run_experiment(run_id, chain_id, X, y, ndpost, nskip, n_trees, m_tries, 
                   tree_alpha, tree_beta, store_preds=False, n_test_points=None):
    """Run a single experiment with same train-test split but different initial trees"""
    
    n_features = X.shape[1]

    # Use the same train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    _, _, _, _, random_trees = generate_data_from_defaultbart_prior(
        X=X_train,
        n_trees=n_trees,
        random_state=run_id,
        return_latent=True
    )

    # Train MTMH BART model
    proposal_probs_mtmh = {
        'multi_grow': 0.25,
        'multi_prune': 0.25,
        'multi_change': 0.4,
        'multi_swap': 0.1
    }
    bart_mtmh = MultiBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees,
                          proposal_probs=proposal_probs_mtmh, multi_tries=m_tries, tol=1, 
                          tree_alpha=tree_alpha, tree_beta=tree_beta, # Only for mtmh prior
                          random_state=run_id*100+chain_id, init_trees=random_trees)
    bart_mtmh.fit(X_train, y_train)
    
    # Extract MTMH BART results
    sigmas_mtmh = [trace.global_params['eps_sigma2'] for trace in bart_mtmh.sampler.trace]
    preds_mtmh = bart_mtmh.posterior_f(X_test, backtransform=True)
    rmses_mtmh = [root_mean_squared_error(y_test, preds_mtmh[:, k]) for k in range(preds_mtmh.shape[1])]
    leaves_mtmh = [count_leaves_in_trees(trace) for trace in bart_mtmh.sampler.trace]
    depths_mtmh = [calculate_avg_depth_per_trace(trace) for trace in bart_mtmh.sampler.trace]
    feature_ratios_mtmh = get_feature_split_ratios(bart_mtmh.sampler.trace, n_features)

    del bart_mtmh
    gc.collect()
    
    # Train default BART model
    proposal_probs_default = {
        'grow': 0.25,
        'prune': 0.25,
        'change': 0.4,
        'swap': 0.1
    }
    bart_default = DefaultBART(ndpost=ndpost, nskip=nskip, n_trees=n_trees, tol=1, 
                               proposal_probs=proposal_probs_default, 
                               random_state=run_id*100+chain_id, init_trees=random_trees)
    bart_default.fit(X_train, y_train)
    
    # Extract default BART results
    sigmas_default = [trace.global_params['eps_sigma2'] for trace in bart_default.sampler.trace]
    preds_default = bart_default.posterior_f(X_test, backtransform=True)
    rmses_default = [root_mean_squared_error(y_test, preds_default[:, k]) for k in range(preds_default.shape[1])]
    leaves_default = [count_leaves_in_trees(trace) for trace in bart_default.sampler.trace]
    depths_default = [calculate_avg_depth_per_trace(trace) for trace in bart_default.sampler.trace]
    feature_ratios_default = get_feature_split_ratios(bart_default.sampler.trace, n_features)

    del bart_default
    gc.collect()

    # Return results as dictionary, optionally include preds
    result = {
        'run_id': run_id,
        'default': {
            'sigmas': np.array(sigmas_default),
            'rmses': np.array(rmses_default),
            'leaves': np.array(leaves_default),
            'depths': np.array(depths_default),
            'feature_ratios': feature_ratios_default  # shape: [n_iterations, n_features]
        },
        'mtmh': {
            'sigmas': np.array(sigmas_mtmh),
            'rmses': np.array(rmses_mtmh),
            'leaves': np.array(leaves_mtmh),
            'depths': np.array(depths_mtmh),
            'feature_ratios': feature_ratios_mtmh  # shape: [n_iterations, n_features]
        }
    }
    if store_preds:
        if n_test_points is not None and n_test_points < preds_default.shape[0]:
            idx = np.random.choice(preds_default.shape[0], n_test_points, replace=False)
            result['default']['preds'] = np.array(preds_default[idx])
            result['mtmh']['preds'] = np.array(preds_mtmh[idx])
        else:
            result['default']['preds'] = np.array(preds_default)
            result['mtmh']['preds'] = np.array(preds_mtmh)
    return result

def run_experiment_multiple_chains(run_id, X, y, ndpost, nskip, n_trees, m_tries, 
                                  tree_alpha, tree_beta, store_preds=False, n_test_points=None, n_chains=4):
    """Run multiple chains (experiments) under the same init/seed, record all results"""
    chain_results = []
    for chain_id in range(n_chains):
        result = run_experiment(
            run_id=run_id, 
            chain_id=chain_id,
            X=X, y=y, 
            ndpost=ndpost, nskip=nskip, n_trees=n_trees, m_tries=m_tries, 
            tree_alpha=tree_alpha, tree_beta=tree_beta, 
            store_preds=store_preds, n_test_points=n_test_points
        )
        result['chain_id'] = chain_id
        chain_results.append(result)
    return chain_results

def run_parallel_experiments(X, y, ndpost, nskip, n_trees, notebook, 
                             tree_alpha=0.95, tree_beta=2.0, m_tries=10, 
                             n_runs=5, n_jobs=-1, store_preds=True, n_test_points=20, n_chains=4):
    """Run parallel experiments, each with multiple chains under same init/seed"""
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_experiment_multiple_chains)(
            run_id, X, y, ndpost, nskip, n_trees, m_tries, 
            tree_alpha, tree_beta, store_preds, n_test_points, n_chains
        )
        for run_id in range(n_runs)
    )
    # results: [n_runs, n_chains, ...]
    combined_results = {
        'default': {
            'sigmas': np.array([[chain['default']['sigmas'] for chain in run] for run in results]),
            'rmses': np.array([[chain['default']['rmses'] for chain in run] for run in results]),
            'leaves': np.array([[chain['default']['leaves'] for chain in run] for run in results]),
            'depths': np.array([[chain['default']['depths'] for chain in run] for run in results]),
            'feature_ratios': np.array([[chain['default']['feature_ratios'] for chain in run] for run in results])
        },
        'mtmh': {
            'sigmas': np.array([[chain['mtmh']['sigmas'] for chain in run] for run in results]),
            'rmses': np.array([[chain['mtmh']['rmses'] for chain in run] for run in results]),
            'leaves': np.array([[chain['mtmh']['leaves'] for chain in run] for run in results]),
            'depths': np.array([[chain['mtmh']['depths'] for chain in run] for run in results]),
            'feature_ratios': np.array([[chain['mtmh']['feature_ratios'] for chain in run] for run in results])
        },
        'metadata': {
            'n_runs': n_runs,
            'ndpost': ndpost,
            'nskip': nskip,
            'n_trees': n_trees,
            'm_tries': m_tries,
            'n_chains': n_chains
        }
    }
    if store_preds:
        combined_results['default']['preds'] = np.array([[chain['default']['preds'] for chain in run] for run in results])
        combined_results['mtmh']['preds'] = np.array([[chain['mtmh']['preds'] for chain in run] for run in results])
    np.savez_compressed(f'store/{notebook}.npz', **combined_results)
    return results