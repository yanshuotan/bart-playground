from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
from bart_playground import *
from scipy.linalg import subspace_angles
import gc, psutil, os

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

def compute_vector_distances(trace, X):
    distances = []
    for i in range(len(trace) - 1):
        vec1 = trace[i].evaluate(X)
        vec2 = trace[i + 1].evaluate(X)
        dist = np.linalg.norm(vec1 - vec2)
        distances.append(dist)
    return distances

def compute_subspace_distances(trace, n_trees, chain_id=None):
    """Compute subspace distances between consecutive states in a trace.

    Note: To obtain .leaf_basis, turn on copy_cache=True inside samplers.py.
    """
    distances = []
    tree_ids = list(range(n_trees))
    for i in range(len(trace) - 1):
        try:
            basis1 = trace[i].leaf_basis(tree_ids)
            basis2 = trace[i + 1].leaf_basis(tree_ids)
            U, _ = np.linalg.qr(basis1)
            V, _ = np.linalg.qr(basis2)
            angles = subspace_angles(U, V)
            dist = np.linalg.norm(angles)
            distances.append(dist)
        except Exception as e:
            print(f"Memory used: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")
            raise RuntimeError(
                f"Error in compute_subspace_distances at chain_id={chain_id}, i={i}: {e}"
            ) from e
    return distances


def make_temperature_schedule(nskip: int, start_temp: float, end_temp: float) -> TemperatureSchedule:
    """Create a temperature schedule that cools from start_temp to end_temp over nskip iterations.

    For iteration t < nskip, the temperature decreases linearly from start_temp to end_temp.
    For t >= nskip, the temperature is fixed at end_temp (typically 1.0).
    """

    def schedule(t: int) -> float:
        if t >= nskip:
            return end_temp
        if nskip <= 1:
            # Degenerate case: no real burn-in, just use end_temp
            return end_temp
        frac = t / (nskip - 1)
        return start_temp + (end_temp - start_temp) * frac

    return TemperatureSchedule(schedule)


def run_chain(chain_id, X, y, ndpost, nskip, n_trees, m_tries,
              tree_alpha, tree_beta, start_temp=1.0, end_temp=1.0,
              store_preds=False, n_test_points=None):
    """Run a single MCMC chain with a cooling temperature schedule.

    The temperature starts at start_temp and linearly decreases to end_temp over the
    burn-in period of length nskip. After that, ndpost samples are collected
    at temperature end_temp (usually 1.0).
    """

    n_features = X.shape[1]

    # Use the same train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Temperature schedule: start_temp -> end_temp over nskip iterations, then fixed at end_temp
    temp_schedule = make_temperature_schedule(nskip, start_temp=start_temp, end_temp=end_temp)

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
                          temperature=temp_schedule,
                          random_state=chain_id)
    bart_mtmh.fit(X_train, y_train)
    
    # Extract MTMH BART results
    sigmas_mtmh = [trace.global_params['eps_sigma2'] for trace in bart_mtmh.sampler.trace]
    preds_mtmh = bart_mtmh.posterior_f(X_test, backtransform=True)
    rmses_mtmh = [root_mean_squared_error(y_test, preds_mtmh[:, k]) for k in range(preds_mtmh.shape[1])]
    leaves_mtmh = [count_leaves_in_trees(trace) for trace in bart_mtmh.sampler.trace]
    depths_mtmh = [calculate_avg_depth_per_trace(trace) for trace in bart_mtmh.sampler.trace]
    feature_ratios_mtmh = get_feature_split_ratios(bart_mtmh.sampler.trace, n_features)
    vector_distances_mtmh = compute_vector_distances(bart_mtmh.sampler.trace, X_train)
    # subspace_distances_mtmh = compute_subspace_distances(bart_mtmh.sampler.trace, n_trees, run_id, chain_id)
    accepted_moves_logmh_mtmh = np.array(bart_mtmh.sampler.accepted_moves_logmh, dtype=object)

    # MTMH prediction interval and coverage
    mtmh_pred_all_test = bart_mtmh.posterior_predict(X_test)  # shape (n_test, n_mcmc)
    mtmh_lower = np.percentile(mtmh_pred_all_test, 2.5, axis=1)
    mtmh_upper = np.percentile(mtmh_pred_all_test, 97.5, axis=1)
    mtmh_covered_bool = ((y_test >= mtmh_lower) & (y_test <= mtmh_upper))  # shape (n_test,)

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
                               temperature=temp_schedule,
                               random_state=chain_id)
    bart_default.fit(X_train, y_train)
    
    # Extract default BART results
    sigmas_default = [trace.global_params['eps_sigma2'] for trace in bart_default.sampler.trace]
    preds_default = bart_default.posterior_f(X_test, backtransform=True)
    rmses_default = [root_mean_squared_error(y_test, preds_default[:, k]) for k in range(preds_default.shape[1])]
    leaves_default = [count_leaves_in_trees(trace) for trace in bart_default.sampler.trace]
    depths_default = [calculate_avg_depth_per_trace(trace) for trace in bart_default.sampler.trace]
    feature_ratios_default = get_feature_split_ratios(bart_default.sampler.trace, n_features)
    vector_distances_default = compute_vector_distances(bart_default.sampler.trace, X_train)
    # subspace_distances_default = compute_subspace_distances(bart_default.sampler.trace, n_trees, run_id, chain_id)
    accepted_moves_logmh_default = np.array(bart_default.sampler.accepted_moves_logmh, dtype=object)

    # Default prediction interval and coverage
    default_pred_all_test = bart_default.posterior_predict(X_test)  # shape (n_test, n_mcmc)
    default_lower = np.percentile(default_pred_all_test, 2.5, axis=1)
    default_upper = np.percentile(default_pred_all_test, 97.5, axis=1)
    default_covered_bool = ((y_test >= default_lower) & (y_test <= default_upper))  # shape (n_test,)

    del bart_default
    gc.collect()

    # Return results as dictionary, optionally include preds
    result = {
        'chain_id': chain_id,
        'default': {
            'sigmas': np.array(sigmas_default),
            'rmses': np.array(rmses_default),
            'leaves': np.array(leaves_default),
            'depths': np.array(depths_default),
            'feature_ratios': feature_ratios_default,  # shape: [n_iterations, n_features]
            'vector_distances': np.array(vector_distances_default),
            # 'subspace_distances': np.array(subspace_distances_default),
            'accepted_moves_logmh': accepted_moves_logmh_default
        },
        'mtmh': {
            'sigmas': np.array(sigmas_mtmh),
            'rmses': np.array(rmses_mtmh),
            'leaves': np.array(leaves_mtmh),
            'depths': np.array(depths_mtmh),
            'feature_ratios': feature_ratios_mtmh,  # shape: [n_iterations, n_features]
            'vector_distances': np.array(vector_distances_mtmh),
            # 'subspace_distances': np.array(subspace_distances_mtmh),
            'accepted_moves_logmh': accepted_moves_logmh_mtmh
        }
    }
    if store_preds:
        if n_test_points is not None and n_test_points < preds_default.shape[0]:
            rng = np.random.default_rng(42) # Choose test points consistently
            idx = rng.choice(preds_default.shape[0], n_test_points, replace=False)
            result['default']['preds'] = np.array(preds_default[idx])
            result['mtmh']['preds'] = np.array(preds_mtmh[idx])
            result['default']['coverage'] = np.array(default_covered_bool[idx])
            result['mtmh']['coverage'] = np.array(mtmh_covered_bool[idx])
        else:
            result['default']['preds'] = np.array(preds_default)
            result['mtmh']['preds'] = np.array(preds_mtmh)
            result['default']['coverage'] = np.array(default_covered_bool)
            result['mtmh']['coverage'] = np.array(mtmh_covered_bool)
    return result

def run_parallel_experiments(X, y, ndpost, nskip, n_trees, notebook, 
                             tree_alpha=0.95, tree_beta=2.0, m_tries=10,
                             n_chains=1000, n_jobs=-1, store_preds=False, n_test_points=None,
                             start_temp: float = 1.0, end_temp: float = 1.0):
    """Run many independent chains in parallel with a cooling temperature schedule.

    Each chain uses the same train-test split but a different random seed
    (based on chain_id). For each chain, the temperature starts at start_temp
    and cools down to end_temp over nskip iterations, then ndpost samples are
    collected at temperature end_temp.
    """

    # results: list of per-chain dictionaries
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_chain)(
            chain_id, X, y, ndpost, nskip, n_trees, m_tries,
            tree_alpha, tree_beta, start_temp, end_temp,
            store_preds, n_test_points
        )
        for chain_id in range(n_chains)
    )

    # Stack results into structured arrays: shape [n_chains, ndpost, ...]
    combined_results = {
        'default': {
            'sigmas': np.array([r['default']['sigmas'] for r in results]),
            'rmses': np.array([r['default']['rmses'] for r in results]),
            'leaves': np.array([r['default']['leaves'] for r in results]),
            'depths': np.array([r['default']['depths'] for r in results]),
            'feature_ratios': np.array([r['default']['feature_ratios'] for r in results]),
            'vector_distances': np.array([r['default']['vector_distances'] for r in results]),
            # 'subspace_distances': np.array([r['default']['subspace_distances'] for r in results]),
            'accepted_moves_logmh': np.array([r['default']['accepted_moves_logmh'] for r in results], dtype=object),
        },
        'mtmh': {
            'sigmas': np.array([r['mtmh']['sigmas'] for r in results]),
            'rmses': np.array([r['mtmh']['rmses'] for r in results]),
            'leaves': np.array([r['mtmh']['leaves'] for r in results]),
            'depths': np.array([r['mtmh']['depths'] for r in results]),
            'feature_ratios': np.array([r['mtmh']['feature_ratios'] for r in results]),
            'vector_distances': np.array([r['mtmh']['vector_distances'] for r in results]),
            # 'subspace_distances': np.array([r['mtmh']['subspace_distances'] for r in results]),
            'accepted_moves_logmh': np.array([r['mtmh']['accepted_moves_logmh'] for r in results], dtype=object),
        },
        'metadata': {
            'n_chains': n_chains,
            'ndpost': ndpost,
            'nskip': nskip,
            'n_trees': n_trees,
            'm_tries': m_tries,
            'tree_alpha': tree_alpha,
            'tree_beta': tree_beta,
            'temperature_start': start_temp,
            'temperature_end': end_temp,
        },
    }

    if store_preds:
        combined_results['default']['preds'] = np.array([r['default']['preds'] for r in results])
        combined_results['mtmh']['preds'] = np.array([r['mtmh']['preds'] for r in results])
        combined_results['default']['coverage'] = np.array([r['default']['coverage'] for r in results])
        combined_results['mtmh']['coverage'] = np.array([r['mtmh']['coverage'] for r in results])

    np.savez_compressed(f'store/{notebook}.npz', **combined_results)
    return results