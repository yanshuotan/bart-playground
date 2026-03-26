from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import numpy as np
from bart_playground import *
from scipy.linalg import subspace_angles
import gc, psutil, os
import csv
import json
from pathlib import Path

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


def crps_from_samples(samples, y_true):
    n_points, n_samples = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    samples_sorted = np.sort(samples, axis=1)
    k = np.arange(1, n_samples + 1)
    coeffs = (2 * k - n_samples - 1)[None, :]
    term2 = np.sum(coeffs * samples_sorted, axis=1) / (n_samples ** 2)
    return term1 - term2


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


def _as_serializable(value):
    if isinstance(value, np.ndarray):
        return [_as_serializable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_as_serializable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: _as_serializable(v) for k, v in value.items()}
    return value


def _make_setting_tag(notebook):
    safe_notebook = str(notebook).replace(' ', '_').replace('/', '_').replace('\\', '_')
    return safe_notebook


def _save_numeric_csv(file_path, data):
    arr = np.asarray(data)
    original_shape = arr.shape

    if arr.ndim == 0:
        arr2d = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr2d = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        # Keep first axis (usually chain axis), flatten remaining axes for CSV output.
        arr2d = arr.reshape(arr.shape[0], -1)

    np.savetxt(
        file_path,
        arr2d,
        delimiter=",",
        fmt="%.10g",
        header=f"original_shape={original_shape}",
    )


def _save_object_csv(file_path, values):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value_json"])
        for idx, value in enumerate(values):
            writer.writerow([idx, json.dumps(_as_serializable(value), ensure_ascii=True)])


def _save_run_results_to_csv(run_result, store_root, setting_tag):
    run_id = run_result['run_id']
    split_seed = run_result['split_seed']

    data_keys = [
        'sigmas',
        'rmses',
        'leaves',
        'depths',
        'feature_ratios',
        'vector_distances',
        'accepted_moves_logmh',
        'subsample_rmse',
        'subsample_crps',
    ]

    for data_name in data_keys:
        (store_root / data_name).mkdir(parents=True, exist_ok=True)

    model_names = ['mtmh']

    if 'preds' in run_result['mtmh'] and 'coverage' in run_result['mtmh']:
        (store_root / 'preds').mkdir(parents=True, exist_ok=True)
        (store_root / 'coverage').mkdir(parents=True, exist_ok=True)

    metadata_dir = store_root / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (store_root / 'subsample_X_test').mkdir(parents=True, exist_ok=True)
    (store_root / 'subsample_y_test').mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        for data_name in data_keys:
            file_name = (
                f"{setting_tag}__run{run_id:03d}"
                f"__{model_name}__{data_name}.csv"
            )
            file_path = store_root / data_name / file_name
            values = run_result[model_name][data_name]

            if data_name == 'accepted_moves_logmh':
                _save_object_csv(file_path, values)
            else:
                _save_numeric_csv(file_path, values)

        if 'preds' in run_result[model_name]:
            preds_name = (
                f"{setting_tag}__run{run_id:03d}"
                f"__{model_name}__preds.csv"
            )
            coverage_name = (
                f"{setting_tag}__run{run_id:03d}"
                f"__{model_name}__coverage.csv"
            )
            _save_numeric_csv(store_root / 'preds' / preds_name, run_result[model_name]['preds'])
            _save_numeric_csv(store_root / 'coverage' / coverage_name, run_result[model_name]['coverage'])

    if 'subsample' in run_result:
        x_sub_name = f"{setting_tag}__run{run_id:03d}__subsample_X_test.csv"
        y_sub_name = f"{setting_tag}__run{run_id:03d}__subsample_y_test.csv"
        _save_numeric_csv(store_root / 'subsample_X_test' / x_sub_name, run_result['subsample']['X_test'])
        _save_numeric_csv(store_root / 'subsample_y_test' / y_sub_name, run_result['subsample']['y_test'])

    metadata_file = metadata_dir / f"{setting_tag}__run{run_id:03d}.csv"
    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["run_id", run_id])
        writer.writerow(["split_seed", split_seed])
        for key, value in run_result['metadata'].items():
            writer.writerow([key, value])


def run_chain(chain_id, chain_seed, split_seed, X, y, ndpost, nskip, n_trees, m_tries,
              tree_alpha, tree_beta, start_temp=1.0, end_temp=1.0,
              store_preds=False, n_test_points=None, test_point_seed=42):
    """Run a single MCMC chain with a cooling temperature schedule.

    The temperature starts at start_temp and linearly decreases to end_temp over the
    burn-in period of length nskip. After that, ndpost samples are collected
    at temperature end_temp (usually 1.0).
    """

    n_features = X.shape[1]

    # Use run-specific train-test split and chain-specific random seed.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=split_seed)

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
                          random_state=chain_seed)
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
    
    # Build a deterministic subsample from test points (or all test points if n_test_points is None/large).
    if n_test_points is not None and n_test_points < X_test.shape[0]:
        rng = np.random.default_rng(test_point_seed)
        idx = rng.choice(X_test.shape[0], n_test_points, replace=False)
    else:
        idx = np.arange(X_test.shape[0])

    X_test_subsample = np.array(X_test[idx])
    y_test_subsample = np.array(y_test[idx])

    mtmh_pred_subsample = mtmh_pred_all_test[idx, :]
    mtmh_subsample_rmse = root_mean_squared_error(
        y_test_subsample,
        np.mean(mtmh_pred_subsample, axis=1)
    )
    mtmh_subsample_crps = float(np.mean(crps_from_samples(mtmh_pred_subsample, y_test_subsample)))

    # Return results as dictionary, optionally include preds
    result = {
        'chain_id': chain_id,
        'chain_seed': chain_seed,
        'split_seed': split_seed,
        'subsample': {
            'X_test': X_test_subsample,
            'y_test': y_test_subsample,
        },
        'mtmh': {
            'sigmas': np.array(sigmas_mtmh),
            'rmses': np.array(rmses_mtmh),
            'leaves': np.array(leaves_mtmh),
            'depths': np.array(depths_mtmh),
            'feature_ratios': feature_ratios_mtmh,  # shape: [n_iterations, n_features]
            'vector_distances': np.array(vector_distances_mtmh),
            # 'subspace_distances': np.array(subspace_distances_mtmh),
            'accepted_moves_logmh': accepted_moves_logmh_mtmh,
            'subsample_rmse': mtmh_subsample_rmse,
            'subsample_crps': mtmh_subsample_crps,
        }
    }
    if store_preds:
        if n_test_points is not None and n_test_points < preds_mtmh.shape[0]:
            result['mtmh']['preds'] = np.array(preds_mtmh[idx])
            result['mtmh']['coverage'] = np.array(mtmh_covered_bool[idx])
        else:
            result['mtmh']['preds'] = np.array(preds_mtmh)
            result['mtmh']['coverage'] = np.array(mtmh_covered_bool)
    return result

def _stack_run_results(chain_results, run_id, split_seed, n_chains, ndpost, nskip,
                       n_trees, m_tries, tree_alpha, tree_beta, start_temp, end_temp):
    run_result = {
        'run_id': run_id,
        'split_seed': split_seed,
        'chains': chain_results,
        'subsample': {
            'X_test': np.array(chain_results[0]['subsample']['X_test']),
            'y_test': np.array(chain_results[0]['subsample']['y_test']),
        },
        'mtmh': {
            'sigmas': np.array([r['mtmh']['sigmas'] for r in chain_results]),
            'rmses': np.array([r['mtmh']['rmses'] for r in chain_results]),
            'leaves': np.array([r['mtmh']['leaves'] for r in chain_results]),
            'depths': np.array([r['mtmh']['depths'] for r in chain_results]),
            'feature_ratios': np.array([r['mtmh']['feature_ratios'] for r in chain_results]),
            'vector_distances': np.array([r['mtmh']['vector_distances'] for r in chain_results]),
            'accepted_moves_logmh': np.array([r['mtmh']['accepted_moves_logmh'] for r in chain_results], dtype=object),
            'subsample_rmse': np.array([r['mtmh']['subsample_rmse'] for r in chain_results]),
            'subsample_crps': np.array([r['mtmh']['subsample_crps'] for r in chain_results]),
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
            'n_subsample_points': int(np.array(chain_results[0]['subsample']['y_test']).shape[0]),
        },
    }

    if 'preds' in chain_results[0]['mtmh']:
        run_result['mtmh']['preds'] = np.array([r['mtmh']['preds'] for r in chain_results])
        run_result['mtmh']['coverage'] = np.array([r['mtmh']['coverage'] for r in chain_results])

    return run_result


def run_parallel_experiments(X, y, ndpost, nskip, n_trees, notebook,
                             tree_alpha=0.95, tree_beta=2.0, m_tries=10,
                             n_runs=5, n_chains=1000, n_jobs=-1,
                             base_split_seed=42, base_chain_seed=2024,
                             run_start_id=0,
                             store_preds=False, n_test_points=None,
                             start_temp: float = 1.0, end_temp: float = 1.0,
                             store_dir='store'):
    """Run experiments by run and chain, and store all outputs as CSV by data type.

    - Different runs use different train-test splits (via split_seed).
    - Each run contains n_chains independent chains (unique chain_seed).
    - Temperature schedule is unchanged: cool from start_temp to end_temp over nskip.
    - Outputs are saved under store_dir/<data_name>/*.csv.
    """

    store_root = Path(store_dir)
    store_root.mkdir(parents=True, exist_ok=True)

    setting_tag = _make_setting_tag(notebook=notebook)

    all_run_results = []
    for run_offset in range(n_runs):
        run_id = run_start_id + run_offset
        split_seed = base_split_seed + run_id
        chain_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_chain)(
                chain_id=chain_id,
                chain_seed=base_chain_seed + run_id * n_chains + chain_id,
                split_seed=split_seed,
                X=X,
                y=y,
                ndpost=ndpost,
                nskip=nskip,
                n_trees=n_trees,
                m_tries=m_tries,
                tree_alpha=tree_alpha,
                tree_beta=tree_beta,
                start_temp=start_temp,
                end_temp=end_temp,
                store_preds=store_preds,
                n_test_points=n_test_points,
                test_point_seed=split_seed,
            )
            for chain_id in range(n_chains)
        )

        run_result = _stack_run_results(
            chain_results=chain_results,
            run_id=run_id,
            split_seed=split_seed,
            n_chains=n_chains,
            ndpost=ndpost,
            nskip=nskip,
            n_trees=n_trees,
            m_tries=m_tries,
            tree_alpha=tree_alpha,
            tree_beta=tree_beta,
            start_temp=start_temp,
            end_temp=end_temp,
        )
        _save_run_results_to_csv(run_result, store_root, setting_tag)
        all_run_results.append(run_result)

    return all_run_results