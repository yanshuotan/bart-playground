from __future__ import annotations

import csv
import gc
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error

from bart_playground import DefaultBART, MultiBART, ParallelTemperingBART
from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs


METHODS_SHORT = ["default", "default_pt", "mtmh", "mtmh_pt"]
METHODS_ALL = ["default", "default_pt", "mtmh", "mtmh_pt", "default_long"]


# ---------------------------------------------------------------------
# CSV / metadata utilities
# ---------------------------------------------------------------------

def _as_serializable(value: Any):
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


def _save_numeric_csv(file_path: Path, data):
    arr = np.asarray(data)
    original_shape = arr.shape
    if arr.ndim == 0:
        arr2d = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr2d = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr2d = arr
    else:
        arr2d = arr.reshape(arr.shape[0], -1)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        file_path,
        arr2d,
        delimiter=",",
        fmt="%.10g",
        header=f"original_shape={original_shape}",
    )


def _save_object_csv(file_path: Path, values):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value_json"])
        for idx, value in enumerate(values):
            writer.writerow([idx, json.dumps(_as_serializable(value), ensure_ascii=True)])


def _write_key_value_csv(file_path: Path, rows: dict[str, Any]):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in rows.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                value = json.dumps(_as_serializable(value), ensure_ascii=True)
            writer.writerow([key, value])


def _append_numeric_rows(file_path: Path, arr2d: np.ndarray):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "ab") as f:
        np.savetxt(f, np.asarray(arr2d), delimiter=",", fmt="%.10g")


def _load_2d_or_empty(path: Path, n_cols: int) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, n_cols))
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ---------------------------------------------------------------------
# Fixed test points + run-specific train subsamples
# ---------------------------------------------------------------------

def make_fixed100_splits(
    X,
    y,
    *,
    n_runs: int,
    n_fixed_test_points: int = 100,
    train_fraction: float = 0.75,
    fixed_test_seed: int = 42,
    base_train_seed: int = 2026,
):
    """Create fixed 100 test points and run-specific train subsets.

    This replaces sklearn train_test_split. The 100 test points are selected once
    and then reused for every run and every method. Each run draws a different
    train subset from the remaining pool. train_fraction defaults to 0.75 to stay
    close to sklearn train_test_split's default 75/25 train/test convention.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    if n_fixed_test_points >= n:
        raise ValueError("n_fixed_test_points must be smaller than number of rows")
    if not (0 < train_fraction <= 1):
        raise ValueError("train_fraction must be in (0, 1]")

    rng_test = np.random.default_rng(fixed_test_seed)
    test_idx = np.sort(rng_test.choice(n, size=n_fixed_test_points, replace=False))
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_pool_idx = np.flatnonzero(mask)
    train_size = int(np.floor(train_fraction * train_pool_idx.shape[0]))
    if train_size < 1:
        raise ValueError("train_fraction produced an empty train subset")

    splits = []
    for run_id in range(n_runs):
        rng_train = np.random.default_rng(base_train_seed + run_id)
        train_idx = np.sort(rng_train.choice(train_pool_idx, size=train_size, replace=False))
        splits.append(
            {
                "run_id": run_id,
                "train_seed": base_train_seed + run_id,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "X_train": X[train_idx],
                "y_train": y[train_idx],
                "X_test_fixed": X[test_idx],
                "y_test_fixed": y[test_idx],
            }
        )
    return splits


# ---------------------------------------------------------------------
# Model summary helpers copied from pt+mtmh_2000, with fixed-test outputs
# ---------------------------------------------------------------------

def count_leaves_in_trees(trace_record):
    total_leaves = 0
    total_trees = len(trace_record.trees)
    for tree in trace_record.trees:
        total_leaves += np.sum(np.array(tree.vars) == -1)
    return total_leaves / total_trees


def calculate_tree_depth(tree):
    vars_array = np.array(tree.vars)
    leaf_positions = np.where(vars_array == -1)[0]
    last_leaf_position = leaf_positions[-1]
    return int(np.ceil(np.log2(last_leaf_position + 2))) - 1


def calculate_avg_depth_per_trace(trace_record):
    total_depth = 0
    total_trees = len(trace_record.trees)
    for tree in trace_record.trees:
        total_depth += calculate_tree_depth(tree)
    return total_depth / total_trees


def crps_from_samples(samples, y_true):
    n_points, n_samples = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    samples_sorted = np.sort(samples, axis=1)
    k = np.arange(1, n_samples + 1)
    coeffs = (2 * k - n_samples - 1)[None, :]
    term2 = np.sum(coeffs * samples_sorted, axis=1) / (n_samples**2)
    return term1 - term2


def _posterior_predict_aligned(model, X, *, backtransform=True):
    """Posterior predictive samples aligned with model.range_post.

    posterior_f() already uses model.range_post. We use the same trace indices
    when drawing observation noise, so eps_sigma2 belongs to the same posterior
    draw as the fitted f sample.
    """
    preds = model.posterior_f(X, backtransform=False)
    for col, trace_idx in enumerate(model.range_post):
        eps_sigma2 = float(np.asarray(model.trace[trace_idx].global_params["eps_sigma2"]).reshape(-1)[0])
        preds[:, col] += model.sampler.generator.normal(
            0.0,
            np.sqrt(eps_sigma2),
            size=preds[:, col].shape,
        )
        if backtransform:
            preds[:, col] = model.preprocessor.backtransform_y(preds[:, col])
    return preds




def _vars_histogram_for_trace(trace, p: int) -> np.ndarray:
    hist = getattr(trace, "vars_histogram", None)
    if hist is None:
        out = np.zeros(p, dtype=float)
        return out
    hist = np.asarray(hist, dtype=float).reshape(-1)
    out = np.zeros(p, dtype=float)
    m = min(p, hist.shape[0])
    out[:m] = hist[:m]
    return out


def _trace_feature_matrix(traces, rmses, leaves, depths, X_test_fixed) -> tuple[np.ndarray, list[str]]:
    """Return per-iteration scalar/structure features for ACF and LDA.

    Rows correspond to posterior draws in the same order as preds[:, draw].
    Columns contain scalar diagnostics and feature split histograms.
    """
    p = int(np.asarray(X_test_fixed).shape[1])
    cols = ["eps_sigma2", "rmse", "avg_leaves", "avg_depth", "total_splits"] + [f"split_count_x{j+1}" for j in range(p)]
    mat = np.zeros((len(traces), len(cols)), dtype=float)
    for row, trace in enumerate(traces):
        hist = _vars_histogram_for_trace(trace, p)
        sigma2 = float(np.asarray(trace.global_params["eps_sigma2"]).reshape(-1)[0])
        mat[row, 0] = sigma2
        mat[row, 1] = float(rmses[row])
        mat[row, 2] = float(leaves[row])
        mat[row, 3] = float(depths[row])
        mat[row, 4] = float(hist.sum())
        mat[row, 5:] = hist
    return mat, cols

def summarize_model_outputs(model, X_test_fixed, y_test_fixed, *, store_preds: bool):
    f_preds = model.posterior_f(X_test_fixed, backtransform=True)
    y_pred_samples = _posterior_predict_aligned(model, X_test_fixed, backtransform=True)

    trace_indices = list(model.range_post)
    traces = [model.trace[k] for k in trace_indices]
    sigmas = np.array([trace.global_params["eps_sigma2"] for trace in traces])
    rmses = np.array([root_mean_squared_error(y_test_fixed, f_preds[:, k]) for k in range(f_preds.shape[1])])
    leaves = np.array([count_leaves_in_trees(trace) for trace in traces])
    depths = np.array([calculate_avg_depth_per_trace(trace) for trace in traces])
    accepted_moves = np.array(model.sampler.accepted_moves_logmh, dtype=object)

    trace_features, trace_feature_columns = _trace_feature_matrix(traces, rmses, leaves, depths, X_test_fixed)

    result = {
        "sigmas": sigmas,
        "rmses": rmses,
        "leaves": leaves,
        "depths": depths,
        "trace_features": trace_features,
        "trace_feature_columns": trace_feature_columns,
        "accepted_moves_logmh": accepted_moves,
        "subsample_rmse": root_mean_squared_error(y_test_fixed, np.mean(f_preds, axis=1)),
        "subsample_crps": float(np.mean(crps_from_samples(y_pred_samples, y_test_fixed))),
    }
    model_params = model.get_params()
    swap_accept_rates = np.asarray(model_params.get("swap_accept_rates", []), dtype=float)
    if swap_accept_rates.size > 0:
        result["swap_accept_rates"] = swap_accept_rates
        result["swap_temperatures"] = np.asarray(model_params.get("temperatures", []), dtype=float)
    if store_preds:
        lower = np.percentile(y_pred_samples, 2.5, axis=1)
        upper = np.percentile(y_pred_samples, 97.5, axis=1)
        result["preds"] = np.asarray(f_preds)
        result["pred_samples"] = np.asarray(y_pred_samples)
        result["coverage"] = np.asarray((y_test_fixed >= lower) & (y_test_fixed <= upper))
    return result


def quick_ladder_search(
    X,
    y,
    *,
    n_trees,
    tree_alpha,
    tree_beta,
    proposal_probs,
    target_rate=0.4,
    max_rounds=10,
    ndpost=500,
    nskip=500,
    n_repeats=3,
    random_state=123,
    swap_interval=5,
    post_swap_repair_steps=0,
    initial_temperatures=(1.0, 3.0),
    progress_print: bool = False,
    progress_prefix: str = "",
):
    """Adaptive temperature-ladder search with repeated fits per round."""
    temps = sorted({float(t) for t in initial_temperatures})
    if not temps:
        raise ValueError("initial_temperatures cannot be empty")
    if temps[0] != 1.0:
        temps = [1.0] + [t for t in temps if t != 1.0]

    history = []
    final_mean_rates = np.array([], dtype=float)

    if progress_print:
        print(
            f"{progress_prefix}[LADDER] start: n_points={X.shape[0]}, rounds<={max_rounds}, "
            f"repeats={n_repeats}, target_rate={target_rate}, init_temps={temps}",
            flush=True,
        )

    for round_id in range(max_rounds):
        round_rates = []

        for rep in range(n_repeats):
            model = ParallelTemperingBART(
                ndpost=ndpost,
                nskip=nskip,
                n_trees=n_trees,
                tree_alpha=tree_alpha,
                tree_beta=tree_beta,
                proposal_probs=proposal_probs,
                random_state=random_state + 1000 * round_id + rep,
                temperatures=temps,
                swap_interval=swap_interval,
                post_swap_repair_steps=post_swap_repair_steps,
                store_chain_traces=False,
                store_swap_diagnostics=False,
                print_swap_diagnostics=False,
            )
            model.fit(X, y, quietly=True)
            rates = np.asarray(model.get_params().get("swap_accept_rates", []), dtype=float)
            if rates.size == len(temps) - 1:
                round_rates.append(rates)
            del model
            gc.collect()

        if round_rates:
            mean_rates = np.mean(np.vstack(round_rates), axis=0)
        else:
            mean_rates = np.array([], dtype=float)

        if progress_print:
            rates_preview = np.round(mean_rates, 4).tolist() if mean_rates.size > 0 else []
            print(
                f"{progress_prefix}[LADDER] round {round_id + 1}/{max_rounds}: "
                f"n_temps={len(temps)}, mean_rates={rates_preview}",
                flush=True,
            )

        history.append(
            {
                "round": round_id,
                "temperatures": [float(t) for t in temps],
                "mean_swap_rates": mean_rates.tolist(),
                "all_swap_rates": [r.tolist() for r in round_rates],
            }
        )

        final_mean_rates = mean_rates
        if mean_rates.size == 0 or np.all(mean_rates >= target_rate):
            if progress_print:
                print(
                    f"{progress_prefix}[LADDER] stop: target reached or no valid rates. final_temps={temps}",
                    flush=True,
                )
            break

        new_temps = set(temps)
        for i, rate in enumerate(mean_rates):
            if rate < target_rate:
                t_low = temps[i]
                t_high = temps[i + 1]
                new_temps.add(2.0 * t_low * t_high / (t_low + t_high))

        updated_temps = sorted(new_temps)
        if len(updated_temps) == len(temps):
            if progress_print:
                print(f"{progress_prefix}[LADDER] stop: no new temperature inserted.", flush=True)
            break
        temps = updated_temps

    if progress_print:
        print(f"{progress_prefix}[LADDER] done: final n_temps={len(temps)}", flush=True)

    return [float(t) for t in temps], final_mean_rates.tolist(), history


# ---------------------------------------------------------------------
# Short-chain four-method experiment: default/default_pt/mtmh/mtmh_pt
# ---------------------------------------------------------------------

def run_short_chain(
    *,
    chain_id,
    chain_seed,
    X_train,
    y_train,
    X_test_fixed,
    y_test_fixed,
    ndpost,
    nskip,
    n_trees,
    tree_alpha,
    tree_beta,
    proposal_probs_default,
    proposal_probs_mtmh,
    temperatures,
    swap_interval,
    post_swap_repair_steps,
    multi_tries,
    dirichlet_prior: bool = False,
    s_alpha: float = 1.0,
    store_preds=True,
):
    """Run the original four methods sequentially inside one chain worker.

    This preserves pt+mtmh_2000's method definitions but uses the fixed 100 test
    points and does not do train_test_split internally.
    """
    out = {"chain_id": chain_id, "chain_seed": chain_seed}

    model = DefaultBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        tol=1,
        proposal_probs=proposal_probs_default,
        random_state=chain_seed,
        dirichlet_prior=dirichlet_prior,
        s_alpha=s_alpha,
    )
    model.fit(X_train, y_train)
    out["default"] = summarize_model_outputs(model, X_test_fixed, y_test_fixed, store_preds=store_preds)
    del model
    gc.collect()

    model = ParallelTemperingBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        tree_alpha=tree_alpha,
        tree_beta=tree_beta,
        tol=1,
        proposal_probs=proposal_probs_default,
        random_state=chain_seed,
        temperatures=temperatures,
        swap_interval=swap_interval,
        post_swap_repair_steps=post_swap_repair_steps,
        store_chain_traces=False,
        store_swap_diagnostics=False,
        print_swap_diagnostics=False,
        dirichlet_prior=dirichlet_prior,
        s_alpha=s_alpha,
    )
    model.fit(X_train, y_train)
    out["default_pt"] = summarize_model_outputs(model, X_test_fixed, y_test_fixed, store_preds=store_preds)
    del model
    gc.collect()

    model = MultiBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        tree_alpha=tree_alpha,
        tree_beta=tree_beta,
        tol=1,
        proposal_probs=proposal_probs_mtmh,
        random_state=chain_seed,
        multi_tries=multi_tries,
        dirichlet_prior=dirichlet_prior,
        s_alpha=s_alpha,
    )
    model.fit(X_train, y_train)
    out["mtmh"] = summarize_model_outputs(model, X_test_fixed, y_test_fixed, store_preds=store_preds)
    del model
    gc.collect()

    model = ParallelTemperingBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        tree_alpha=tree_alpha,
        tree_beta=tree_beta,
        tol=1,
        proposal_probs=proposal_probs_mtmh,
        random_state=chain_seed,
        temperatures=temperatures,
        swap_interval=swap_interval,
        post_swap_repair_steps=post_swap_repair_steps,
        store_chain_traces=False,
        store_swap_diagnostics=False,
        print_swap_diagnostics=False,
        sampler_kind="multi",
        multi_tries=multi_tries,
        dirichlet_prior=dirichlet_prior,
        s_alpha=s_alpha,
    )
    model.fit(X_train, y_train)
    out["mtmh_pt"] = summarize_model_outputs(model, X_test_fixed, y_test_fixed, store_preds=store_preds)
    del model
    gc.collect()

    return out


def save_short_run(
    *,
    store_root: Path,
    dataset_tag: str,
    run_id: int,
    split_info: dict[str, Any],
    chain_results: list[dict[str, Any]],
    metadata: dict[str, Any],
):
    for sub in ["preds", "pred_samples", "coverage", "sigmas", "rmses", "leaves", "depths", "trace_features", "trace_feature_columns", "accepted_moves_logmh", "subsample_rmse", "subsample_crps", "swap_accept_rates", "swap_temperatures", "subsample_X_test", "subsample_y_test", "indices", "metadata"]:
        (store_root / dataset_tag / sub).mkdir(parents=True, exist_ok=True)

    for method in METHODS_SHORT:
        arrays = {
            "sigmas": np.array([r[method]["sigmas"] for r in chain_results]),
            "rmses": np.array([r[method]["rmses"] for r in chain_results]),
            "leaves": np.array([r[method]["leaves"] for r in chain_results]),
            "depths": np.array([r[method]["depths"] for r in chain_results]),
            "subsample_rmse": np.array([r[method]["subsample_rmse"] for r in chain_results]),
            "subsample_crps": np.array([r[method]["subsample_crps"] for r in chain_results]),
        }
        for name, arr in arrays.items():
            _save_numeric_csv(store_root / dataset_tag / name / f"{dataset_tag}__run{run_id:03d}__{method}__{name}.csv", arr)
        _save_numeric_csv(
            store_root / dataset_tag / "trace_features" / f"{dataset_tag}__run{run_id:03d}__{method}__trace_features.csv",
            np.array([r[method]["trace_features"] for r in chain_results]),
        )
        _save_object_csv(
            store_root / dataset_tag / "trace_feature_columns" / f"{dataset_tag}__run{run_id:03d}__{method}__trace_feature_columns.csv",
            chain_results[0][method]["trace_feature_columns"],
        )
        _save_object_csv(
            store_root / dataset_tag / "accepted_moves_logmh" / f"{dataset_tag}__run{run_id:03d}__{method}__accepted_moves_logmh.csv",
            np.array([r[method]["accepted_moves_logmh"] for r in chain_results], dtype=object),
        )
        if "swap_accept_rates" in chain_results[0][method]:
            _save_numeric_csv(
                store_root / dataset_tag / "swap_accept_rates" / f"{dataset_tag}__run{run_id:03d}__{method}__swap_accept_rates.csv",
                np.array([r[method].get("swap_accept_rates", []) for r in chain_results], dtype=float),
            )
            _save_numeric_csv(
                store_root / dataset_tag / "swap_temperatures" / f"{dataset_tag}__run{run_id:03d}__{method}__swap_temperatures.csv",
                np.array([r[method].get("swap_temperatures", []) for r in chain_results], dtype=float),
            )
        if "preds" in chain_results[0][method]:
            _save_numeric_csv(
                store_root / dataset_tag / "preds" / f"{dataset_tag}__run{run_id:03d}__{method}__preds.csv",
                np.array([r[method]["preds"] for r in chain_results]),
            )
            if "pred_samples" in chain_results[0][method]:
                _save_numeric_csv(
                    store_root / dataset_tag / "pred_samples" / f"{dataset_tag}__run{run_id:03d}__{method}__pred_samples.csv",
                    np.array([r[method]["pred_samples"] for r in chain_results]),
                )
            _save_numeric_csv(
                store_root / dataset_tag / "coverage" / f"{dataset_tag}__run{run_id:03d}__{method}__coverage.csv",
                np.array([r[method]["coverage"] for r in chain_results]),
            )

    _save_numeric_csv(store_root / dataset_tag / "subsample_X_test" / f"{dataset_tag}__run{run_id:03d}__subsample_X_test.csv", split_info["X_test_fixed"])
    _save_numeric_csv(store_root / dataset_tag / "subsample_y_test" / f"{dataset_tag}__run{run_id:03d}__subsample_y_test.csv", split_info["y_test_fixed"])
    _save_numeric_csv(store_root / dataset_tag / "indices" / f"{dataset_tag}__run{run_id:03d}__train_idx.csv", split_info["train_idx"])
    _save_numeric_csv(store_root / dataset_tag / "indices" / f"{dataset_tag}__run{run_id:03d}__fixed_test_idx.csv", split_info["test_idx"])
    _write_key_value_csv(store_root / dataset_tag / "metadata" / f"{dataset_tag}__run{run_id:03d}__short_metadata.csv", metadata)


# ---------------------------------------------------------------------
# Long-chain default baseline with disk streaming
# ---------------------------------------------------------------------

def _thin_trace(trace, *, ndpost: int, nskip: int, store_every: int):
    if store_every < 1:
        raise ValueError("store_every must be >= 1")
    trimmed = list(trace)
    # DefaultBART trace includes the initial state in this repo: len = ndpost + 1.
    # Drop it so each chunk contributes exactly chunk_post states before thinning.
    if nskip == 0 and len(trimmed) == ndpost + 1:
        trimmed = trimmed[1:]
    elif len(trimmed) > ndpost:
        trimmed = trimmed[-ndpost:]
    return trimmed[::store_every]


def summarize_thinned_default(model, X_test_fixed, y_test_fixed, *, store_preds: bool):
    preds = model.posterior_f(X_test_fixed, backtransform=True)
    sigmas = np.array([trace.global_params["eps_sigma2"] for trace in model.trace])
    rmses = np.array([root_mean_squared_error(y_test_fixed, preds[:, k]) for k in range(preds.shape[1])])
    out = {"sigmas": sigmas, "rmses": rmses}
    if store_preds:
        out["preds"] = np.asarray(preds)
    return out


def run_long_default_chain_streaming(
    *,
    chain_id,
    chain_seed,
    run_id,
    X_train,
    y_train,
    X_test_fixed,
    y_test_fixed,
    ndpost,
    nskip,
    n_trees,
    proposal_probs_default,
    store_every,
    chunk_size,
    tmp_dir: str | Path,
    store_preds=True,
    dirichlet_prior: bool = False,
    s_alpha: float = 1.0,
):
    if ndpost % store_every != 0:
        raise ValueError("ndpost must be divisible by store_every")
    if chunk_size % store_every != 0:
        raise ValueError("chunk_size must be divisible by store_every")

    chain_dir = Path(tmp_dir) / f"run{run_id:03d}_chain{chain_id:03d}"
    if chain_dir.exists():
        shutil.rmtree(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)
    sigmas_file = chain_dir / "sigmas_rows.csv"
    rmses_file = chain_dir / "rmses_rows.csv"
    preds_file = chain_dir / "preds_rows.csv"

    first_chunk = min(chunk_size, ndpost)
    model = DefaultBART(
        ndpost=first_chunk,
        nskip=nskip,
        n_trees=n_trees,
        tol=1,
        proposal_probs=proposal_probs_default,
        random_state=chain_seed,
        dirichlet_prior=dirichlet_prior,
        s_alpha=s_alpha,
    )
    model.fit(X_train, y_train)

    remaining = ndpost
    chunks_done = 0
    last_state = None
    while remaining > 0:
        if chunks_done == 0:
            chunk_post = first_chunk
            chunk_trace = model.trace
            current_nskip = nskip
        else:
            chunk_post = min(chunk_size, remaining)
            chunk_trace = model.sampler.continue_run(chunk_post, last_state=last_state, quietly=True)
            current_nskip = 0

        thinned = _thin_trace(chunk_trace, ndpost=chunk_post, nskip=current_nskip, store_every=store_every)
        if len(thinned) == 0:
            raise RuntimeError("No thinned states collected from long-chain chunk")

        model.trace = thinned
        model.ndpost = len(thinned)
        model.nskip = 0
        chunk_result = summarize_thinned_default(model, X_test_fixed, y_test_fixed, store_preds=store_preds)
        _append_numeric_rows(sigmas_file, np.asarray(chunk_result["sigmas"]).reshape(-1, 1))
        _append_numeric_rows(rmses_file, np.asarray(chunk_result["rmses"]).reshape(-1, 1))
        if store_preds:
            _append_numeric_rows(preds_file, np.asarray(chunk_result["preds"]).T)

        last_state = chunk_trace[-1]
        model.trace = [last_state]
        del chunk_trace, thinned, chunk_result
        gc.collect()
        remaining -= chunk_post
        chunks_done += 1

    del model, last_state
    gc.collect()
    return {
        "chain_id": chain_id,
        "chain_seed": chain_seed,
        "run_id": run_id,
        "tmp_dir": str(chain_dir),
    }


def assemble_long_default(
    *,
    store_root: Path,
    dataset_tag: str,
    run_id: int,
    split_info: dict[str, Any],
    chain_results: list[dict[str, Any]],
    metadata: dict[str, Any],
    store_preds=True,
):
    for sub in ["preds", "sigmas", "rmses", "metadata"]:
        (store_root / dataset_tag / sub).mkdir(parents=True, exist_ok=True)
    sigmas_by_chain = []
    rmses_by_chain = []
    preds_by_chain = []
    for cr in chain_results:
        chain_dir = Path(cr["tmp_dir"])
        sigmas_rows = _load_2d_or_empty(chain_dir / "sigmas_rows.csv", 1)
        rmses_rows = _load_2d_or_empty(chain_dir / "rmses_rows.csv", 1)
        sigmas_by_chain.append(sigmas_rows.reshape(-1, 1))
        rmses_by_chain.append(rmses_rows.reshape(-1))
        if store_preds:
            preds_rows = _load_2d_or_empty(chain_dir / "preds_rows.csv", split_info["X_test_fixed"].shape[0])
            preds_by_chain.append(preds_rows.T)

    sigmas = np.asarray(sigmas_by_chain)
    rmses = np.asarray(rmses_by_chain)
    _save_numeric_csv(store_root / dataset_tag / "sigmas" / f"{dataset_tag}__run{run_id:03d}__default_long__sigmas.csv", sigmas)
    _save_numeric_csv(store_root / dataset_tag / "rmses" / f"{dataset_tag}__run{run_id:03d}__default_long__rmses.csv", rmses)
    if store_preds:
        preds = np.asarray(preds_by_chain)
        _save_numeric_csv(store_root / dataset_tag / "preds" / f"{dataset_tag}__run{run_id:03d}__default_long__preds.csv", preds)
        metadata = dict(metadata)
        metadata["default_long_preds_shape"] = str(preds.shape)
    metadata = dict(metadata)
    metadata["default_long_sigmas_shape"] = str(sigmas.shape)
    metadata["default_long_rmses_shape"] = str(rmses.shape)
    _write_key_value_csv(store_root / dataset_tag / "metadata" / f"{dataset_tag}__run{run_id:03d}__default_long_metadata.csv", metadata)

    for cr in chain_results:
        shutil.rmtree(cr["tmp_dir"], ignore_errors=True)


# ---------------------------------------------------------------------
# High-level per-dataset orchestration
# ---------------------------------------------------------------------

def run_fixed100_dataset(
    *,
    X,
    y,
    dataset_tag: str,
    store_dir: str | Path,
    n_runs: int,
    n_chains: int,
    n_jobs: int,
    n_trees: int = 100,
    short_ndpost: int = 2000,
    short_nskip: int = 0,
    long_ndpost: int = 1_000_000,
    long_nskip: int = 0,
    long_store_every: int = 100,
    long_chunk_size: int = 10_000,
    n_fixed_test_points: int = 100,
    train_fraction: float = 0.75,
    fixed_test_seed: int = 42,
    base_train_seed: int = 2026,
    base_chain_seed: int = 2024,
    tree_alpha: float = 0.95,
    tree_beta: float = 2.0,
    temperatures=(1.0, 2.0, 3.0, 5.0),
    ladder_target_rate: float = 0.4,
    ladder_max_rounds: int = 10,
    ladder_ndpost: int = 500,
    ladder_nskip: int = 500,
    ladder_repeats: int = 3,
    ladder_search_points: int | None = 1000,
    ladder_random_state: int = 123,
    swap_interval: int = 50,
    post_swap_repair_steps: int = 0,
    multi_tries: int = 10,
    proposal_probs_default=None,
    proposal_probs_mtmh=None,
    store_preds: bool = True,
    progress_print: bool = True,
    run_short: bool = True,
    run_long: bool = True,
    dirichlet_prior: bool = False,
    s_alpha: float = 1.0,
):
    if not run_short and not run_long:
        raise ValueError("At least one of run_short or run_long must be True.")
    if proposal_probs_default is None:
        proposal_probs_default = default_proposal_probs
    if proposal_probs_mtmh is None:
        proposal_probs_mtmh = mtmh_proposal_probs

    store_root = Path(store_dir)
    store_root.mkdir(parents=True, exist_ok=True)
    dataset_root = store_root / dataset_tag
    dataset_root.mkdir(parents=True, exist_ok=True)

    splits = make_fixed100_splits(
        X,
        y,
        n_runs=n_runs,
        n_fixed_test_points=n_fixed_test_points,
        train_fraction=train_fraction,
        fixed_test_seed=fixed_test_seed,
        base_train_seed=base_train_seed,
    )

    _write_key_value_csv(
        dataset_root / "metadata" / f"{dataset_tag}__dataset_metadata.csv",
        {
            "dataset_tag": dataset_tag,
            "n_rows": int(np.asarray(X).shape[0]),
            "n_features": int(np.asarray(X).shape[1]),
            "n_runs": n_runs,
            "n_chains": n_chains,
            "n_jobs": n_jobs,
            "n_fixed_test_points": n_fixed_test_points,
            "train_fraction": train_fraction,
            "fixed_test_seed": fixed_test_seed,
            "base_train_seed": base_train_seed,
            "short_ndpost": short_ndpost,
            "long_ndpost": long_ndpost,
            "long_store_every": long_store_every,
            "temperatures": list(temperatures),
            "ladder_target_rate": ladder_target_rate,
            "ladder_max_rounds": ladder_max_rounds,
            "ladder_ndpost": ladder_ndpost,
            "ladder_nskip": ladder_nskip,
            "ladder_repeats": ladder_repeats,
            "ladder_search_points": ladder_search_points,
            "swap_interval": swap_interval,
            "multi_tries": multi_tries,
            "run_short": run_short,
            "run_long": run_long,
            "dirichlet_prior": dirichlet_prior,
            "s_alpha": s_alpha,
        },
    )

    for split_info in splits:
        run_id = int(split_info["run_id"])
        X_train = split_info["X_train"]
        y_train = split_info["y_train"]
        X_test_fixed = split_info["X_test_fixed"]
        y_test_fixed = split_info["y_test_fixed"]
        if progress_print:
            print(f"[{dataset_tag} RUN {run_id:03d}] fixed test n={X_test_fixed.shape[0]}, train n={X_train.shape[0]}", flush=True)

        if run_short:
            if ladder_search_points is not None and ladder_search_points < X_train.shape[0]:
                rng = np.random.default_rng(split_info["train_seed"])
                idx_search = rng.choice(X_train.shape[0], ladder_search_points, replace=False)
                X_search = X_train[idx_search]
                y_search = y_train[idx_search]
            else:
                X_search = X_train
                y_search = y_train

            run_temperatures, ladder_mean_rates, ladder_history = quick_ladder_search(
                X_search,
                y_search,
                n_trees=n_trees,
                tree_alpha=tree_alpha,
                tree_beta=tree_beta,
                proposal_probs=proposal_probs_default,
                target_rate=ladder_target_rate,
                max_rounds=ladder_max_rounds,
                ndpost=ladder_ndpost,
                nskip=ladder_nskip,
                n_repeats=ladder_repeats,
                random_state=ladder_random_state + 10000 * run_id,
                swap_interval=swap_interval,
                post_swap_repair_steps=post_swap_repair_steps,
                initial_temperatures=temperatures,
                progress_print=progress_print,
                progress_prefix=f"[{dataset_tag} RUN {run_id:03d}] ",
            )

            if progress_print:
                print(f"[{dataset_tag} RUN {run_id:03d}] ladder selected temps={np.round(run_temperatures, 4).tolist()}", flush=True)
                print(f"[{dataset_tag} RUN {run_id:03d}] short methods start: n_chains={n_chains}, n_jobs={n_jobs}", flush=True)

            short_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(run_short_chain)(
                    chain_id=chain_id,
                    chain_seed=base_chain_seed + run_id * n_chains + chain_id,
                    X_train=X_train,
                    y_train=y_train,
                    X_test_fixed=X_test_fixed,
                    y_test_fixed=y_test_fixed,
                    ndpost=short_ndpost,
                    nskip=short_nskip,
                    n_trees=n_trees,
                    tree_alpha=tree_alpha,
                    tree_beta=tree_beta,
                    proposal_probs_default=proposal_probs_default,
                    proposal_probs_mtmh=proposal_probs_mtmh,
                    temperatures=run_temperatures,
                    swap_interval=swap_interval,
                    post_swap_repair_steps=post_swap_repair_steps,
                    multi_tries=multi_tries,
                    dirichlet_prior=dirichlet_prior,
                    s_alpha=s_alpha,
                    store_preds=store_preds,
                )
                for chain_id in range(n_chains)
            )
            save_short_run(
                store_root=store_root,
                dataset_tag=dataset_tag,
                run_id=run_id,
                split_info=split_info,
                chain_results=short_results,
                metadata={
                    "phase": "short_methods",
                    "run_id": run_id,
                    "train_seed": split_info["train_seed"],
                    "short_ndpost": short_ndpost,
                    "short_nskip": short_nskip,
                    "n_chains": n_chains,
                    "n_jobs": n_jobs,
                    "methods": METHODS_SHORT,
                    "temperatures": run_temperatures,
                    "ladder_mean_rates": ladder_mean_rates,
                    "ladder_history": ladder_history,
                    "ladder_search_points": int(X_search.shape[0]),
                    "dirichlet_prior": dirichlet_prior,
                    "s_alpha": s_alpha,
                },
            )
            del short_results
            gc.collect()
        elif progress_print:
            print(f"[{dataset_tag} RUN {run_id:03d}] skip short methods", flush=True)

        if not run_long:
            if progress_print:
                print(f"[{dataset_tag} RUN {run_id:03d}] skip default_long", flush=True)
            continue

        if progress_print:
            print(f"[{dataset_tag} RUN {run_id:03d}] default_long start: ndpost={long_ndpost}, store_every={long_store_every}", flush=True)
        tmp_root = dataset_root / "_stream_tmp" / f"run{run_id:03d}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        long_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_long_default_chain_streaming)(
                chain_id=chain_id,
                chain_seed=base_chain_seed + 100000 + run_id * n_chains + chain_id,
                run_id=run_id,
                X_train=X_train,
                y_train=y_train,
                X_test_fixed=X_test_fixed,
                y_test_fixed=y_test_fixed,
                ndpost=long_ndpost,
                nskip=long_nskip,
                n_trees=n_trees,
                proposal_probs_default=proposal_probs_default,
                store_every=long_store_every,
                chunk_size=long_chunk_size,
                tmp_dir=tmp_root,
                store_preds=store_preds,
                dirichlet_prior=dirichlet_prior,
                s_alpha=s_alpha,
            )
            for chain_id in range(n_chains)
        )
        assemble_long_default(
            store_root=store_root,
            dataset_tag=dataset_tag,
            run_id=run_id,
            split_info=split_info,
            chain_results=long_results,
            metadata={
                "phase": "default_long",
                "run_id": run_id,
                "train_seed": split_info["train_seed"],
                "long_ndpost": long_ndpost,
                "long_nskip": long_nskip,
                "long_store_every": long_store_every,
                "long_chunk_size": long_chunk_size,
                "n_chains": n_chains,
                "n_jobs": n_jobs,
                "dirichlet_prior": dirichlet_prior,
                "s_alpha": s_alpha,
            },
            store_preds=store_preds,
        )
        shutil.rmtree(tmp_root, ignore_errors=True)
        del long_results
        gc.collect()
        if progress_print:
            print(f"[{dataset_tag} RUN {run_id:03d}] done", flush=True)
