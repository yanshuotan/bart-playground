from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from bart_playground import DefaultBART
from bart_playground.samplers import default_proposal_probs



def _summarize_model_outputs(model, X_test, y_test, *, store_preds: bool = False):
    preds = model.posterior_f(X_test, backtransform=True)
    traces = model.trace
    sigmas = [trace.global_params["eps_sigma2"] for trace in traces]
    rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]

    result = {
        "sigmas": np.array(sigmas),
        "rmses": np.array(rmses),
    }

    if store_preds:
        result["preds"] = np.array(preds)

    return result, preds


def _thin_trace(trace, *, ndpost: int, nskip: int, store_every: int):
    if store_every < 1:
        raise ValueError("store_every must be >= 1")

    trimmed = list(trace)
    if nskip == 0 and len(trimmed) == ndpost + 1:
        trimmed = trimmed[1:]
    elif len(trimmed) > ndpost:
        trimmed = trimmed[-ndpost:]

    return trimmed[::store_every]


def _make_setting_tag(notebook):
    return str(notebook).replace(" ", "_").replace("/", "_").replace("\\", "_")


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

    np.savetxt(
        file_path,
        arr2d,
        delimiter=",",
        fmt="%.10g",
        header=f"original_shape={original_shape}",
    )


def _save_run_results_to_csv(run_result, store_root: Path, setting_tag: str):
    run_id = run_result["run_id"]

    data_names = ["sigmas", "rmses"]

    for data_name in data_names:
        (store_root / data_name).mkdir(parents=True, exist_ok=True)

    if "preds" in run_result["default"]:
        (store_root / "preds").mkdir(parents=True, exist_ok=True)
    (store_root / "subsample_X_test").mkdir(parents=True, exist_ok=True)
    (store_root / "subsample_y_test").mkdir(parents=True, exist_ok=True)

    model_name = "default"
    for data_name in data_names:
        file_name = f"{setting_tag}__run{run_id:03d}__{model_name}__{data_name}.csv"
        file_path = store_root / data_name / file_name
        values = run_result[model_name][data_name]
        _save_numeric_csv(file_path, values)

    if "preds" in run_result[model_name]:
        preds_name = f"{setting_tag}__run{run_id:03d}__{model_name}__preds.csv"
        _save_numeric_csv(store_root / "preds" / preds_name, run_result[model_name]["preds"])

    if "subsample" in run_result:
        x_sub_name = f"{setting_tag}__run{run_id:03d}__subsample_X_test.csv"
        y_sub_name = f"{setting_tag}__run{run_id:03d}__subsample_y_test.csv"
        _save_numeric_csv(store_root / "subsample_X_test" / x_sub_name, run_result["subsample"]["X_test"])
        _save_numeric_csv(store_root / "subsample_y_test" / y_sub_name, run_result["subsample"]["y_test"])

def run_chain(
    *,
    chain_id,
    chain_seed,
    split_seed,
    X_train,
    X_test,
    y_train,
    y_test,
    ndpost,
    nskip,
    n_trees,
    tree_alpha,
    tree_beta,
    proposal_probs_default,
    store_preds=True,
    n_test_points=None,
    test_point_seed=42,
    store_every=1,
    chunk_size=10000,
):
    """Run DefaultBART on the same split with chunked sampling to limit memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    first_chunk = min(chunk_size, ndpost)
    bart_default = DefaultBART(
        ndpost=first_chunk,
        nskip=nskip,
        n_trees=n_trees,
        tol=1,
        proposal_probs=proposal_probs_default,
        random_state=chain_seed,
    )
    bart_default.fit(X_train, y_train)

    collected_trace = []
    thinned_chunk = _thin_trace(
        bart_default.trace,
        ndpost=first_chunk,
        nskip=nskip,
        store_every=store_every,
    )
    collected_trace.extend(thinned_chunk)

    last_state = bart_default.trace[-1]
    bart_default.trace = [last_state]

    remaining = ndpost - first_chunk
    while remaining > 0:
        chunk_post = min(chunk_size, remaining)
        chunk_trace = bart_default.sampler.continue_run(
            chunk_post,
            last_state=last_state,
            quietly=True,
        )
        thinned_chunk = _thin_trace(
            chunk_trace,
            ndpost=chunk_post,
            nskip=0,
            store_every=store_every,
        )
        collected_trace.extend(thinned_chunk)
        last_state = chunk_trace[-1]
        bart_default.trace = [last_state]
        remaining -= chunk_post

    bart_default.trace = collected_trace
    bart_default.ndpost = len(collected_trace)
    bart_default.nskip = 0

    default_result, default_pred_all_test = _summarize_model_outputs(
        bart_default,
        X_test,
        y_test,
        store_preds=False,
    )
    del bart_default
    gc.collect()

    if n_test_points is not None and n_test_points < X_test.shape[0]:
        rng = np.random.default_rng(test_point_seed)
        idx = rng.choice(X_test.shape[0], n_test_points, replace=False)
    else:
        idx = np.arange(X_test.shape[0])

    X_test_subsample = np.array(X_test[idx])
    y_test_subsample = np.array(y_test[idx])

    default_pred_subsample = default_pred_all_test[idx, :]

    result = {
        "chain_id": chain_id,
        "chain_seed": chain_seed,
        "split_seed": split_seed,
        "subsample": {
            "X_test": X_test_subsample,
            "y_test": y_test_subsample,
        },
        "default": default_result,
    }

    if store_preds:
        result["default"]["preds"] = np.array(default_pred_subsample)

    return result


def _stack_run_results(
    *,
    chain_results,
    run_id,
    split_seed,
    n_chains,
    n_trees,
    tree_alpha,
    tree_beta,
):
    run_result = {
        "run_id": run_id,
        "split_seed": split_seed,
        "chains": chain_results,
        "subsample": {
            "X_test": np.array(chain_results[0]["subsample"]["X_test"]),
            "y_test": np.array(chain_results[0]["subsample"]["y_test"]),
        },
        "default": {
            "sigmas": np.array([r["default"]["sigmas"] for r in chain_results]),
            "rmses": np.array([r["default"]["rmses"] for r in chain_results]),
        },
    }

    if "preds" in chain_results[0]["default"]:
        run_result["default"]["preds"] = np.array([r["default"]["preds"] for r in chain_results])

    return run_result


def run_parallel_experiments(
    X,
    y,
    ndpost,
    nskip,
    n_trees,
    notebook,
    tree_alpha=0.95,
    tree_beta=2.0,
    n_runs=5,
    n_chains=5,
    n_jobs=-1,
    base_split_seed=42,
    base_chain_seed=2024,
    run_start_id=0,
    store_preds=True,
    n_test_points=100,
    proposal_probs_default=None,
    store_every=1,
    chunk_size=10000,
    store_dir="store",
    progress_print: bool = True,
):
    """Run experiments grouped by run and chain for DefaultBART.

    - Different runs use different train/test splits (via split_seed).
    - Within a run, chains share the split and differ only by random seed.
    - Outputs are saved as categorized CSV files under store_dir.
    """
    if proposal_probs_default is None:
        proposal_probs_default = default_proposal_probs

    store_root = Path(store_dir)
    store_root.mkdir(parents=True, exist_ok=True)

    setting_tag = _make_setting_tag(notebook=notebook)
    all_run_results = []

    for run_offset in range(n_runs):
        run_id = run_start_id + run_offset
        split_seed = base_split_seed + run_id

        if progress_print:
            print(
                f"[RUN {run_id}] start ({run_offset + 1}/{n_runs}) | split_seed={split_seed}",
                flush=True,
            )

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=split_seed)

        if progress_print:
            print(f"[RUN {run_id}] chains start: n_chains={n_chains}", flush=True)

        chain_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_chain)(
                chain_id=chain_id,
                chain_seed=base_chain_seed + run_id * n_chains + chain_id,
                split_seed=split_seed,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                ndpost=ndpost,
                nskip=nskip,
                n_trees=n_trees,
                tree_alpha=tree_alpha,
                tree_beta=tree_beta,
                proposal_probs_default=proposal_probs_default,
                store_preds=store_preds,
                n_test_points=n_test_points,
                test_point_seed=split_seed,
                store_every=store_every,
                chunk_size=chunk_size,
            )
            for chain_id in range(n_chains)
        )

        run_result = _stack_run_results(
            chain_results=chain_results,
            run_id=run_id,
            split_seed=split_seed,
            n_chains=n_chains,
            n_trees=n_trees,
            tree_alpha=tree_alpha,
            tree_beta=tree_beta,
        )

        _save_run_results_to_csv(run_result, store_root=store_root, setting_tag=setting_tag)
        all_run_results.append(run_result)

        if progress_print:
            print(f"[RUN {run_id}] done and saved to {store_root}", flush=True)

    return all_run_results
