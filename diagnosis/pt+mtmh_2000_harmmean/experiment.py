from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from bart_playground import DefaultBART, MultiBART, ParallelTemperingBART
from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs


def count_leaves_in_trees(trace_record):
    """Count leaves (vars == -1) in all trees of a single trace record and return average."""
    total_leaves = 0
    total_trees = len(trace_record.trees)

    for tree in trace_record.trees:
        leaves_count = np.sum(np.array(tree.vars) == -1)
        total_leaves += leaves_count

    return total_leaves / total_trees


def calculate_tree_depth(tree):
    """Calculate tree depth as ceil(log2(position_of_last_-1)) - 1."""
    vars_array = np.array(tree.vars)
    leaf_positions = np.where(vars_array == -1)[0]
    last_leaf_position = leaf_positions[-1]
    depth = int(np.ceil(np.log2(last_leaf_position + 2))) - 1
    return depth


def calculate_avg_depth_per_trace(trace_record):
    """Calculate average tree depth for all trees in a trace record."""
    total_depth = 0
    total_trees = len(trace_record.trees)

    for tree in trace_record.trees:
        tree_depth = calculate_tree_depth(tree)
        total_depth += tree_depth

    return total_depth / total_trees


def crps_from_samples(samples, y_true):
    """Compute CRPS per point from posterior samples.

    samples: (n_points, n_samples)
    y_true:  (n_points,)
    """
    n_points, n_samples = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    samples_sorted = np.sort(samples, axis=1)
    k = np.arange(1, n_samples + 1)
    coeffs = (2 * k - n_samples - 1)[None, :]
    term2 = np.sum(coeffs * samples_sorted, axis=1) / (n_samples**2)
    return term1 - term2


def _summarize_model_outputs(model, X_test, y_test, *, store_preds: bool = False):
    preds = model.posterior_f(X_test, backtransform=True)
    traces = model.trace
    sigmas = [trace.global_params["eps_sigma2"] for trace in traces]
    rmses = [root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])]
    leaves = [count_leaves_in_trees(trace) for trace in traces]
    depths = [calculate_avg_depth_per_trace(trace) for trace in traces]
    accepted_moves = np.array(model.sampler.accepted_moves_logmh, dtype=object)

    pred_all_test = model.posterior_predict(X_test)
    lower = np.percentile(pred_all_test, 2.5, axis=1)
    upper = np.percentile(pred_all_test, 97.5, axis=1)
    covered_bool = (y_test >= lower) & (y_test <= upper)

    result = {
        "sigmas": np.array(sigmas),
        "rmses": np.array(rmses),
        "leaves": np.array(leaves),
        "depths": np.array(depths),
        "accepted_moves_logmh": accepted_moves,
        "subsample_rmse": None,
        "subsample_crps": None,
    }

    swap_accept_rates = np.asarray(model.get_params().get("swap_accept_rates", []), dtype=float)
    if swap_accept_rates.size > 0:
        result["swap_accept_rates"] = swap_accept_rates

    if store_preds:
        result["preds"] = np.array(preds)
        result["coverage"] = np.array(covered_bool)

    return result, pred_all_test


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
):
    """Adaptive temperature-ladder search with repeated fits per round.

    For each round, run PT multiple times and average adjacent swap rates.
    Split intervals whose mean swap rate is below target_rate.
    """
    temps = sorted({float(t) for t in initial_temperatures})
    if not temps:
        raise ValueError("initial_temperatures cannot be empty")
    if temps[0] != 1.0:
        temps = [1.0] + [t for t in temps if t != 1.0]

    history = []
    final_mean_rates = np.array([], dtype=float)

    if progress_print:
        print(
            f"[LADDER] start: n_points={X.shape[0]}, rounds<={max_rounds}, repeats={n_repeats}, "
            f"target_rate={target_rate}, init_temps={temps}",
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

        if round_rates:
            mean_rates = np.mean(np.vstack(round_rates), axis=0)
        else:
            mean_rates = np.array([], dtype=float)

        if progress_print:
            rates_preview = np.round(mean_rates, 4).tolist() if mean_rates.size > 0 else []
            print(
                f"[LADDER] round {round_id + 1}/{max_rounds}: n_temps={len(temps)}, mean_rates={rates_preview}",
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
                    f"[LADDER] stop: target reached or no valid rates. final_temps={temps}",
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
                print("[LADDER] stop: no new temperature inserted.", flush=True)
            break
        temps = updated_temps

    if progress_print:
        print(f"[LADDER] done: final n_temps={len(temps)}", flush=True)

    return [float(t) for t in temps], final_mean_rates.tolist(), history


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


def _save_object_csv(file_path: Path, values):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "value_json"])
        for idx, value in enumerate(values):
            writer.writerow([idx, json.dumps(_as_serializable(value), ensure_ascii=True)])


def _save_run_results_to_csv(run_result, store_root: Path, setting_tag: str):
    run_id = run_result["run_id"]

    model_data_keys = {
        "default": [
            "sigmas",
            "rmses",
            "leaves",
            "depths",
            "accepted_moves_logmh",
            "subsample_rmse",
            "subsample_crps",
        ],
        "default_pt": [
            "sigmas",
            "rmses",
            "leaves",
            "depths",
            "accepted_moves_logmh",
            "subsample_rmse",
            "subsample_crps",
            "swap_accept_rates",
        ],
        "mtmh": [
            "sigmas",
            "rmses",
            "leaves",
            "depths",
            "accepted_moves_logmh",
            "subsample_rmse",
            "subsample_crps",
        ],
        "mtmh_pt": [
            "sigmas",
            "rmses",
            "leaves",
            "depths",
            "accepted_moves_logmh",
            "subsample_rmse",
            "subsample_crps",
            "swap_accept_rates",
        ],
    }

    for data_name in sorted({k for keys in model_data_keys.values() for k in keys}):
        (store_root / data_name).mkdir(parents=True, exist_ok=True)

    model_names = ["default", "default_pt", "mtmh", "mtmh_pt"]

    if "preds" in run_result["default"] and "coverage" in run_result["default"]:
        (store_root / "preds").mkdir(parents=True, exist_ok=True)
        (store_root / "coverage").mkdir(parents=True, exist_ok=True)

    metadata_dir = store_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (store_root / "subsample_X_test").mkdir(parents=True, exist_ok=True)
    (store_root / "subsample_y_test").mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        for data_name in model_data_keys[model_name]:
            file_name = f"{setting_tag}__run{run_id:03d}__{model_name}__{data_name}.csv"
            file_path = store_root / data_name / file_name
            values = run_result[model_name][data_name]

            if data_name == "accepted_moves_logmh":
                _save_object_csv(file_path, values)
            else:
                _save_numeric_csv(file_path, values)

        if "preds" in run_result[model_name]:
            preds_name = f"{setting_tag}__run{run_id:03d}__{model_name}__preds.csv"
            coverage_name = f"{setting_tag}__run{run_id:03d}__{model_name}__coverage.csv"
            _save_numeric_csv(store_root / "preds" / preds_name, run_result[model_name]["preds"])
            _save_numeric_csv(store_root / "coverage" / coverage_name, run_result[model_name]["coverage"])

    if "subsample" in run_result:
        x_sub_name = f"{setting_tag}__run{run_id:03d}__subsample_X_test.csv"
        y_sub_name = f"{setting_tag}__run{run_id:03d}__subsample_y_test.csv"
        _save_numeric_csv(store_root / "subsample_X_test" / x_sub_name, run_result["subsample"]["X_test"])
        _save_numeric_csv(store_root / "subsample_y_test" / y_sub_name, run_result["subsample"]["y_test"])

    metadata_file = metadata_dir / f"{setting_tag}__run{run_id:03d}.csv"
    with open(metadata_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["run_id", run_id])
        writer.writerow(["split_seed", run_result["split_seed"]])
        for key, value in run_result["metadata"].items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(_as_serializable(value), ensure_ascii=True)
            writer.writerow([key, value])


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
    temperatures,
    swap_interval,
    post_swap_repair_steps,
    store_preds=False,
    n_test_points=None,
    test_point_seed=42,
    multi_tries=10,
    proposal_probs_mtmh=None,
):
    """Run default, default_pt, mtmh, and mtmh+pt on the same split."""
    if proposal_probs_mtmh is None:
        proposal_probs_mtmh = mtmh_proposal_probs

    bart_default = DefaultBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        tol=1,
        proposal_probs=proposal_probs_default,
        random_state=chain_seed,
    )
    bart_default.fit(X_train, y_train)
    default_result, default_pred_all_test = _summarize_model_outputs(
        bart_default,
        X_test,
        y_test,
        store_preds=False,
    )
    del bart_default
    gc.collect()

    bart_default_pt = ParallelTemperingBART(
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
    )
    bart_default_pt.fit(X_train, y_train)
    default_pt_result, default_pt_pred_all_test = _summarize_model_outputs(
        bart_default_pt,
        X_test,
        y_test,
        store_preds=False,
    )
    del bart_default_pt
    gc.collect()

    bart_mtmh = MultiBART(
        ndpost=ndpost,
        nskip=nskip,
        n_trees=n_trees,
        tree_alpha=tree_alpha,
        tree_beta=tree_beta,
        tol=1,
        proposal_probs=proposal_probs_mtmh,
        random_state=chain_seed,
        multi_tries=multi_tries,
    )
    bart_mtmh.fit(X_train, y_train)
    mtmh_result, mtmh_pred_all_test = _summarize_model_outputs(
        bart_mtmh,
        X_test,
        y_test,
        store_preds=False,
    )
    del bart_mtmh
    gc.collect()

    bart_mtmh_pt = ParallelTemperingBART(
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
    )
    bart_mtmh_pt.fit(X_train, y_train)
    mtmh_pt_result, mtmh_pt_pred_all_test = _summarize_model_outputs(
        bart_mtmh_pt,
        X_test,
        y_test,
        store_preds=False,
    )
    del bart_mtmh_pt
    gc.collect()

    if n_test_points is not None and n_test_points < X_test.shape[0]:
        rng = np.random.default_rng(test_point_seed)
        idx = rng.choice(X_test.shape[0], n_test_points, replace=False)
    else:
        idx = np.arange(X_test.shape[0])

    X_test_subsample = np.array(X_test[idx])
    y_test_subsample = np.array(y_test[idx])

    default_pred_subsample = default_pred_all_test[idx, :]
    default_pt_pred_subsample = default_pt_pred_all_test[idx, :]
    mtmh_pred_subsample = mtmh_pred_all_test[idx, :]
    mtmh_pt_pred_subsample = mtmh_pt_pred_all_test[idx, :]

    default_subsample_rmse = root_mean_squared_error(y_test_subsample, np.mean(default_pred_subsample, axis=1))
    default_pt_subsample_rmse = root_mean_squared_error(y_test_subsample, np.mean(default_pt_pred_subsample, axis=1))
    mtmh_subsample_rmse = root_mean_squared_error(y_test_subsample, np.mean(mtmh_pred_subsample, axis=1))
    mtmh_pt_subsample_rmse = root_mean_squared_error(y_test_subsample, np.mean(mtmh_pt_pred_subsample, axis=1))
    default_subsample_crps = float(np.mean(crps_from_samples(default_pred_subsample, y_test_subsample)))
    default_pt_subsample_crps = float(np.mean(crps_from_samples(default_pt_pred_subsample, y_test_subsample)))
    mtmh_subsample_crps = float(np.mean(crps_from_samples(mtmh_pred_subsample, y_test_subsample)))
    mtmh_pt_subsample_crps = float(np.mean(crps_from_samples(mtmh_pt_pred_subsample, y_test_subsample)))

    result = {
        "chain_id": chain_id,
        "chain_seed": chain_seed,
        "split_seed": split_seed,
        "subsample": {
            "X_test": X_test_subsample,
            "y_test": y_test_subsample,
        },
        "default": default_result,
        "default_pt": default_pt_result,
        "mtmh": mtmh_result,
        "mtmh_pt": mtmh_pt_result,
    }

    result["default"]["subsample_rmse"] = default_subsample_rmse
    result["default"]["subsample_crps"] = default_subsample_crps
    result["default_pt"]["subsample_rmse"] = default_pt_subsample_rmse
    result["default_pt"]["subsample_crps"] = default_pt_subsample_crps
    result["mtmh"]["subsample_rmse"] = mtmh_subsample_rmse
    result["mtmh"]["subsample_crps"] = mtmh_subsample_crps
    result["mtmh_pt"]["subsample_rmse"] = mtmh_pt_subsample_rmse
    result["mtmh_pt"]["subsample_crps"] = mtmh_pt_subsample_crps

    if store_preds:
        result["default"]["preds"] = np.array(default_pred_subsample)
        result["default"]["coverage"] = np.array((y_test >= np.percentile(default_pred_all_test, 2.5, axis=1)) & (y_test <= np.percentile(default_pred_all_test, 97.5, axis=1)))
        result["default_pt"]["preds"] = np.array(default_pt_pred_subsample)
        result["default_pt"]["coverage"] = np.array((y_test >= np.percentile(default_pt_pred_all_test, 2.5, axis=1)) & (y_test <= np.percentile(default_pt_pred_all_test, 97.5, axis=1)))
        result["mtmh"]["preds"] = np.array(mtmh_pred_subsample)
        result["mtmh"]["coverage"] = np.array((y_test >= np.percentile(mtmh_pred_all_test, 2.5, axis=1)) & (y_test <= np.percentile(mtmh_pred_all_test, 97.5, axis=1)))
        result["mtmh_pt"]["preds"] = np.array(mtmh_pt_pred_subsample)
        result["mtmh_pt"]["coverage"] = np.array((y_test >= np.percentile(mtmh_pt_pred_all_test, 2.5, axis=1)) & (y_test <= np.percentile(mtmh_pt_pred_all_test, 97.5, axis=1)))

    return result


def _stack_run_results(
    *,
    chain_results,
    run_id,
    split_seed,
    n_chains,
    ndpost,
    nskip,
    n_trees,
    tree_alpha,
    tree_beta,
    temperatures,
    post_swap_repair_steps,
    target_rate,
    ladder_history,
    ladder_mean_rates,
    ladder_search_points,
    proposal_probs_default,
    proposal_probs_mtmh,
    multi_tries,
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
            "leaves": np.array([r["default"]["leaves"] for r in chain_results]),
            "depths": np.array([r["default"]["depths"] for r in chain_results]),
            "accepted_moves_logmh": np.array(
                [r["default"]["accepted_moves_logmh"] for r in chain_results],
                dtype=object,
            ),
            "subsample_rmse": np.array([r["default"]["subsample_rmse"] for r in chain_results]),
            "subsample_crps": np.array([r["default"]["subsample_crps"] for r in chain_results]),
        },
        "default_pt": {
            "sigmas": np.array([r["default_pt"]["sigmas"] for r in chain_results]),
            "rmses": np.array([r["default_pt"]["rmses"] for r in chain_results]),
            "leaves": np.array([r["default_pt"]["leaves"] for r in chain_results]),
            "depths": np.array([r["default_pt"]["depths"] for r in chain_results]),
            "accepted_moves_logmh": np.array(
                [r["default_pt"]["accepted_moves_logmh"] for r in chain_results],
                dtype=object,
            ),
            "subsample_rmse": np.array([r["default_pt"]["subsample_rmse"] for r in chain_results]),
            "subsample_crps": np.array([r["default_pt"]["subsample_crps"] for r in chain_results]),
            "swap_accept_rates": np.array([r.get("default_pt", {}).get("swap_accept_rates", []) for r in chain_results]),
        },
        "mtmh": {
            "sigmas": np.array([r["mtmh"]["sigmas"] for r in chain_results]),
            "rmses": np.array([r["mtmh"]["rmses"] for r in chain_results]),
            "leaves": np.array([r["mtmh"]["leaves"] for r in chain_results]),
            "depths": np.array([r["mtmh"]["depths"] for r in chain_results]),
            "accepted_moves_logmh": np.array(
                [r["mtmh"]["accepted_moves_logmh"] for r in chain_results],
                dtype=object,
            ),
            "subsample_rmse": np.array([r["mtmh"]["subsample_rmse"] for r in chain_results]),
            "subsample_crps": np.array([r["mtmh"]["subsample_crps"] for r in chain_results]),
        },
        "mtmh_pt": {
            "sigmas": np.array([r["mtmh_pt"]["sigmas"] for r in chain_results]),
            "rmses": np.array([r["mtmh_pt"]["rmses"] for r in chain_results]),
            "leaves": np.array([r["mtmh_pt"]["leaves"] for r in chain_results]),
            "depths": np.array([r["mtmh_pt"]["depths"] for r in chain_results]),
            "accepted_moves_logmh": np.array(
                [r["mtmh_pt"]["accepted_moves_logmh"] for r in chain_results],
                dtype=object,
            ),
            "subsample_rmse": np.array([r["mtmh_pt"]["subsample_rmse"] for r in chain_results]),
            "subsample_crps": np.array([r["mtmh_pt"]["subsample_crps"] for r in chain_results]),
            "swap_accept_rates": np.array([r.get("mtmh_pt", {}).get("swap_accept_rates", []) for r in chain_results]),
        },
        "metadata": {
            "n_chains": n_chains,
            "ndpost": ndpost,
            "nskip": nskip,
            "n_trees": n_trees,
            "tree_alpha": tree_alpha,
            "tree_beta": tree_beta,
            "target_rate": target_rate,
            "temperatures": list(temperatures),
            "post_swap_repair_steps": post_swap_repair_steps,
            "ladder_mean_rates": list(ladder_mean_rates),
            "ladder_history": ladder_history,
            "ladder_search_points": ladder_search_points,
            "proposal_probs_default": proposal_probs_default,
            "proposal_probs_pt": proposal_probs_default,
            "proposal_probs_mtmh": proposal_probs_mtmh,
            "proposal_probs_mtmh_pt": proposal_probs_mtmh,
            "multi_tries": multi_tries,
            "n_subsample_points": int(np.array(chain_results[0]["subsample"]["y_test"]).shape[0]),
        },
    }

    if "preds" in chain_results[0]["default"]:
        run_result["default"]["preds"] = np.array([r["default"]["preds"] for r in chain_results])
        run_result["default"]["coverage"] = np.array([r["default"]["coverage"] for r in chain_results])
        run_result["default_pt"]["preds"] = np.array([r["default_pt"]["preds"] for r in chain_results])
        run_result["default_pt"]["coverage"] = np.array([r["default_pt"]["coverage"] for r in chain_results])
        run_result["mtmh"]["preds"] = np.array([r["mtmh"]["preds"] for r in chain_results])
        run_result["mtmh"]["coverage"] = np.array([r["mtmh"]["coverage"] for r in chain_results])
        run_result["mtmh_pt"]["preds"] = np.array([r["mtmh_pt"]["preds"] for r in chain_results])
        run_result["mtmh_pt"]["coverage"] = np.array([r["mtmh_pt"]["coverage"] for r in chain_results])

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
    n_chains=4,
    n_jobs=-1,
    base_split_seed=42,
    base_chain_seed=2024,
    run_start_id=0,
    store_preds=False,
    n_test_points=100,
    proposal_probs_default=None,
    proposal_probs_mtmh=None,
    multi_tries=10,
    swap_interval=5,
    post_swap_repair_steps=0,
    # quick ladder search controls
    ladder_target_rate=0.4,
    ladder_max_rounds=10,
    ladder_ndpost=500,
    ladder_nskip=500,
    ladder_repeats=3,
    ladder_search_points=1000,
    ladder_random_state=123,
    ladder_initial_temperatures=(1.0, 3.0),
    store_dir="store",
    progress_print: bool = True,
):
    """Run experiments grouped by run and chain for DefaultBART and PT-BART.

    - Different runs use different train/test splits (via split_seed).
    - Within a run, chains share the split and differ only by random seed.
    - Before each run's PT chains, quick_ladder_search is executed once.
    - Outputs are saved as categorized CSV files under store_dir.
    """
    if proposal_probs_default is None:
        proposal_probs_default = default_proposal_probs

    if proposal_probs_mtmh is None:
        proposal_probs_mtmh = mtmh_proposal_probs

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

        if ladder_search_points is not None and ladder_search_points < X_train.shape[0]:
            rng = np.random.default_rng(split_seed)
            idx_search = rng.choice(X_train.shape[0], ladder_search_points, replace=False)
            X_search = X_train[idx_search]
            y_search = y_train[idx_search]
        else:
            X_search = X_train
            y_search = y_train

        search_temps, search_rates, ladder_history = quick_ladder_search(
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
            initial_temperatures=ladder_initial_temperatures,
            progress_print=progress_print,
        )

        if progress_print:
            print(
                f"[RUN {run_id}] ladder selected temps={np.round(search_temps, 4).tolist()}",
                flush=True,
            )
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
                proposal_probs_mtmh=proposal_probs_mtmh,
                multi_tries=multi_tries,
                temperatures=search_temps,
                swap_interval=swap_interval,
                post_swap_repair_steps=post_swap_repair_steps,
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
            tree_alpha=tree_alpha,
            tree_beta=tree_beta,
            temperatures=search_temps,
            post_swap_repair_steps=post_swap_repair_steps,
            target_rate=ladder_target_rate,
            ladder_history=ladder_history,
            ladder_mean_rates=search_rates,
            ladder_search_points=X_search.shape[0],
            proposal_probs_default=proposal_probs_default,
            proposal_probs_mtmh=proposal_probs_mtmh,
            multi_tries=multi_tries,
        )

        _save_run_results_to_csv(run_result, store_root=store_root, setting_tag=setting_tag)
        all_run_results.append(run_result)

        if progress_print:
            print(f"[RUN {run_id}] done and saved to {store_root}", flush=True)

    return all_run_results
