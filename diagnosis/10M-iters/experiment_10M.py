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
from sklearn.model_selection import train_test_split

from bart_playground import DefaultBART
from bart_playground.samplers import default_proposal_probs


# ---------------------------------------------------------------------
# Small utilities copied/kept compatible with the original 1M-iters runner
# ---------------------------------------------------------------------

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

    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        file_path,
        arr2d,
        delimiter=",",
        fmt="%.10g",
        header=f"original_shape={original_shape}",
    )


def _thin_trace(trace, *, ndpost: int, nskip: int, store_every: int):
    """Keep the same thinning convention as the original runner."""
    if store_every < 1:
        raise ValueError("store_every must be >= 1")

    trimmed = list(trace)
    if nskip == 0 and len(trimmed) == ndpost + 1:
        trimmed = trimmed[1:]
    elif len(trimmed) > ndpost:
        trimmed = trimmed[-ndpost:]

    return trimmed[::store_every]


def _summarize_trace_chunk(model, X_test, y_test, *, store_preds: bool):
    """Compute only the quantities needed for the notebook plots.

    Important: this function assumes model.trace already contains ONLY the
    thinned states for the current chunk.
    """
    preds = model.posterior_f(X_test, backtransform=True)
    sigmas = np.array([trace.global_params["eps_sigma2"] for trace in model.trace])
    rmses = np.array([root_mean_squared_error(y_test, preds[:, k]) for k in range(preds.shape[1])])

    out = {
        "sigmas": sigmas,
        "rmses": rmses,
    }
    if store_preds:
        out["preds"] = np.asarray(preds)
    return out


def _append_numeric_rows(file_path: Path, arr2d: np.ndarray):
    """Append 2D numeric data without header for intermediate per-chain files."""
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
# Streaming per-chain runner
# ---------------------------------------------------------------------

def run_chain(
    *,
    chain_id,
    chain_seed,
    split_seed,
    run_id,
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
    tmp_dir: str | Path,
):
    """Run one DefaultBART chain with true disk streaming.

    This preserves the scientific setup of the original 1M-iters runner but
    avoids accumulating all thinned tree states in RAM. For each chunk:

    1. keep only every `store_every`-th state from that chunk;
    2. compute sigma/rmse/preds for those states;
    3. append those numeric summaries to temporary disk files;
    4. delete the chunk trace and keep only `last_state` for continuation.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if ndpost % store_every != 0:
        raise ValueError("ndpost should be divisible by store_every for clean final shapes")

    # Same subsample logic as original runner.
    if n_test_points is not None and n_test_points < X_test.shape[0]:
        rng = np.random.default_rng(test_point_seed)
        idx = rng.choice(X_test.shape[0], n_test_points, replace=False)
    else:
        idx = np.arange(X_test.shape[0])

    X_test_subsample = np.asarray(X_test[idx])
    y_test_subsample = np.asarray(y_test[idx])

    chain_dir = Path(tmp_dir) / f"run{run_id:03d}_chain{chain_id:03d}"
    if chain_dir.exists():
        shutil.rmtree(chain_dir)
    chain_dir.mkdir(parents=True, exist_ok=True)

    sigmas_file = chain_dir / "sigmas_rows.csv"       # rows: stored states, cols: 1
    rmses_file = chain_dir / "rmses_rows.csv"         # rows: stored states, cols: 1
    preds_file = chain_dir / "preds_rows.csv"         # rows: stored states, cols: n_test_points
    meta_file = chain_dir / "meta.json"

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

    chunks_done = 0
    remaining = ndpost
    last_state = None

    # Process first chunk plus continuation chunks through the same summarize/flush logic.
    while True:
        if chunks_done == 0:
            chunk_trace = bart_default.trace
            chunk_post = first_chunk
            current_nskip = nskip
        else:
            chunk_post = min(chunk_size, remaining)
            chunk_trace = bart_default.sampler.continue_run(
                chunk_post,
                last_state=last_state,
                quietly=True,
            )
            current_nskip = 0

        thinned_chunk = _thin_trace(
            chunk_trace,
            ndpost=chunk_post,
            nskip=current_nskip,
            store_every=store_every,
        )
        if len(thinned_chunk) == 0:
            raise RuntimeError("No thinned states were collected from a chunk")

        # Temporarily expose only the thinned states to the model prediction method.
        bart_default.trace = thinned_chunk
        bart_default.ndpost = len(thinned_chunk)
        bart_default.nskip = 0

        chunk_result = _summarize_trace_chunk(
            bart_default,
            X_test_subsample,
            y_test_subsample,
            store_preds=store_preds,
        )

        _append_numeric_rows(sigmas_file, np.asarray(chunk_result["sigmas"]).reshape(-1, 1))
        _append_numeric_rows(rmses_file, np.asarray(chunk_result["rmses"]).reshape(-1, 1))
        if store_preds:
            # preds is n_test_points x n_stored_chunk; write rows as stored states.
            _append_numeric_rows(preds_file, np.asarray(chunk_result["preds"]).T)

        # Keep only last state for Markov continuation. Drop everything else.
        last_state = chunk_trace[-1]
        bart_default.trace = [last_state]
        del chunk_result, thinned_chunk, chunk_trace
        gc.collect()

        remaining -= chunk_post
        chunks_done += 1
        if remaining <= 0:
            break

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chain_id": chain_id,
                "chain_seed": chain_seed,
                "split_seed": split_seed,
                "run_id": run_id,
                "n_test_points": int(X_test_subsample.shape[0]),
                "expected_stored": int(ndpost // store_every),
                "sigmas_file": str(sigmas_file),
                "rmses_file": str(rmses_file),
                "preds_file": str(preds_file) if store_preds else None,
            },
            f,
            indent=2,
        )

    del bart_default, last_state
    gc.collect()

    # Return only small metadata, not arrays.
    return {
        "chain_id": chain_id,
        "chain_seed": chain_seed,
        "split_seed": split_seed,
        "run_id": run_id,
        "tmp_dir": str(chain_dir),
        "subsample": {
            "X_test": X_test_subsample,
            "y_test": y_test_subsample,
        },
    }


# ---------------------------------------------------------------------
# Final assembly from temporary per-chain files into notebook-compatible CSVs
# ---------------------------------------------------------------------

def _assemble_and_save_run_results(
    *,
    run_result,
    store_root: Path,
    setting_tag: str,
    store_preds: bool,
):
    run_id = run_result["run_id"]
    chain_results = run_result["chains"]

    # Load compact numeric summaries from disk. These are small compared with tree traces.
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
            preds_rows = _load_2d_or_empty(chain_dir / "preds_rows.csv", run_result["subsample"]["X_test"].shape[0])
            # rows: stored states, cols: n_test_points -> original notebook expects n_test_points x stored states
            preds_by_chain.append(preds_rows.T)

    sigmas = np.asarray(sigmas_by_chain)  # (n_chains, n_samples, 1)
    rmses = np.asarray(rmses_by_chain)    # (n_chains, n_samples)

    model_name = "default"
    _save_numeric_csv(
        store_root / "sigmas" / f"{setting_tag}__run{run_id:03d}__{model_name}__sigmas.csv",
        sigmas,
    )
    _save_numeric_csv(
        store_root / "rmses" / f"{setting_tag}__run{run_id:03d}__{model_name}__rmses.csv",
        rmses,
    )
    if store_preds:
        preds = np.asarray(preds_by_chain)  # (n_chains, n_test_points, n_samples)
        _save_numeric_csv(
            store_root / "preds" / f"{setting_tag}__run{run_id:03d}__{model_name}__preds.csv",
            preds,
        )

    _save_numeric_csv(
        store_root / "subsample_X_test" / f"{setting_tag}__run{run_id:03d}__subsample_X_test.csv",
        run_result["subsample"]["X_test"],
    )
    _save_numeric_csv(
        store_root / "subsample_y_test" / f"{setting_tag}__run{run_id:03d}__subsample_y_test.csv",
        run_result["subsample"]["y_test"],
    )

    meta_dir = store_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / f"{setting_tag}__run{run_id:03d}__metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["run_id", run_id])
        writer.writerow(["split_seed", run_result["split_seed"]])
        writer.writerow(["n_chains", len(chain_results)])
        writer.writerow(["sigmas_shape", str(sigmas.shape)])
        writer.writerow(["rmses_shape", str(rmses.shape)])
        if store_preds:
            writer.writerow(["preds_shape", str(np.asarray(preds_by_chain).shape)])

    # Return small shape info only.
    out = {
        "run_id": run_id,
        "split_seed": run_result["split_seed"],
        "subsample": run_result["subsample"],
        "default": {
            "sigmas_shape": sigmas.shape,
            "rmses_shape": rmses.shape,
        },
    }
    if store_preds:
        out["default"]["preds_shape"] = np.asarray(preds_by_chain).shape
    return out


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
    """Disk-streaming long-chain DefaultBART runner.

    Same high-level experimental design as the original 1M-iters runner:
    - different runs use different train/test splits;
    - within a run, chains share the split but differ by seed;
    - output file names and shapes remain notebook-compatible.

    Main difference: per-chunk numeric summaries are flushed to disk so RAM does
    not grow with the number of chunks.
    """
    if proposal_probs_default is None:
        proposal_probs_default = default_proposal_probs

    store_root = Path(store_dir)
    store_root.mkdir(parents=True, exist_ok=True)
    tmp_root = store_root / "_stream_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

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
            print(f"[RUN {run_id}] chains start: n_chains={n_chains}, n_jobs={n_jobs}", flush=True)

        chain_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_chain)(
                chain_id=chain_id,
                chain_seed=base_chain_seed + run_id * n_chains + chain_id,
                split_seed=split_seed,
                run_id=run_id,
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
                tmp_dir=tmp_root,
            )
            for chain_id in range(n_chains)
        )

        run_result = {
            "run_id": run_id,
            "split_seed": split_seed,
            "chains": chain_results,
            "subsample": {
                "X_test": np.asarray(chain_results[0]["subsample"]["X_test"]),
                "y_test": np.asarray(chain_results[0]["subsample"]["y_test"]),
            },
        }

        small_result = _assemble_and_save_run_results(
            run_result=run_result,
            store_root=store_root,
            setting_tag=setting_tag,
            store_preds=store_preds,
        )
        all_run_results.append(small_result)

        # Remove per-chain temp files after final notebook-compatible CSVs are saved.
        for cr in chain_results:
            shutil.rmtree(cr["tmp_dir"], ignore_errors=True)

        if progress_print:
            print(f"[RUN {run_id}] done and saved to {store_root}", flush=True)

    # Remove temp root if empty.
    try:
        tmp_root.rmdir()
    except OSError:
        pass

    return all_run_results
