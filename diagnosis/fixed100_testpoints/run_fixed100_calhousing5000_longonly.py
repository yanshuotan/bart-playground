from __future__ import annotations

import argparse
import csv
import gc
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_california_housing

from bart_playground.samplers import default_proposal_probs
from diagnosis.fixed100_testpoints.experiment_fixed100 import (
    _save_numeric_csv,
    _write_key_value_csv,
    assemble_long_default,
    make_fixed100_splits,
    run_fixed100_dataset,
    run_long_default_chain_streaming,
)


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "calhousing": {
        "dataset_tag": "fixed100_CalHousing_subsample5000",
        "source": "sklearn.fetch_california_housing",
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "subsample_n": 5000,
        "subsample_seed": 42,
    },
}


def _safe_column_names(df) -> list[str]:
    return [str(c) for c in getattr(df, "columns", [])]


def _select_target_and_features(features, targets, *, target_column=None):
    """Return feature DataFrame and numeric 1D target.

    Robust to ucimlrepo datasets where a named target may appear either in
    ds.data.targets or inside ds.data.features.
    """
    X_df = features.copy()
    targets_df = targets.copy() if hasattr(targets, "copy") else pd.DataFrame(targets)

    if isinstance(target_column, str):
        if targets_df is not None and target_column in _safe_column_names(targets_df):
            y = targets_df[target_column].to_numpy()
        elif target_column in _safe_column_names(X_df):
            y = X_df[target_column].to_numpy()
            X_df = X_df.drop(columns=[target_column])
        else:
            raise ValueError(
                f"target_column={target_column!r} not found. "
                f"target columns={_safe_column_names(targets_df)}, feature columns={_safe_column_names(X_df)}"
            )
    elif target_column is not None:
        if targets_df is None or targets_df.shape[1] == 0:
            raise ValueError("Integer target_column requested, but targets are empty.")
        y = targets_df.iloc[:, int(target_column)].to_numpy()
    else:
        if targets_df is None or targets_df.shape[1] == 0:
            raise ValueError("Dataset has no targets in ucimlrepo response. Set target_column from features if needed.")
        if targets_df.shape[1] != 1:
            raise ValueError(
                f"Multiple target columns found {list(targets_df.columns)}. "
                "Please set target_column in DATASET_CONFIGS."
            )
        y = targets_df.iloc[:, 0].to_numpy()

    return X_df, np.asarray(y).reshape(-1).astype(float)


def _preprocess_features(features, *, drop_columns=None, categorical_columns=None):
    """Drop columns, one-hot encode categorical columns, coerce numeric."""
    X_df = features.copy()
    drop_columns = list(drop_columns or [])
    for col in drop_columns:
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    if categorical_columns == "auto":
        cat_cols = list(X_df.select_dtypes(include=["object", "category", "bool"]).columns)
    else:
        cat_cols = [c for c in list(categorical_columns or []) if c in X_df.columns]

    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X = X_df.to_numpy(dtype=float)
    return X, list(X_df.columns), cat_cols


def _clean_finite_rows(X, y):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n_removed = int((~mask).sum())
    return X[mask], y[mask], n_removed


def load_dataset(name: str, *, return_info: bool = False):
    cfg = DATASET_CONFIGS[name]

    if name == "calhousing":
        ds = fetch_california_housing()
        X_full = np.asarray(ds.data, dtype=float)
        y_full = np.asarray(ds.target, dtype=float).reshape(-1)
        feature_names = list(getattr(ds, "feature_names", [f"x{j+1}" for j in range(X_full.shape[1])]))
        subsample_n = int(cfg.get("subsample_n", 5000))
        subsample_seed = int(cfg.get("subsample_seed", 42))
        if subsample_n <= 0 or subsample_n > X_full.shape[0]:
            raise ValueError(f"subsample_n must be in [1, {X_full.shape[0]}], got {subsample_n}")
        rng = np.random.default_rng(subsample_seed)
        idx = np.sort(rng.choice(X_full.shape[0], size=subsample_n, replace=False))
        X = X_full[idx]
        y = y_full[idx]
        X, y, n_removed = _clean_finite_rows(X, y)
        info = {
            "name": name,
            "dataset_tag": cfg["dataset_tag"],
            "source": cfg.get("source"),
            "raw_shape": tuple(X_full.shape),
            "final_shape": tuple(X.shape),
            "subsample_n": subsample_n,
            "subsample_seed": subsample_seed,
            "removed_nonfinite_rows": n_removed,
            "feature_names": feature_names,
            "categorical_columns_encoded": [],
            "dropped_columns": [],
            "target_column": "MedHouseVal",
            "y_mean": float(np.mean(y)),
            "y_std": float(np.std(y)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
        }
        return (X, y, info) if return_info else (X, y)

    if name == "friedman":
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        n_samples = int(cfg.get("n_samples", 2000))
        n_features = int(cfg.get("n_features", 10))
        noise_std = float(cfg.get("noise_std", 1.0))
        X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
        eps = rng.normal(0.0, noise_std, size=n_samples)
        y = (
            10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20.0 * (X[:, 2] - 0.5) ** 2
            + 10.0 * X[:, 3]
            + 5.0 * X[:, 4]
            + eps
        )
        feature_names = [f"x{j+1}" for j in range(n_features)]
        info = {
            "name": name,
            "dataset_tag": cfg["dataset_tag"],
            "raw_shape": (n_samples, n_features),
            "final_shape": (n_samples, n_features),
            "removed_nonfinite_rows": 0,
            "feature_names": feature_names,
            "categorical_columns_encoded": [],
            "target_column": "synthetic_friedman_y",
        }
        return (X.astype(float), np.asarray(y).reshape(-1).astype(float), info) if return_info else (X.astype(float), np.asarray(y).reshape(-1).astype(float))

    ds = fetch_ucirepo(id=cfg["uci_id"])
    raw_features = ds.data.features.copy()
    raw_shape = raw_features.shape
    features, y = _select_target_and_features(raw_features, ds.data.targets, target_column=cfg.get("target_column", None))
    X, feature_names, cat_cols = _preprocess_features(
        features,
        drop_columns=cfg.get("drop_columns", []),
        categorical_columns=cfg.get("categorical_columns", "auto"),
    )
    X, y, n_removed = _clean_finite_rows(X, y)

    info = {
        "name": name,
        "dataset_tag": cfg["dataset_tag"],
        "uci_id": cfg.get("uci_id"),
        "raw_shape": tuple(raw_shape),
        "final_shape": tuple(X.shape),
        "removed_nonfinite_rows": n_removed,
        "feature_names": feature_names,
        "categorical_columns_encoded": cat_cols,
        "dropped_columns": cfg.get("drop_columns", []),
        "target_column": cfg.get("target_column", None),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
    }
    return (X, y, info) if return_info else (X, y)


def print_preprocessing_report(dataset_names, *, n_runs, n_fixed_test_points, train_fraction, fixed_test_seed, base_train_seed):
    print("=== PREPROCESSING CHECK ONLY: no model fitting will be run ===", flush=True)
    for name in dataset_names:
        X, y, info = load_dataset(name, return_info=True)
        print(f"\n[{name}] {info['dataset_tag']}", flush=True)
        print(f"  raw_shape={info['raw_shape']} -> final_X={X.shape}, final_y={y.shape}", flush=True)
        print(f"  removed_nonfinite_rows={info['removed_nonfinite_rows']}", flush=True)
        print(f"  dropped_columns={info.get('dropped_columns', [])}", flush=True)
        print(f"  categorical_columns_encoded={info.get('categorical_columns_encoded', [])}", flush=True)
        print(f"  n_features_after_preprocessing={X.shape[1]}", flush=True)
        print(f"  feature_names={info['feature_names']}", flush=True)
        print(
            f"  y: mean={np.mean(y):.6g}, std={np.std(y):.6g}, "
            f"min={np.min(y):.6g}, max={np.max(y):.6g}",
            flush=True,
        )
        splits = make_fixed100_splits(
            X,
            y,
            n_runs=n_runs,
            n_fixed_test_points=n_fixed_test_points,
            train_fraction=train_fraction,
            fixed_test_seed=fixed_test_seed,
            base_train_seed=base_train_seed,
        )
        test0 = splits[0]["test_idx"]
        same_test = all(np.array_equal(s["test_idx"], test0) for s in splits)
        print(
            f"  split check: fixed_test_n={len(test0)}, train_n_per_run={[len(s['train_idx']) for s in splits]}, "
            f"same_fixed_test_across_runs={same_test}",
            flush=True,
        )
        print(f"  first_10_fixed_test_idx={test0[:10].tolist()}", flush=True)


def get_python_memory_gb():
    try:
        out = subprocess.check_output(
            "ps -C python -o rss= | awk '{s+=$1} END {print s/1024/1024}'",
            shell=True,
            text=True,
        ).strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def get_n_python_processes():
    try:
        out = subprocess.check_output("ps -C python -o pid= | wc -l", shell=True, text=True).strip()
        return int(out)
    except Exception:
        return 0


def memory_logger(out_csv: Path, stop_event: threading.Event, interval_sec: int = 60):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "rss_gb_total", "n_python_processes"])
        while not stop_event.is_set():
            writer.writerow([
                round(time.time() - start, 1),
                round(get_python_memory_gb(), 4),
                get_n_python_processes(),
            ])
            f.flush()
            time.sleep(interval_sec)




def _estimate_long_output_gb(*, n_runs: int, n_chains: int, n_fixed_test_points: int, long_ndpost: int, long_store_every: int, store_preds: bool) -> float:
    """Conservative estimate of final+temporary CSV output for default_long.

    The dominant file is predictions with shape roughly
    (n_chains, n_fixed_test_points, long_ndpost / long_store_every).
    CSV is text, so we use a conservative 16 bytes per numeric value plus a
    2.5x factor for temporary files and assembly overlap.
    """
    saved_draws = int(long_ndpost // long_store_every)
    numeric_values = n_runs * n_chains * saved_draws * (2 + (n_fixed_test_points if store_preds else 0))
    return numeric_values * 16 * 2.5 / (1024 ** 3)


def _check_disk_space(path: Path, *, estimated_gb: float, min_free_disk_gb: float):
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    required_gb = float(min_free_disk_gb) + float(estimated_gb)
    print(
        f"[DISK] path={path} free={free_gb:.2f}GB, estimated_needed={estimated_gb:.2f}GB, "
        f"min_free_after_run={min_free_disk_gb:.2f}GB",
        flush=True,
    )
    if free_gb < required_gb:
        raise RuntimeError(
            f"Not enough free disk space at {path}. free={free_gb:.2f}GB, "
            f"estimated_needed={estimated_gb:.2f}GB, min_free_after_run={min_free_disk_gb:.2f}GB."
        )


def _print_run_resource_plan(*, dataset_tag: str, X_shape, n_runs: int, n_chains: int, n_jobs: int, long_ndpost: int, long_store_every: int, long_chunk_size: int, n_fixed_test_points: int, store_preds: bool):
    saved_draws = int(long_ndpost // long_store_every)
    print(
        f"[PLAN] {dataset_tag}: X_shape={tuple(X_shape)}, runs={n_runs}, chains={n_chains}, "
        f"parallel_jobs={n_jobs}, long_ndpost={long_ndpost}, store_every={long_store_every}, "
        f"saved_draws_per_chain={saved_draws}, chunk_size={long_chunk_size}, store_preds={store_preds}",
        flush=True,
    )
    print(
        f"[PLAN] expected default_long preds shape per run: "
        f"({n_chains}, {n_fixed_test_points}, {saved_draws})",
        flush=True,
    )

def run_fixed100_dataset_long_only(
    *,
    X,
    y,
    dataset_tag: str,
    store_dir: str | Path,
    n_runs: int,
    n_chains: int,
    n_jobs: int,
    n_trees: int = 100,
    long_ndpost: int = 1_000_000,
    long_nskip: int = 0,
    long_store_every: int = 100,
    long_chunk_size: int = 10_000,
    n_fixed_test_points: int = 100,
    train_fraction: float = 0.75,
    fixed_test_seed: int = 42,
    base_train_seed: int = 2026,
    base_chain_seed: int = 2024,
    store_preds: bool = True,
    progress_print: bool = True,
    min_free_disk_gb: float = 10.0,
    dataset_info: dict[str, Any] | None = None,
):
    store_root = Path(store_dir)
    store_root.mkdir(parents=True, exist_ok=True)
    dataset_root = store_root / dataset_tag
    dataset_root.mkdir(parents=True, exist_ok=True)

    _print_run_resource_plan(
        dataset_tag=dataset_tag,
        X_shape=np.asarray(X).shape,
        n_runs=n_runs,
        n_chains=n_chains,
        n_jobs=n_jobs,
        long_ndpost=long_ndpost,
        long_store_every=long_store_every,
        long_chunk_size=long_chunk_size,
        n_fixed_test_points=n_fixed_test_points,
        store_preds=store_preds,
    )
    est_gb = _estimate_long_output_gb(
        n_runs=n_runs,
        n_chains=n_chains,
        n_fixed_test_points=n_fixed_test_points,
        long_ndpost=long_ndpost,
        long_store_every=long_store_every,
        store_preds=store_preds,
    )
    _check_disk_space(store_root, estimated_gb=est_gb, min_free_disk_gb=min_free_disk_gb)

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
            "phase": "default_long_only",
            "n_rows": int(np.asarray(X).shape[0]),
            "n_features": int(np.asarray(X).shape[1]),
            "n_runs": n_runs,
            "n_chains": n_chains,
            "n_jobs": n_jobs,
            "n_fixed_test_points": n_fixed_test_points,
            "train_fraction": train_fraction,
            "fixed_test_seed": fixed_test_seed,
            "base_train_seed": base_train_seed,
            "base_chain_seed": base_chain_seed,
            "long_ndpost": long_ndpost,
            "long_nskip": long_nskip,
            "long_store_every": long_store_every,
            "long_chunk_size": long_chunk_size,
            "store_preds": store_preds,
            "preprocessing_info": dataset_info or {},
        },
    )

    for split_info in splits:
        run_id = int(split_info["run_id"])
        X_train = split_info["X_train"]
        y_train = split_info["y_train"]
        X_test_fixed = split_info["X_test_fixed"]
        y_test_fixed = split_info["y_test_fixed"]
        if progress_print:
            print(
                f"[{dataset_tag} RUN {run_id:03d}] default_long only: "
                f"fixed test n={X_test_fixed.shape[0]}, train n={X_train.shape[0]}, "
                f"ndpost={long_ndpost}, store_every={long_store_every}",
                flush=True,
            )

        # Since short-chain saving is skipped, save split/test artifacts here.
        for sub in ["subsample_X_test", "subsample_y_test", "indices"]:
            (dataset_root / sub).mkdir(parents=True, exist_ok=True)
        _save_numeric_csv(dataset_root / "subsample_X_test" / f"{dataset_tag}__run{run_id:03d}__subsample_X_test.csv", X_test_fixed)
        _save_numeric_csv(dataset_root / "subsample_y_test" / f"{dataset_tag}__run{run_id:03d}__subsample_y_test.csv", y_test_fixed)
        _save_numeric_csv(dataset_root / "indices" / f"{dataset_tag}__run{run_id:03d}__train_idx.csv", split_info["train_idx"])
        _save_numeric_csv(dataset_root / "indices" / f"{dataset_tag}__run{run_id:03d}__fixed_test_idx.csv", split_info["test_idx"])

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
                proposal_probs_default=default_proposal_probs,
                store_every=long_store_every,
                chunk_size=long_chunk_size,
                tmp_dir=tmp_root,
                store_preds=store_preds,
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
                "phase": "default_long_only",
                "run_id": run_id,
                "train_seed": split_info["train_seed"],
                "long_ndpost": long_ndpost,
                "long_nskip": long_nskip,
                "long_store_every": long_store_every,
                "long_chunk_size": long_chunk_size,
                "n_chains": n_chains,
                "n_jobs": n_jobs,
            },
            store_preds=store_preds,
        )
        shutil.rmtree(tmp_root, ignore_errors=True)
        del long_results
        gc.collect()
        if progress_print:
            print(f"[{dataset_tag} RUN {run_id:03d}] default_long only done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fixed-100 default_long-only benchmark for California Housing subsample.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=["calhousing"])
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--store-dir", type=str, default="store")
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fixed-test-seed", type=int, default=42)
    parser.add_argument("--base-train-seed", type=int, default=2026)
    parser.add_argument("--base-chain-seed", type=int, default=2024)
    parser.add_argument("--short-ndpost", type=int, default=2000, help="Only used if --run-full is set.")
    parser.add_argument("--short-nskip", type=int, default=0, help="Only used if --run-full is set.")
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--long-chunk-size", type=int, default=5000)
    parser.add_argument("--long-ndpost-override", type=int, default=None, help="Override per-dataset long_ndpost, useful for smoke tests.")
    parser.add_argument("--long-store-every-override", type=int, default=None, help="Override per-dataset long_store_every, useful for smoke tests.")
    # Keep original default requested by the project. Long-only mode does not use temperatures.
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1000000.0])
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--memory-log-interval", type=int, default=60)
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--no-store-preds", action="store_true", help="Do not save default_long prediction arrays. Usually keep predictions for LDA/comparison; use only for emergency disk saving.")
    parser.add_argument("--check-preprocessing", action="store_true", help="Load/preprocess/split-check only; do not fit any model.")
    parser.add_argument("--run-full", action="store_true", help="Run the original full pipeline with short methods + default_long. Default is long-only.")
    parser.add_argument("--skip-long", action="store_true", help="Only used with --run-full: run only short-chain methods.")
    args = parser.parse_args()

    if args.check_preprocessing:
        print_preprocessing_report(
            args.datasets,
            n_runs=args.n_runs,
            n_fixed_test_points=args.n_fixed_test_points,
            train_fraction=args.train_fraction,
            fixed_test_seed=args.fixed_test_seed,
            base_train_seed=args.base_train_seed,
        )
        return

    here = Path(__file__).resolve().parent
    store_root = here / args.store_dir
    print(f"Writing outputs to: {store_root}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"n_runs={args.n_runs}, n_chains={args.n_chains}, n_jobs={args.n_jobs}", flush=True)
    print(f"fixed_test_points={args.n_fixed_test_points}, train_fraction={args.train_fraction}", flush=True)
    print(f"run_full={args.run_full}, skip_long={args.skip_long}", flush=True)
    print(f"store_preds={not args.no_store_preds}, min_free_disk_gb={args.min_free_disk_gb}", flush=True)
    print(f"temperatures={args.temperatures}", flush=True)

    stop_event = threading.Event()
    mem_thread = threading.Thread(
        target=memory_logger,
        args=(store_root / "memory_log.csv", stop_event, args.memory_log_interval),
        daemon=True,
    )
    mem_thread.start()

    start = time.time()
    try:
        for name in args.datasets:
            cfg = DATASET_CONFIGS[name]
            X, y, info = load_dataset(name, return_info=True)
            long_ndpost = int(args.long_ndpost_override if args.long_ndpost_override is not None else cfg["long_ndpost"])
            long_store_every = int(args.long_store_every_override if args.long_store_every_override is not None else cfg["long_store_every"])
            print(f"\n=== DATASET {name} ===", flush=True)
            print(f"X={X.shape}, y={y.shape}, removed_nonfinite_rows={info.get('removed_nonfinite_rows', 0)}", flush=True)
            print(f"long_ndpost={long_ndpost}, long_store_every={long_store_every}", flush=True)

            if args.run_full:
                run_fixed100_dataset(
                    X=X,
                    y=y,
                    dataset_tag=cfg["dataset_tag"],
                    store_dir=store_root,
                    n_runs=args.n_runs,
                    n_chains=args.n_chains,
                    n_jobs=args.n_jobs,
                    n_trees=args.n_trees,
                    short_ndpost=args.short_ndpost,
                    short_nskip=args.short_nskip,
                    long_ndpost=long_ndpost,
                    long_store_every=long_store_every,
                    long_chunk_size=args.long_chunk_size,
                    n_fixed_test_points=args.n_fixed_test_points,
                    train_fraction=args.train_fraction,
                    fixed_test_seed=args.fixed_test_seed,
                    base_train_seed=args.base_train_seed,
                    base_chain_seed=args.base_chain_seed,
                    temperatures=tuple(args.temperatures),
                    swap_interval=args.swap_interval,
                    multi_tries=args.multi_tries,
                    store_preds=not args.no_store_preds,
                    progress_print=True,
                    run_long=not args.skip_long,
                )
            else:
                if args.skip_long:
                    raise ValueError("--skip-long is incompatible with default long-only mode. Use --run-full --skip-long for short-only.")
                run_fixed100_dataset_long_only(
                    X=X,
                    y=y,
                    dataset_tag=cfg["dataset_tag"],
                    store_dir=store_root,
                    n_runs=args.n_runs,
                    n_chains=args.n_chains,
                    n_jobs=args.n_jobs,
                    n_trees=args.n_trees,
                    long_ndpost=long_ndpost,
                    long_nskip=0,
                    long_store_every=long_store_every,
                    long_chunk_size=args.long_chunk_size,
                    n_fixed_test_points=args.n_fixed_test_points,
                    train_fraction=args.train_fraction,
                    fixed_test_seed=args.fixed_test_seed,
                    base_train_seed=args.base_train_seed,
                    base_chain_seed=args.base_chain_seed,
                    store_preds=not args.no_store_preds,
                    progress_print=True,
                    min_free_disk_gb=args.min_free_disk_gb,
                    dataset_info=info,
                )
    finally:
        stop_event.set()
        mem_thread.join(timeout=5)

    runtime_min = (time.time() - start) / 60
    print(f"DONE fixed100 benchmark in {runtime_min:.2f} min", flush=True)
    print(f"Memory log: {store_root / 'memory_log.csv'}", flush=True)


if __name__ == "__main__":
    main()
