from __future__ import annotations

import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from bart_playground.DataGenerator import DataGenerator

from experiment_fixed100 import run_fixed100_dataset

GLOBAL_FIXED_TEST_SEED = 42
GLOBAL_BASE_TRAIN_SEED = 2026
GLOBAL_BASE_CHAIN_SEED = 2024


DATASET_CONFIGS = {
    "abalone": {
        "dataset_tag": "fixed100_Abalone",
        "uci_id": 1,
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "drop_columns": ["Sex"],
        "target_column": None,
    },
    "concrete": {
        "dataset_tag": "fixed100_Concrete",
        "uci_id": 165,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "target_column": None,
    },
    "friedman": {
        "dataset_tag": "fixed100_Friedman",
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "n_samples": 2000,
        "n_features": 10,
        "noise": 1.0,
        "seed": 42,
    },
    "friedman_sparse": {
        "dataset_tag": "fixed100_FriedmanSparse",
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "n_samples": 2000,
        "n_features": 100,
        "noise": 1.0,
        "seed": 42,
    },
    "friedman_sparse_dir": {
        "dataset_tag": "fixed100_FriedmanSparseDir",
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "n_samples": 2000,
        "n_features": 100,
        "noise": 1.0,
        "seed": 42,
        "dirichlet_prior": True,
        "s_alpha": 1.0,
    },
    "friedman2": {
        "dataset_tag": "fixed100_Friedman2",
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "n_samples": 2000,
        "n_features": 10,
        "noise": 1.0,
        "seed": 42,
    },
    "friedman3": {
        "dataset_tag": "fixed100_Friedman3",
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "n_samples": 2000,
        "n_features": 10,
        "noise": 1.0,
        "seed": 42,
    },

    "ccpp": {
        "dataset_tag": "fixed100_CCPP",
        "uci_id": 294,
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "drop_columns": [],
        "categorical_columns": [],
        "target_column": None,
    },

    "parkinsons": {
        "dataset_tag": "fixed100_ParkinsonsTelemonitoring",
        "uci_id": 189,
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "drop_columns": ["subject#"],
        "target_column": None,
    },

    "seoul_bike": {
        "dataset_tag": "fixed100_SeoulBike",
        "uci_id": 560,
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "drop_columns": ["Date"],
        "categorical_columns": "auto",
        "target_column": "Rented Bike Count",
    },

    "calhousing": {
        "dataset_tag": "fixed100_CalHousing_subsample5000",
        "subsample_n": 5000,
        "subsample_seed": 42,
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
    },
}


GENERATOR_SCENARIOS = {
    "friedman": "friedman1",
    "friedman_sparse": "friedman1",
    "friedman_sparse_dir": "friedman1",
    "friedman2": "friedman2",
    "friedman3": "friedman3",
}


def _load_generator_dataset(cfg: dict, scenario: str):
    gen_kwargs = {
        "n_samples": int(cfg.get("n_samples", 2000)),
        "n_features": int(cfg.get("n_features", 10)),
        "random_seed": int(cfg.get("seed", 42)),
    }
    if "snr" in cfg:
        gen_kwargs["snr"] = float(cfg["snr"])
    else:
        gen_kwargs["noise"] = float(cfg.get("noise", cfg.get("noise_std", 1.0)))
    generator = DataGenerator(**gen_kwargs)
    X, y = generator.generate(scenario)
    return X.astype(float), np.asarray(y).reshape(-1).astype(float)


def _safe_column_names(df) -> list[str]:
    return [str(c) for c in getattr(df, "columns", [])]


def _select_target_and_features(features, targets, *, target_column=None):
    """Select a numeric target whether UCI stores it in targets or features."""
    X_df = features.copy()
    targets_df = (
        targets.copy() if hasattr(targets, "copy") else pd.DataFrame(targets)
    )

    if isinstance(target_column, str):
        if targets_df is not None and target_column in _safe_column_names(targets_df):
            y = targets_df[target_column].to_numpy()
        elif target_column in _safe_column_names(X_df):
            y = X_df[target_column].to_numpy()
            X_df = X_df.drop(columns=[target_column])
        else:
            raise ValueError(
                f"target_column={target_column!r} not found. "
                f"target columns={_safe_column_names(targets_df)}, "
                f"feature columns={_safe_column_names(X_df)}"
            )
    elif target_column is not None:
        if targets_df is None or targets_df.shape[1] == 0:
            raise ValueError("Integer target_column requested, but targets are empty")
        y = targets_df.iloc[:, int(target_column)].to_numpy()
    else:
        if targets_df is None or targets_df.shape[1] == 0:
            raise ValueError("Dataset has no target")
        if targets_df.shape[1] != 1:
            raise ValueError(
                f"Multiple target columns found {list(targets_df.columns)}; "
                "set target_column in DATASET_CONFIGS"
            )
        y = targets_df.iloc[:, 0].to_numpy()

    return X_df, np.asarray(y).reshape(-1).astype(float)


def _preprocess_features(features, *, drop_columns=None, categorical_columns=None):
    """Match the prior long-only preprocessing: drop, full one-hot, numeric."""
    X_df = features.copy()
    for col in list(drop_columns or []):
        if col in X_df.columns:
            X_df = X_df.drop(columns=[col])

    if categorical_columns == "auto":
        cat_cols = list(
            X_df.select_dtypes(include=["object", "category", "bool"]).columns
        )
    else:
        cat_cols = [
            c for c in list(categorical_columns or []) if c in X_df.columns
        ]

    if cat_cols:
        # Keep every category, matching the previous long-only runner exactly.
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    return X_df.to_numpy(dtype=float), list(X_df.columns), cat_cols


def _clean_finite_rows(X, y):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask], int((~mask).sum())


def load_dataset(name: str):
    cfg = DATASET_CONFIGS[name]

    if name in GENERATOR_SCENARIOS:
        return _load_generator_dataset(cfg, GENERATOR_SCENARIOS[name])

    if name == "calhousing":
        from sklearn.datasets import fetch_california_housing

        X, y = fetch_california_housing(return_X_y=True)
        rng = np.random.default_rng(cfg["subsample_seed"])
        # Sorting is required to reproduce the earlier California long-run
        # loader and therefore its fixed-test indices exactly.
        idx = np.sort(
            rng.choice(len(X), size=cfg["subsample_n"], replace=False)
        )
        X = X[idx].astype(float)
        y = np.asarray(y[idx]).reshape(-1).astype(float)
        X, y, n_removed = _clean_finite_rows(X, y)
        print(
            f"[LOAD] {name}: X={X.shape}, y={y.shape}, "
            f"removed_nonfinite_rows={n_removed}",
            flush=True,
        )
        return X, y

    ds = fetch_ucirepo(id=cfg["uci_id"])
    features, y = _select_target_and_features(
        ds.data.features,
        ds.data.targets,
        target_column=cfg.get("target_column"),
    )
    X, feature_names, cat_cols = _preprocess_features(
        features,
        drop_columns=cfg.get("drop_columns", []),
        categorical_columns=cfg.get("categorical_columns", "auto"),
    )
    X, y, n_removed = _clean_finite_rows(X, y)
    print(
        f"[LOAD] {name}: X={X.shape}, y={y.shape}, "
        f"removed_nonfinite_rows={n_removed}, "
        f"categorical_columns_encoded={cat_cols}, "
        f"features={feature_names}",
        flush=True,
    )

    return X, y


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


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-100 test point benchmark for selected datasets."
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=["abalone", "concrete"])
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--store-dir", type=str, default="store")
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fixed-test-seed", type=int, default=GLOBAL_FIXED_TEST_SEED)
    parser.add_argument("--base-train-seed", type=int, default=GLOBAL_BASE_TRAIN_SEED)
    parser.add_argument("--base-chain-seed", type=int, default=GLOBAL_BASE_CHAIN_SEED)
    parser.add_argument("--short-ndpost", type=int, default=2000)
    parser.add_argument("--short-nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--long-chunk-size", type=int, default=10000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1000000.0])
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--memory-log-interval", type=int, default=60)
    parser.add_argument("--skip-long", action="store_true", help="Run only the four short-chain methods.")
    parser.add_argument("--skip-short", action="store_true", help="Run only default_long.")
    args = parser.parse_args()

    if args.skip_long and args.skip_short:
        parser.error("--skip-long and --skip-short cannot be used together.")

    here = Path(__file__).resolve().parent
    store_root = here / args.store_dir
    print(f"Writing outputs to: {store_root}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"n_runs={args.n_runs}, n_chains={args.n_chains}, n_jobs={args.n_jobs}", flush=True)
    print(f"fixed_test_points={args.n_fixed_test_points}, train_fraction={args.train_fraction}", flush=True)
    print(f"skip_short={args.skip_short}, skip_long={args.skip_long}", flush=True)

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
            X, y = load_dataset(name)
            print(f"\n=== DATASET {name} ===", flush=True)
            print(f"X={X.shape}, y={y.shape}", flush=True)
            print(
                f"long_ndpost={cfg['long_ndpost']}, long_store_every={cfg['long_store_every']}",
                flush=True,
            )
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
                long_ndpost=cfg["long_ndpost"],
                long_store_every=cfg["long_store_every"],
                long_chunk_size=args.long_chunk_size,
                n_fixed_test_points=args.n_fixed_test_points,
                train_fraction=args.train_fraction,
                fixed_test_seed=args.fixed_test_seed,
                base_train_seed=args.base_train_seed,
                base_chain_seed=args.base_chain_seed,
                temperatures=tuple(args.temperatures),
                swap_interval=args.swap_interval,
                multi_tries=args.multi_tries,
                store_preds=True,
                progress_print=True,
                run_short=not args.skip_short,
                run_long=not args.skip_long,
                dirichlet_prior=cfg.get("dirichlet_prior", False),
                s_alpha=float(cfg.get("s_alpha", 1.0)),
            )
    finally:
        stop_event.set()
        mem_thread.join(timeout=5)

    runtime_min = (time.time() - start) / 60
    print(f"DONE fixed100 benchmark in {runtime_min:.2f} min", flush=True)
    print(f"Memory log: {store_root / 'memory_log.csv'}", flush=True)


if __name__ == "__main__":
    main()
