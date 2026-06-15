from __future__ import annotations

import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from ucimlrepo import fetch_ucirepo

from bart_playground.DataGenerator import DataGenerator

from experiment_fixed100 import run_fixed100_dataset


DATASET_CONFIGS = {
    "abalone": {
        "dataset_tag": "fixed100_Abalone",
        "uci_id": 1,
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
        "drop_columns": ["Sex"],
        "target_kind": "targets",
    },
    "concrete": {
        "dataset_tag": "fixed100_Concrete",
        "uci_id": 165,
        "long_ndpost": 10_000_000,
        "long_store_every": 1000,
        "drop_columns": [],
        "target_kind": "targets",
    },
    "friedman": {
        "dataset_tag": "fixed100_Friedman",
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
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
        "long_ndpost": 1_000_000,
        "long_store_every": 100,
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


def load_dataset(name: str):
    cfg = DATASET_CONFIGS[name]

    if name in GENERATOR_SCENARIOS:
        return _load_generator_dataset(cfg, GENERATOR_SCENARIOS[name])

    ds = fetch_ucirepo(id=cfg["uci_id"])
    features = ds.data.features.copy()
    for col in cfg.get("drop_columns", []):
        if col in features.columns:
            features = features.drop(columns=[col])
    X = features.values.astype(float)
    y = np.asarray(ds.data.targets).reshape(-1).astype(float)
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
    parser.add_argument("--fixed-test-seed", type=int, default=42)
    parser.add_argument("--base-train-seed", type=int, default=2026)
    parser.add_argument("--base-chain-seed", type=int, default=2024)
    parser.add_argument("--short-ndpost", type=int, default=2000)
    parser.add_argument("--short-nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--long-chunk-size", type=int, default=10000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1000000.0])
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--memory-log-interval", type=int, default=60)
    parser.add_argument("--skip-long", action="store_true", help="Run only the four short-chain methods.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    store_root = here / args.store_dir
    print(f"Writing outputs to: {store_root}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"n_runs={args.n_runs}, n_chains={args.n_chains}, n_jobs={args.n_jobs}", flush=True)
    print(f"fixed_test_points={args.n_fixed_test_points}, train_fraction={args.train_fraction}", flush=True)
    print(f"skip_long={args.skip_long}", flush=True)

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
