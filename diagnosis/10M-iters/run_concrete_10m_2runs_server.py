from __future__ import annotations

import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from ucimlrepo import fetch_ucirepo

from experiment import run_parallel_experiments


def load_concrete_data():
    concrete = fetch_ucirepo(id=165)
    X = concrete.data.features.values.astype(float)
    y = np.asarray(concrete.data.targets).reshape(-1)
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
        out = subprocess.check_output(
            "ps -C python -o pid= | wc -l",
            shell=True,
            text=True,
        ).strip()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndpost", type=int, default=10_000_000)
    parser.add_argument("--nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--store-every", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--store-dir", type=str, default="store")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    store_dir = here / args.store_dir

    X, y = load_concrete_data()
    print(f"Concrete data: X={X.shape}, y={y.shape}", flush=True)
    print(f"Writing outputs to: {store_dir}", flush=True)
    print(
        f"Settings: ndpost={args.ndpost}, nskip={args.nskip}, "
        f"n_trees={args.n_trees}, n_runs={args.n_runs}, "
        f"n_chains={args.n_chains}, n_jobs={args.n_jobs}, "
        f"store_every={args.store_every}, chunk_size={args.chunk_size}",
        flush=True,
    )
    print(f"Expected stored states per chain: {args.ndpost // args.store_every}", flush=True)

    stop_event = threading.Event()
    mem_thread = threading.Thread(
        target=memory_logger,
        args=(store_dir / "memory_log.csv", stop_event, 60),
        daemon=True,
    )
    mem_thread.start()

    start = time.time()
    try:
        results = run_parallel_experiments(
            X,
            y,
            ndpost=args.ndpost,
            nskip=args.nskip,
            n_trees=args.n_trees,
            notebook="real4_Concrete",
            n_runs=args.n_runs,
            n_chains=args.n_chains,
            n_jobs=args.n_jobs,
            store_every=args.store_every,
            chunk_size=args.chunk_size,
            store_dir=str(store_dir),
            progress_print=True,
        )
    finally:
        stop_event.set()
        mem_thread.join(timeout=5)

    runtime_min = (time.time() - start) / 60
    print(f"DONE: streaming 10M 2-run experiment completed in {runtime_min:.2f} min", flush=True)
    print(f"Memory log: {store_dir / 'memory_log.csv'}", flush=True)

    for r in results:
        print(f"run {r['run_id']}: sigmas shape: {r['default']['sigmas_shape']}", flush=True)
        print(f"run {r['run_id']}: rmses shape: {r['default']['rmses_shape']}", flush=True)
        if "preds_shape" in r["default"]:
            print(f"run {r['run_id']}: preds shape: {r['default']['preds_shape']}", flush=True)


if __name__ == "__main__":
    main()
