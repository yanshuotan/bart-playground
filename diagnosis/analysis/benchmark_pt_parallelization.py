#!/usr/bin/env python3
"""Compare PT execution modes using repeated Abalone timing runs.

The fixed test set, training subset, and PT temperature ladder are loaded from
``diagnosis/store/fixed100_Abalone`` so the benchmark uses the same setup as
the existing fixed-100 analysis.  A PT fit is counted as one requested chain;
its temperature replicas are the chain-internal work that is parallelized.

``default`` and ``mtmh`` are timed once each as non-PT baselines.  Each PT
method is timed once per entry in ``--pt-backends``, which defaults to both
``multiprocessing-pipe`` (parallel; joblib-loky and joblib-threading were
removed) and ``serial`` (the same sampler with ``n_jobs=1``).  The serial row
is what the parallel row must be divided by to get the parallel speed-up, so
the summary carries ``default_pt__serial`` next to
``default_pt__multiprocessing-pipe``.  Drop ``serial`` from ``--pt-backends``
for timings comparable to runs made before it existed.

``--repeats`` repeats the complete experiment with a new seed while keeping
the seed matched across configurations within a repeat.  PT runs one worker
process per physical core by default, each hosting one or more temperature
chains; see ``--pt-n-jobs``.  ``elapsed_seconds`` starts after model
construction and all preprocessing, then covers chain initialization,
backend start-up/shutdown, and the complete MCMC run.
"""

from __future__ import annotations

import os

# Avoid nested BLAS parallelism competing with the PT temperature workers.
# Existing scheduler/user settings remain authoritative.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import csv
import gc
import multiprocessing as mp
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = DIAGNOSIS_DIR.parent
STORE_DIR = DIAGNOSIS_DIR / "store" / "fixed100_Abalone"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bart_playground.bart import (
    DefaultBART,
    MultiBART,
    ParallelTemperingBART,
    resolve_pt_workers,
)
from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs


METHODS = ("default", "default_pt", "mtmh", "mtmh_pt")
# "serial" runs a PT method with n_jobs=1, i.e. every temperature stepped in
# this process. It is the baseline the parallel backend has to beat, so both
# are timed by default.
PT_BACKENDS = ("multiprocessing-pipe", "serial")
SUMMARY_FIELDS = (
    "mean_seconds",
    "std_seconds",
    "relative_to_default",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time the non-PT Abalone baselines once, then time default+PT and MTMH+PT "
            "once per selected PT execution mode (parallel and serial by default)."
        )
    )
    parser.add_argument("--run-id", type=int, default=0, help="Stored fixed-100 run to reuse (default: 0).")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Complete experiment repeats; repeat r uses chain-seed + r - 1 for every configuration (default: 1).",
    )
    parser.add_argument("--ndpost", type=int, default=10_000, help="Posterior iterations per method.")
    parser.add_argument("--nskip", type=int, default=0, help="Burn-in iterations per method.")
    parser.add_argument("--n-trees", type=int, default=100, help="Number of BART trees.")
    parser.add_argument("--multi-tries", type=int, default=10, help="Number of MTMH tries.")
    parser.add_argument("--chain-seed", type=int, default=3024, help="Shared base seed for all four methods.")
    parser.add_argument("--tree-alpha", type=float, default=0.95)
    parser.add_argument("--tree-beta", type=float, default=2.0)
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument(
        "--pt-n-jobs",
        type=int,
        default=-1,
        help=(
            "PT worker processes: -1 uses one per physical core (default), -2 all but "
            "one, 0 one per stored temperature, or an explicit positive count. When "
            "there are fewer workers than temperatures each worker hosts several "
            "chains, which is faster than oversubscribing. Timings depend on this; "
            "sampled results do not."
        ),
    )
    parser.add_argument(
        "--pt-backends",
        "--pt-backend",
        dest="pt_backends",
        nargs="+",
        choices=PT_BACKENDS,
        default=list(PT_BACKENDS),
        help=(
            "PT execution modes to time (default: both). 'multiprocessing-pipe' is "
            "the parallel backend; 'serial' is the same sampler with n_jobs=1 and is "
            "the baseline for the speed-up. Dropping 'serial' roughly halves the "
            "default_pt run and cuts far more from mtmh_pt."
        ),
    )
    parser.add_argument(
        "--temperature-file",
        type=Path,
        help="Optional temperature CSV override; the first row is used.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Methods to include (default: all four).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Timing summary CSV. Default: diagnosis/analysis/timing_outputs/<timestamp>.csv",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show sampler progress bars (off for cleaner timing).")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed method.")
    parser.add_argument("--dry-run", action="store_true", help="Validate data, indices, temperatures, and worker counts without fitting.")
    args = parser.parse_args()

    if args.run_id < 0:
        parser.error("--run-id must be non-negative")
    for name in ("repeats", "ndpost", "n_trees", "multi_tries", "swap_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.nskip < 0:
        parser.error("--nskip must be non-negative")
    return args


def load_abalone() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess UCI Abalone exactly as the fixed-100 runner did."""
    dataset = fetch_ucirepo(id=1)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one Abalone target column, found {list(targets.columns)}")

    if "Sex" in features.columns:
        features = features.drop(columns=["Sex"])
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    X = features.to_numpy(dtype=float)
    y = pd.to_numeric(targets.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not finite.all():
        X = X[finite]
        y = y[finite]
    return X, y, [str(column) for column in features.columns]


def _load_integer_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required stored index file not found: {path}")
    values = np.loadtxt(path, delimiter=",", comments="#", dtype=np.int64)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def load_stored_split(X: np.ndarray, y: np.ndarray, run_id: int) -> tuple[np.ndarray, ...]:
    """Reuse the exact train/test indices saved by the fixed-100 experiment."""
    stem = f"fixed100_Abalone__run{run_id:03d}"
    index_dir = STORE_DIR / "indices"
    test_idx = _load_integer_vector(index_dir / f"{stem}__fixed_test_idx.csv")
    train_idx = _load_integer_vector(index_dir / f"{stem}__train_idx.csv")

    if len(test_idx) != 100:
        raise ValueError(f"Expected 100 fixed test rows, found {len(test_idx)}")
    if len(np.unique(test_idx)) != len(test_idx) or len(np.unique(train_idx)) != len(train_idx):
        raise ValueError("Stored train/test indices contain duplicates")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Stored train and test indices overlap")
    all_idx = np.concatenate([train_idx, test_idx])
    if all_idx.min(initial=0) < 0 or all_idx.max(initial=-1) >= len(X):
        raise IndexError(f"Stored indices are incompatible with Abalone row count {len(X)}")

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], train_idx, test_idx


def default_temperature_path(run_id: int) -> Path:
    return (
        STORE_DIR
        / "swap_temperatures"
        / f"fixed100_Abalone__run{run_id:03d}__default_pt__swap_temperatures.csv"
    )


def load_temperature_ladder(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Temperature file not found: {path}")
    values = np.loadtxt(path, delimiter=",", comments="#", dtype=float)
    if values.ndim == 1:
        temperatures = values
    elif values.ndim == 2:
        temperatures = values[0]
        if not np.allclose(values, temperatures[None, :], rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Temperature CSV contains different ladders across rows; pass a single-row file with --temperature-file."
            )
    else:
        raise ValueError(f"Unexpected temperature array shape: {values.shape}")

    temperatures = np.asarray(temperatures, dtype=float).reshape(-1)
    if len(temperatures) < 2:
        raise ValueError("PT requires at least two temperatures")
    if not np.isfinite(temperatures).all() or np.any(temperatures <= 0):
        raise ValueError("Temperatures must be finite and positive")
    if not np.isclose(temperatures[0], 1.0):
        raise ValueError(f"The cold-chain temperature must be 1.0, found {temperatures[0]}")
    if np.any(np.diff(temperatures) <= 0):
        raise ValueError("Temperatures must be strictly increasing")
    return temperatures


def allocated_cpu_count() -> int | None:
    """Read common scheduler CPU-allocation variables when available."""
    for variable in ("SLURM_CPUS_PER_TASK", "PBS_NCPUS", "NSLOTS"):
        raw = os.environ.get(variable)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def make_model(
    method: str,
    args: argparse.Namespace,
    temperatures: np.ndarray,
    pt_workers: int,
    pt_backend: str,
    chain_seed: int,
):
    common: dict[str, Any] = {
        "ndpost": args.ndpost,
        "nskip": args.nskip,
        "n_trees": args.n_trees,
        "tree_alpha": args.tree_alpha,
        "tree_beta": args.tree_beta,
        "tol": 1,
        "random_state": chain_seed,
    }
    if method == "default":
        return DefaultBART(proposal_probs=default_proposal_probs, **common)
    if method == "mtmh":
        return MultiBART(proposal_probs=mtmh_proposal_probs, multi_tries=args.multi_tries, **common)

    pt_common: dict[str, Any] = {
        **common,
        "temperatures": temperatures,
        "swap_interval": args.swap_interval,
        "post_swap_repair_steps": 0,
        "store_chain_traces": False,
        "store_swap_diagnostics": False,
        "print_swap_diagnostics": False,
        # The serial backend is not a different code path, just n_jobs=1.
        "n_jobs": 1 if pt_backend == "serial" else pt_workers,
        "local_move_backend": "multiprocessing-pipe",
    }
    if method == "default_pt":
        return ParallelTemperingBART(proposal_probs=default_proposal_probs, **pt_common)
    if method == "mtmh_pt":
        return ParallelTemperingBART(
            proposal_probs=mtmh_proposal_probs,
            sampler_kind="multi",
            multi_tries=args.multi_tries,
            **pt_common,
        )
    raise ValueError(f"Unknown method: {method}")


def run_numba_warmup(
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    temperatures: np.ndarray,
    pt_workers: int,
) -> None:
    """Compile/cache the MTMH+PT Numba paths before any timed experiment."""
    print(
        "Numba warm-up: mtmh_pt, 10 trees, 20 steps, swap_interval=10, "
        "serial in the main process (excluded from timings)",
        flush=True,
    )
    warmup_model = ParallelTemperingBART(
        ndpost=20,
        nskip=0,
        n_trees=10,
        tree_alpha=args.tree_alpha,
        tree_beta=args.tree_beta,
        tol=1,
        proposal_probs=mtmh_proposal_probs,
        random_state=args.chain_seed,
        temperatures=temperatures,
        swap_interval=10,
        post_swap_repair_steps=0,
        store_chain_traces=False,
        store_swap_diagnostics=False,
        print_swap_diagnostics=False,
        n_jobs=1,
        sampler_kind="multi",
        multi_tries=args.multi_tries,
    )
    warmup_data = warmup_model.preprocessor.fit_transform(X_train, y_train)
    warmup_start = time.perf_counter()
    warmup_model.fit_with_data(warmup_data, quietly=True)
    warmup_seconds = time.perf_counter() - warmup_start
    print(f"Numba warm-up complete in {warmup_seconds:.3f}s; formal timing starts next.", flush=True)
    del warmup_model, warmup_data
    gc.collect()


def resolve_output_path(path: Path | None, run_id: int) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (SCRIPT_DIR / "timing_outputs" / f"abalone_run{run_id:03d}_pt_backend_timing_{stamp}.csv").resolve()


def build_run_plan(args: argparse.Namespace) -> list[tuple[str, int, str, int]]:
    """Return (method, repeat, backend, seed) jobs in execution order."""
    plan: list[tuple[str, int, str, int]] = []
    serial_methods = [method for method in args.methods if not method.endswith("_pt")]
    pt_methods = [method for method in args.methods if method.endswith("_pt")]
    for repeat in range(1, args.repeats + 1):
        chain_seed = args.chain_seed + repeat - 1
        for method in serial_methods:
            plan.append((method, repeat, "serial", chain_seed))
        for method in pt_methods:
            # Backends adjacent per method: on a throttling machine the serial
            # and parallel timings of a method should see similar conditions.
            for backend in args.pt_backends:
                plan.append((method, repeat, backend, chain_seed))
    return plan


def write_summary(path: Path, completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in completed:
        key = (str(result["method"]), str(result["pt_backend"]))
        grouped.setdefault(key, []).append(result)

    summary: list[dict[str, Any]] = []
    for (method, backend), rows in grouped.items():
        elapsed_seconds = [float(row["elapsed_seconds"]) for row in rows]
        # Non-PT methods keep a bare label; every PT row names its backend so
        # that default_pt__serial and default_pt__multiprocessing-pipe coexist.
        row_label = f"{method}__{backend}" if method.endswith("_pt") else method
        summary.append(
            {
                "row_label": row_label,
                "method": method,
                "pt_backend": backend,
                "mean_seconds": f"{statistics.mean(elapsed_seconds):.9f}" if elapsed_seconds else "",
                "std_seconds": (
                    f"{statistics.stdev(elapsed_seconds):.9f}"
                    if len(elapsed_seconds) > 1
                    else "0.000000000" if elapsed_seconds else ""
                ),
                "relative_to_default": "",
            }
        )

    default_row = next(
        (row for row in summary if row["method"] == "default" and row["mean_seconds"]),
        None,
    )
    if default_row is not None:
        default_mean = float(default_row["mean_seconds"])
        for row in summary:
            if row["mean_seconds"]:
                row["relative_to_default"] = f"{float(row['mean_seconds']) / default_mean:.6f}"

    method_order = {method: position for position, method in enumerate(METHODS)}
    backend_order = {name: position for position, name in enumerate(PT_BACKENDS)}
    summary.sort(key=lambda row: (method_order[row["method"]], backend_order[row["pt_backend"]]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *SUMMARY_FIELDS])
        for row in summary:
            writer.writerow([row["row_label"], *(row[field] for field in SUMMARY_FIELDS)])
    return summary


def run_benchmark(args: argparse.Namespace) -> int:
    X, y, feature_names = load_abalone()
    X_train, y_train, X_test, _y_test, train_idx, test_idx = load_stored_split(X, y, args.run_id)
    temperature_path = (args.temperature_file or default_temperature_path(args.run_id)).expanduser().resolve()
    temperatures = load_temperature_ladder(temperature_path)
    # 0 keeps the historical "one worker per temperature"; everything else is
    # resolved by the model's own rule so the two cannot drift apart.
    pt_workers = resolve_pt_workers(
        None if args.pt_n_jobs == 0 else args.pt_n_jobs, len(temperatures)
    )
    if pt_workers < 2:
        raise ValueError("PT internal parallelization requires at least two workers; use --pt-n-jobs >= 2")

    output_path = resolve_output_path(args.output, args.run_id)
    run_plan = build_run_plan(args)
    allocation = allocated_cpu_count()
    print(f"Abalone: X={X.shape}; train={X_train.shape}; fixed test={X_test.shape}")
    print(f"Features: {feature_names}")
    print(f"Stored indices: train={len(train_idx)}, test={len(test_idx)}, run={args.run_id}")
    print(f"Temperature ladder: {len(temperatures)} points from {temperatures[0]:g} to {temperatures[-1]:g}")
    print(f"PT internal parallelism: {pt_workers} workers")
    print(f"PT backends: {', '.join(args.pt_backends)}")
    print(f"Complete experiment repeats: {args.repeats}")
    print(f"Run plan: {len(run_plan)} timed fits")
    print("Pre-run warm-up: mtmh_pt, 10 trees, 20 steps, swap_interval=10 (not timed)")
    available_cpus = allocation if allocation is not None else os.cpu_count()
    if available_cpus is not None and available_cpus < pt_workers:
        remedy = "request more CPUs"
        print(f"WARNING: this environment reports {available_cpus} available CPUs but PT requests {pt_workers}; {remedy}.")
    if args.dry_run:
        for position, (method, repeat, backend, chain_seed) in enumerate(run_plan, start=1):
            print(f"  {position:>2}. {method:<10} backend={backend:<20} repeat={repeat} seed={chain_seed}")
        print("Dry run complete; no models were fitted and no timing CSV was written.")
        return 0
    
    run_numba_warmup(args, X_train, y_train, temperatures, pt_workers)
    print(f"Timing summary CSV: {output_path}")
    failures = 0
    completed: list[dict[str, Any]] = []
    for position, (method, repeat, backend, chain_seed) in enumerate(run_plan, start=1):
        gc.collect()
        print(
            f"\n[{position}/{len(run_plan)}] {method}: backend={backend}, repeat={repeat}, seed={chain_seed}",
            flush=True,
        )
        elapsed_start = None
        model = None
        try:
            model = make_model(method, args, temperatures, pt_workers, backend, chain_seed)
            prepared_data = model.preprocessor.fit_transform(X_train, y_train)

            elapsed_start = time.perf_counter()
            model.fit_with_data(prepared_data, quietly=not args.show_progress)
            elapsed_seconds = time.perf_counter() - elapsed_start
            completed.append(
                {
                    "method": method,
                    "pt_backend": backend,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            write_summary(output_path, completed)
            print(f"[{method}/{backend}] elapsed={elapsed_seconds:.3f}s", flush=True)
        except Exception:
            failures += 1
            traceback.print_exc()
        finally:
            del model
            gc.collect()

        if failures and args.fail_fast:
            break

    summary = write_summary(output_path, completed)
    print("\nElapsed-time summary")
    for row in summary:
        timing = f"{float(row['mean_seconds']):.3f}s" if row["mean_seconds"] else "failed"
        print(
            f"  {row['method']:<10} {row['pt_backend']:<20} {timing:>12} "
            f"std={row['std_seconds'] or '-'} relative_to_default={row['relative_to_default'] or '-'}"
        )
    print(f"Summary: {output_path}")
    return 1 if failures else 0


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
