#!/usr/bin/env python3
"""Time serial vs parallel PT on the fixed-100 datasets, four chains per run.

For every dataset in ``--datasets`` the benchmark rebuilds the fixed-100 split of
``--run-id`` with the experiment's own code (checked against stored indices when
they exist) and reuses the stored PT temperature ladder of that run.  It then
times, chain by chain (chains are *not* run in parallel, so they do not compete
for CPUs):

    default                 DefaultBART
    mtmh                    MultiBART
    default_pt (serial)     ParallelTemperingBART, n_jobs=1
    default_pt (parallel)   ParallelTemperingBART, one worker per temperature
    mtmh_pt (serial)        MTMH + PT, n_jobs=1
    mtmh_pt (parallel)      MTMH + PT, one worker per temperature

Chain c of run r uses seed ``2024 + 1000*r + c`` (the store's short-chain seeds)
unless ``--chain-seed`` overrides the base.  ``elapsed_seconds`` starts after
model construction and preprocessing and covers chain initialisation, worker
start-up/shutdown and the complete MCMC run.

With ``--search-ladder`` the stored ladder is not reused. The production
ladder search (``run_fixed100.py`` defaults) runs again first with one worker
per temperature and is timed; its ladder feeds the PT rows, and it is compared
with the stored one (``--ladder-store``). The summaries then add the search time
and each PT arm's cost including it.

For each fit the script measures, on Linux, the peak number of live PT worker
processes.  It also records the CPUs the scheduler allocated (``PBS_NCPUS``)
and the process CPU affinity.

Outputs (updated after every fit) in ``--output-dir``, by default
``diagnosis/timing``:
    <dataset>_run<r>_per_chain.csv   one row per timed fit
    <dataset>_run<r>_summary.csv     one row per method, totals over chains
    summary.md                       one table per dataset, rebuilt from every
                                     per-chain CSV in the directory, so jobs
                                     sharing it add tables instead of clobbering
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
import json
import multiprocessing as mp
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DIAGNOSIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = DIAGNOSIS_DIR.parent

for _path in (REPO_ROOT, DIAGNOSIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bart_playground.bart import (  # noqa: E402
    DefaultBART,
    MultiBART,
    ParallelTemperingBART,
    resolve_pt_workers,
)
from bart_playground.samplers import default_proposal_probs, mtmh_proposal_probs  # noqa: E402

import experiment_fixed100 as exp  # noqa: E402
from fixed100_support import (  # noqa: E402
    DATASET_CONFIGS,
    generate_sparse_variant,
    load_dataset,
)
from run_fixed100 import repo_nested_friedman1_generator  # noqa: E402


class _Source(NamedTuple):
    """Where a dataset's draws live and how to rebuild its inputs.

    `directory` is the store folder, `tag` the prefix inside its file names.
    The two differ for the sparse variants, which were produced by two
    different entry points.
    """

    directory: str
    tag: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]
    # Prior settings the production run used, as recorded in its metadata. The
    # DART sweep runs with the Dirichlet prior on splitting proportions, so it
    # has to be timed with it too.
    cfg: dict = {}


def _named(key: str) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    return lambda: load_dataset(key)[:2]


def _sparse(variant: str) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    def build() -> tuple[np.ndarray, np.ndarray]:
        X, y, _ = generate_sparse_variant(
            variant, n_samples=2000, seed=42, noise_sd=1.0,
            friedman_generator=repo_nested_friedman1_generator,
        )
        return X, y

    return build


DART = {"dirichlet_prior": True, "s_alpha": 1.0}

SOURCES = {
    "abalone": _Source("fixed100_Abalone", "fixed100_Abalone", _named("abalone")),
    "airfoil": _Source("fixed100_Airfoil", "fixed100_Airfoil", _named("airfoil")),
    "calhousing": _Source("fixed100_CalHousing_subsample5000",
                          "fixed100_CalHousing_subsample5000", _named("calhousing")),
    "ccpp": _Source("fixed100_CCPP", "fixed100_CCPP", _named("ccpp")),
    "concrete": _Source("fixed100_Concrete", "fixed100_Concrete", _named("concrete")),
    "cpu_act": _Source("fixed100_CPUAct", "fixed100_CPUAct", _named("cpu_act")),
    "friedman": _Source("fixed100_Friedman", "fixed100_Friedman", _named("friedman")),
    "friedman_p20": _Source("fixed100_FriedmanSparseDir_p20",
                            "fixed100_FriedmanSparseDirP20K5", _sparse("friedman_p20_k5"),
                            DART),
    "friedman_p100": _Source("fixed100_FriedmanSparseDir_p100", "fixed100_FriedmanSparseDir",
                             _named("friedman_sparse_dir"), DART),
    "friedman_p200": _Source("fixed100_FriedmanSparseDir_p200",
                             "fixed100_FriedmanSparseDirP200K5", _sparse("friedman_p200_k5"),
                             DART),
    "seoul_bike": _Source("fixed100_SeoulBike", "fixed100_SeoulBike", _named("seoul_bike")),
    "superconduct": _Source("fixed100_Superconduct_subsample5000",
                            "fixed100_Superconduct_subsample5000", _named("superconduct")),
}
DATASETS = tuple(SOURCES)
# (label, method, backend) in execution order within a chain. Serial and
# parallel runs of a PT method are adjacent so both see similar node conditions.
# The paper's cost table reads the four production rows; the serial PT variants
# cost two orders of magnitude more and appear in no reported number, so they
# are off by default and can be restored with --with-serial.
CONFIGS = (
    ("default", "default", "none"),
    ("mtmh", "mtmh", "none"),
    ("default_pt (parallel)", "default_pt", "parallel"),
    ("mtmh_pt (parallel)", "mtmh_pt", "parallel"),
)
SERIAL_CONFIGS = (
    ("default_pt (serial)", "default_pt", "serial"),
    ("mtmh_pt (serial)", "mtmh_pt", "serial"),
)
PER_CHAIN_FIELDS = (
    "dataset", "run_id", "chain", "seed", "label", "method", "backend", "n_temperatures",
    "pt_workers_requested", "peak_worker_processes", "allocated_cpus",
    "affinity_cpus", "n_rows", "n_train", "n_test", "elapsed_seconds",
)
SUMMARY_FIELDS = (
    "label", "n_temperatures", "parallel_cpus", "n_chains_timed",
    "mean_seconds_per_chain", "std_seconds_per_chain", "relative_to_default",
)
# Summary row for the timed search; its "per chain" time is the one search.
SEARCH_LABEL = "ladder search (parallel)"
SEARCH_FIELDS = (
    "dataset", "run_id", "search_seconds", "rounds", "n_temperatures", "peak_worker_processes",
    "allocated_cpus", "affinity_cpus", "n_search_points", "stored_n_temperatures",
    "matches_stored_ladder", "temperatures",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time default/MTMH and serial vs parallel PT, chain by chain, on the fixed-100 datasets."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--run-id", type=int, default=0, help="Fixed-100 run whose split and ladder are reused (default: 0).")
    parser.add_argument("--n-chains", type=int, default=4, help="Chains per dataset, run one after another (default: 4).")
    parser.add_argument(
        "--chain-seed",
        type=int,
        default=None,
        help="Base seed; chain c uses base + c. Default: 2024 + 1000*run_id, the store's short-chain seeds.",
    )
    parser.add_argument("--ndpost", type=int, default=10_000, help="Posterior iterations per fit.")
    parser.add_argument("--nskip", type=int, default=0, help="Burn-in iterations per fit.")
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--tree-alpha", type=float, default=0.95)
    parser.add_argument("--tree-beta", type=float, default=2.0)
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument(
        "--pt-n-jobs",
        type=int,
        default=0,
        help=(
            "Workers for the parallel PT rows: 0 = one per temperature (default), -1 one per "
            "physical core, or an explicit count. Serial rows always use n_jobs=1."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=[label for label, _, _ in CONFIGS + SERIAL_CONFIGS],
        default=None,
        help="Subset of the six configurations to time (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: diagnosis/timing/. Files are named after the dataset, so "
             "several jobs can share one directory; summary.md is rebuilt from all of them.",
    )
    parser.add_argument("--with-serial", action="store_true",
                        help="Also time the serial PT variants, which the paper does not report.")
    parser.add_argument(
        "--search-ladder", action="store_true",
        help="Search the ladder again with the production settings, one worker per temperature, "
             "time the search, and use its ladder for the PT rows instead of the stored one. "
             "The stored ladder, when present, is only compared against.",
    )
    parser.add_argument("--ladder-store", type=Path, default=None,
                        help="Store whose run ladder the searched one is compared with "
                             "(default: diagnosis/store).")
    parser.add_argument("--ladder-tmax", type=float, default=100.0)
    parser.add_argument("--ladder-init-size", type=int, default=10)
    parser.add_argument("--ladder-target-rate", type=float, default=0.4)
    parser.add_argument("--ladder-max-rounds", type=int, default=10)
    parser.add_argument("--ladder-ndpost", type=int, default=500)
    parser.add_argument("--ladder-nskip", type=int, default=500)
    parser.add_argument("--ladder-search-points", type=int, default=1000)
    parser.add_argument("--ladder-random-state", type=int, default=123)
    parser.add_argument("--show-progress", action="store_true", help="Show sampler progress bars (off for cleaner timing).")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed fit.")
    parser.add_argument("--dry-run", action="store_true", help="Check data, splits, ladders and worker counts without fitting.")
    args = parser.parse_args()
    if args.labels is None:
        args.labels = [label for label, _, _ in
                       (CONFIGS + SERIAL_CONFIGS if args.with_serial else CONFIGS)]

    if args.run_id < 0:
        parser.error("--run-id must be non-negative")
    for name in ("n_chains", "ndpost", "n_trees", "multi_tries", "swap_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.nskip < 0:
        parser.error("--nskip must be non-negative")
    if args.chain_seed is None:
        args.chain_seed = 2024 + 1000 * args.run_id
    return args


# ---------------------------------------------------------------------------
# Data, split and temperature ladder
# ---------------------------------------------------------------------------

LADDER_STORE: Path | None = None  # set from --ladder-store


def _stored_run_dirs(name: str, run_id: int) -> list[Path]:
    """Directories that may hold this run's stored indices and ladders."""
    source = SOURCES[name]
    first = [LADDER_STORE / source.directory] if LADDER_STORE is not None else []
    return first + [
        DIAGNOSIS_DIR / "store" / source.directory,
        DIAGNOSIS_DIR / "store_seed2024" / f"s24_short_{name}_r{run_id}" / source.directory,
    ]


def _load_vector(path: Path, dtype) -> np.ndarray:
    return np.asarray(np.loadtxt(path, delimiter=",", comments="#", dtype=dtype)).reshape(-1)


def load_split(name: str, run_id: int) -> dict[str, Any]:
    """Rebuild the fixed-100 split and check it against any stored indices."""
    X, y = SOURCES[name].loader()
    split = exp.make_fixed100_splits(X, y, n_runs=run_id + 1)[run_id]
    tag = SOURCES[name].tag
    checked = []
    for directory in _stored_run_dirs(name, run_id):
        train_path = directory / "indices" / f"{tag}__run{run_id:03d}__train_idx.csv"
        test_path = directory / "indices" / f"{tag}__run{run_id:03d}__fixed_test_idx.csv"
        if train_path.is_file() and test_path.is_file():
            if not (np.array_equal(_load_vector(train_path, np.int64), split["train_idx"])
                    and np.array_equal(_load_vector(test_path, np.int64), split["test_idx"])):
                raise ValueError(f"Rebuilt split for {name} run {run_id} differs from {directory / 'indices'}")
            checked.append(str(directory))
    split["n_rows"] = X.shape[0]
    split["checked_against"] = checked
    return split


def load_temperature_ladder(name: str, run_id: int) -> tuple[np.ndarray, Path]:
    """The stored ladder of this run; default_pt and mtmh_pt must agree."""
    tag = SOURCES[name].tag
    for directory in _stored_run_dirs(name, run_id):
        ladders = {}
        for method in ("default_pt", "mtmh_pt"):
            path = directory / "swap_temperatures" / f"{tag}__run{run_id:03d}__{method}__swap_temperatures.csv"
            if path.is_file():
                values = np.loadtxt(path, delimiter=",", comments="#", dtype=float, ndmin=2)
                if not np.allclose(values, values[0][None, :], rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Different ladders across chains in {path}")
                ladders[method] = values[0]
        if not ladders:
            continue
        temperatures = next(iter(ladders.values()))
        if any(len(t) != len(temperatures) or not np.allclose(t, temperatures) for t in ladders.values()):
            raise ValueError(f"default_pt and mtmh_pt ladders differ in {directory}")
        if len(temperatures) < 2 or not np.isclose(temperatures[0], 1.0) or np.any(np.diff(temperatures) <= 0):
            raise ValueError(f"Invalid stored ladder in {directory}: {temperatures}")
        return np.asarray(temperatures, dtype=float), directory
    raise FileNotFoundError(f"No stored temperature ladder for {name} run {run_id} in {_stored_run_dirs(name, run_id)}")


def timed_ladder_search(name: str, split: dict[str, Any], args) -> dict[str, Any]:
    """Run the production ladder search again, one worker per temperature, and time it.

    Same subset, initial ladder, seeds and rules as `run_fixed100.py` with its
    defaults (bisection on one continuing pilot, 500 burn-in + 500 counted
    iterations per round); the only change is `n_jobs=None`, so every round's
    fit gives each temperature its own process.
    """
    X_train, y_train = split["X_train"], split["y_train"]
    if args.ladder_search_points and args.ladder_search_points < X_train.shape[0]:
        rng = np.random.default_rng(split["train_seed"])
        idx = rng.choice(X_train.shape[0], args.ladder_search_points, replace=False)
        X_search, y_search = X_train[idx], y_train[idx]
    else:
        X_search, y_search = X_train, y_train
    cfg = SOURCES[name].cfg
    with ChildProcessMonitor() as monitor:
        start = time.perf_counter()
        temperatures, _rates, history = exp.quick_ladder_search(
            X_search, y_search,
            n_trees=args.n_trees, tree_alpha=args.tree_alpha, tree_beta=args.tree_beta,
            proposal_probs=default_proposal_probs,
            target_rate=args.ladder_target_rate, max_rounds=args.ladder_max_rounds,
            max_temperatures=None, ndpost=args.ladder_ndpost, nskip=args.ladder_nskip,
            random_state=args.ladder_random_state + 10000 * args.run_id,
            swap_interval=args.swap_interval, post_swap_repair_steps=0,
            initial_temperatures=initial_temperatures(args),
            dirichlet_prior=cfg.get("dirichlet_prior", False),
            s_alpha=float(cfg.get("s_alpha", 1.0)),
            progress_print=True, progress_prefix=f"[{SOURCES[name].tag} search] ",
            n_jobs=None,
        )
        elapsed = time.perf_counter() - start
    return {
        "temperatures": np.asarray(temperatures, dtype=float),
        "search_seconds": elapsed,
        "rounds": len(history),
        "peak_worker_processes": monitor.peak_children if monitor.available else "",
        "n_search_points": X_search.shape[0],
    }


def initial_temperatures(args) -> tuple[float, ...]:
    """The production starting ladder, as `run_fixed100.py` builds it."""
    return tuple(np.geomspace(1.0, args.ladder_tmax, args.ladder_init_size).tolist())


# ---------------------------------------------------------------------------
# CPU measurement
# ---------------------------------------------------------------------------

def allocated_cpu_count() -> int | None:
    for variable in ("PBS_NCPUS", "SLURM_CPUS_PER_TASK", "NSLOTS"):
        raw = os.environ.get(variable)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def affinity_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


class ChildProcessMonitor:
    """Count live PT worker processes of this process via /proc (Linux only).

    Workers are the ``multiprocessing.spawn`` children; the ``resource_tracker``
    helper that multiprocessing also starts is not one.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.available = Path("/proc/self/stat").exists()
        self.peak_children = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _workers(self) -> int:
        me = os.getpid()
        workers = 0
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as handle:
                    stat = handle.read().decode(errors="replace")
                fields = stat[stat.rfind(")") + 2:].split()
                if int(fields[1]) != me:  # fields[1] is ppid
                    continue
                with open(f"/proc/{entry.name}/cmdline", "rb") as handle:
                    cmdline = handle.read()
            except (OSError, IndexError, ValueError):
                continue
            if b"multiprocessing.spawn" in cmdline:
                workers += 1
        return workers

    def _loop(self):
        while not self._stop.is_set():
            self.peak_children = max(self.peak_children, self._workers())
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return False


def timed_fit(model, data, quietly: bool) -> dict[str, Any]:
    with ChildProcessMonitor() as monitor:
        start = time.perf_counter()
        model.fit_with_data(data, quietly=quietly)
        elapsed = time.perf_counter() - start
    return {
        "elapsed_seconds": elapsed,
        "peak_worker_processes": monitor.peak_children if monitor.available else "",
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model(method: str, backend: str, args, temperatures: np.ndarray, pt_workers: int, seed: int, cfg: dict):
    common: dict[str, Any] = {
        "ndpost": args.ndpost,
        "nskip": args.nskip,
        "n_trees": args.n_trees,
        "tree_alpha": args.tree_alpha,
        "tree_beta": args.tree_beta,
        "tol": 1,
        "random_state": seed,
        "dirichlet_prior": cfg.get("dirichlet_prior", False),
        "s_alpha": float(cfg.get("s_alpha", 1.0)),
    }
    if method == "default":
        return DefaultBART(proposal_probs=default_proposal_probs, **common)
    if method == "mtmh":
        return MultiBART(proposal_probs=mtmh_proposal_probs, multi_tries=args.multi_tries, **common)
    pt_common = {
        **common,
        "temperatures": temperatures,
        "swap_interval": args.swap_interval,
        "post_swap_repair_steps": 0,
        "store_chain_traces": False,
        "store_swap_diagnostics": False,
        "print_swap_diagnostics": False,
        "n_jobs": 1 if backend == "serial" else pt_workers,
    }
    if method == "default_pt":
        return ParallelTemperingBART(proposal_probs=default_proposal_probs, **pt_common)
    if method == "mtmh_pt":
        return ParallelTemperingBART(
            proposal_probs=mtmh_proposal_probs, sampler_kind="multi", multi_tries=args.multi_tries, **pt_common
        )
    raise ValueError(f"Unknown method: {method}")


def run_numba_warmup(args, X_train, y_train, temperatures, cfg=None) -> None:
    """Compile the MTMH+PT Numba paths before any timed fit (not timed)."""
    model = ParallelTemperingBART(
        ndpost=20, nskip=0, n_trees=10, tree_alpha=args.tree_alpha, tree_beta=args.tree_beta, tol=1,
        proposal_probs=mtmh_proposal_probs, random_state=args.chain_seed, temperatures=temperatures,
        swap_interval=10, post_swap_repair_steps=0, store_chain_traces=False, store_swap_diagnostics=False,
        print_swap_diagnostics=False, n_jobs=1, sampler_kind="multi", multi_tries=args.multi_tries,
        dirichlet_prior=(cfg or {}).get("dirichlet_prior", False),
        s_alpha=float((cfg or {}).get("s_alpha", 1.0)),
    )
    data = model.preprocessor.fit_transform(X_train, y_train)
    start = time.perf_counter()
    model.fit_with_data(data, quietly=True)
    print(f"  Numba warm-up (not timed): {time.perf_counter() - start:.1f}s", flush=True)
    del model, data
    gc.collect()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(rows: list[dict[str, Any]], search: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Per-method means relative to Default's mean per chain.

    A timed ladder search becomes one more row: its single wall time, divided by
    the same Default mean per chain.
    """
    summary = []
    # Serial rows only exist when --with-serial was given; the lookup below
    # skips any configuration that was not timed, so listing both is safe.
    for label, _method, backend in CONFIGS + SERIAL_CONFIGS:
        mine = [row for row in rows if row["label"] == label]
        if not mine:
            continue
        seconds = [float(row["elapsed_seconds"]) for row in mine]
        if backend == "parallel":
            peaks = sorted({row["peak_worker_processes"] for row in mine if row["peak_worker_processes"] != ""})
            parallel_cpus = "/".join(str(p) for p in peaks) if peaks else "n/a"
        else:
            parallel_cpus = "1"
        summary.append({
            "label": label,
            "n_temperatures": mine[0]["n_temperatures"] if backend != "none" else "",
            "parallel_cpus": parallel_cpus,
            "n_chains_timed": len(mine),
            "mean_seconds_per_chain": f"{statistics.mean(seconds):.1f}",
            "std_seconds_per_chain": f"{statistics.stdev(seconds):.1f}" if len(seconds) > 1 else "",
            "relative_to_default": "",
            "_mean": statistics.mean(seconds),
        })
    if search is not None:
        seconds = float(search["search_seconds"])
        summary.append({
            "label": SEARCH_LABEL,
            "n_temperatures": search["n_temperatures"],
            "parallel_cpus": search["peak_worker_processes"] or "n/a",
            "n_chains_timed": 1,
            "mean_seconds_per_chain": f"{seconds:.1f}",
            "std_seconds_per_chain": "",
            "relative_to_default": "",
            "_mean": seconds,
        })
    default = next((row for row in summary if row["label"] == "default"), None)
    if default is not None:
        for row in summary:
            row["relative_to_default"] = f"{row['_mean'] / default['_mean']:.2f}"
    return summary


def read_search(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def sections_from_dir(directory: Path) -> list[dict[str, Any]]:
    """Rebuild one table section per dataset from the per-chain CSVs in a directory.

    summary.md is written from these, so several jobs writing into the same
    directory add tables instead of overwriting each other's.
    """
    sections = []
    for path in sorted(directory.glob("*_per_chain.csv")):
        with path.open(encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if not rows:
            continue
        for row in rows:
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
        first = rows[0]
        search = read_search(path.with_name(path.name.replace("_per_chain.csv", "_ladder_search.csv")))
        present = {r["label"] for r in rows}
        last_label = next(label for label, _, _ in reversed(CONFIGS) if label in present)
        sections.append({
            "dataset": first["dataset"],
            "n_rows": first.get("n_rows", ""), "n_train": first.get("n_train", ""), "n_test": first.get("n_test", ""),
            "n_temps": next((r["n_temperatures"] for r in rows if r["n_temperatures"]), "-"),
            "allocated": first.get("allocated_cpus") or "-", "affinity": first.get("affinity_cpus") or "-",
            # A chain counts as done once its last timed configuration is in.
            "chains_done": len({r["chain"] for r in rows if r["label"] == last_label}),
            "search": search,
            "summary": summarize(rows, search),
        })
    order = [SOURCES[name].tag for name in DATASETS]
    sections.sort(key=lambda s: order.index(s["dataset"]) if s["dataset"] in order else len(order))
    return sections


def write_markdown(path: Path, sections: list[dict[str, Any]], args) -> None:
    lines = [
        "# PT serial vs parallel timing",
        "",
        f"run_id={args.run_id}, chains per dataset={args.n_chains} (run one after another), ndpost={args.ndpost}, "
        f"nskip={args.nskip}, n_trees={args.n_trees}, swap_interval={args.swap_interval}, "
        f"chain seeds={args.chain_seed}+c, updated {datetime.now().isoformat(timespec='seconds')}",
        "",
        "`parallel CPUs` = peak number of live PT worker processes measured during each parallel fit "
        "(1 for serial / non-PT rows). Times are per chain; `x default` = mean per-chain time "
        "/ default's mean per-chain time. When the ladder was searched in the job, the "
        f"`{SEARCH_LABEL}` row is that one search, also divided by default's mean per-chain time.",
        "",
    ]
    for section in sections:
        lines += [
            f"## {section['dataset']} (run {args.run_id:03d})",
            "",
            f"n={section['n_rows']}, train={section['n_train']}, test={section['n_test']}, temperatures={section['n_temps']}, "
            f"allocated CPUs={section['allocated']}, affinity CPUs={section['affinity']}, "
            f"chains timed so far={section['chains_done']}/{args.n_chains}",
            "",
        ]
        search = section["search"]
        if search:
            lines += [
                f"Ladder search: {float(search['search_seconds']):.1f}s, {search['rounds']} rounds, "
                f"{search['n_temperatures']} temperatures, peak workers={search['peak_worker_processes']}, "
                f"{search['n_search_points']} search points; stored ladder has "
                f"{search['stored_n_temperatures'] or '-'} temperatures, identical={search['matches_stored_ladder']}.",
                "",
            ]
        lines += [
            "| method | temperatures | parallel CPUs | mean s/chain | std s/chain | x default |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in section["summary"]:
            lines.append(
                f"| {row['label']} | {row['n_temperatures'] or '-'} | {row['parallel_cpus']} | "
                f"{row['mean_seconds_per_chain']} | {row['std_seconds_per_chain'] or '-'} | "
                f"{row['relative_to_default']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(args) -> int:
    output_dir = (args.output_dir or SCRIPT_DIR).expanduser().resolve()
    allocated, affinity = allocated_cpu_count(), affinity_cpu_count()
    available = CONFIGS + SERIAL_CONFIGS if args.with_serial else CONFIGS
    configs = [c for c in available if c[0] in args.labels]
    print(f"Output directory: {output_dir}")
    print(f"Allocated CPUs (PBS_NCPUS etc.)={allocated}; process affinity CPUs={affinity}; os.cpu_count()={os.cpu_count()}")

    prepared = []
    for name in args.datasets:
        split = load_split(name, args.run_id)
        try:
            temperatures, ladder_dir = load_temperature_ladder(name, args.run_id)
        except FileNotFoundError:
            if not args.search_ladder:
                raise
            temperatures, ladder_dir = np.array([]), None
        pt_workers = resolve_pt_workers(None if args.pt_n_jobs == 0 else args.pt_n_jobs, len(temperatures))
        print(
            f"{name}: n={split['n_rows']} train={len(split['train_idx'])} test={len(split['test_idx'])}; "
            f"split checked against {split['checked_against'] or 'nothing (no stored indices found)'}; "
            f"{len(temperatures)} temperatures from {ladder_dir}; parallel PT workers={pt_workers}",
            flush=True,
        )
        limit = min(x for x in (allocated, affinity) if x is not None) if (allocated or affinity) else None
        if limit is not None and pt_workers > limit:
            print(f"WARNING: {name} needs {pt_workers} PT workers but only {limit} CPUs are available.", flush=True)
        prepared.append((name, split, temperatures, pt_workers))

    if args.dry_run:
        for name, _split, _t, _w in prepared:
            for chain in range(args.n_chains):
                print(f"  {name} chain {chain} seed={args.chain_seed + chain}: " + ", ".join(label for label, _, _ in configs))
        print("Dry run complete; nothing fitted or written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    failures = 0
    for name, split, temperatures, pt_workers in prepared:
        tag = SOURCES[name].tag
        per_chain_path = output_dir / f"{tag}_run{args.run_id:03d}_per_chain.csv"
        summary_path = output_dir / f"{tag}_run{args.run_id:03d}_summary.csv"
        rows: list[dict[str, Any]] = []
        print(f"\n=== {tag}: {len(temperatures)} temperatures, parallel PT workers={pt_workers} ===", flush=True)
        run_numba_warmup(args, split["X_train"], split["y_train"],
                         np.asarray(initial_temperatures(args)) if args.search_ladder else temperatures,
                         SOURCES[name].cfg)
        if args.search_ladder:
            gc.collect()
            searched = timed_ladder_search(name, split, args)
            stored = temperatures
            temperatures = searched["temperatures"]
            pt_workers = resolve_pt_workers(None if args.pt_n_jobs == 0 else args.pt_n_jobs, len(temperatures))
            search_row = {
                "dataset": tag, "run_id": args.run_id,
                "search_seconds": f"{searched['search_seconds']:.3f}", "rounds": searched["rounds"],
                "n_temperatures": len(temperatures),
                "peak_worker_processes": searched["peak_worker_processes"],
                "allocated_cpus": allocated if allocated is not None else "",
                "affinity_cpus": affinity if affinity is not None else "",
                "n_search_points": searched["n_search_points"],
                "stored_n_temperatures": len(stored) if len(stored) else "",
                "matches_stored_ladder": (len(stored) == len(temperatures) and bool(np.allclose(stored, temperatures)))
                                         if len(stored) else "",
                "temperatures": json.dumps([round(float(t), 10) for t in temperatures]),
            }
            _write_csv(output_dir / f"{tag}_run{args.run_id:03d}_ladder_search.csv", SEARCH_FIELDS, [search_row])
            print(f"[{tag}] ladder search {searched['search_seconds']:.1f}s, {searched['rounds']} rounds, "
                  f"{len(temperatures)} temperatures, peak workers={searched['peak_worker_processes']}, "
                  f"identical to stored ladder: {search_row['matches_stored_ladder']}; "
                  f"parallel PT workers={pt_workers}", flush=True)

        for chain in range(args.n_chains):
            seed = args.chain_seed + chain
            for label, method, backend in configs:
                gc.collect()
                model = None
                try:
                    model = make_model(method, backend, args, temperatures, pt_workers,
                                       seed, SOURCES[name].cfg)
                    data = model.preprocessor.fit_transform(split["X_train"], split["y_train"])
                    measured = timed_fit(model, data, quietly=not args.show_progress)
                except Exception:
                    failures += 1
                    traceback.print_exc()
                    if args.fail_fast:
                        return 1
                    continue
                finally:
                    del model
                    gc.collect()
                row = {
                    "dataset": tag, "run_id": args.run_id, "chain": chain, "seed": seed, "label": label,
                    "method": method, "backend": backend,
                    "n_temperatures": len(temperatures) if backend != "none" else "",
                    "pt_workers_requested": pt_workers if backend == "parallel" else 1,
                    "allocated_cpus": allocated if allocated is not None else "",
                    "affinity_cpus": affinity if affinity is not None else "",
                    "n_rows": split["n_rows"], "n_train": len(split["train_idx"]), "n_test": len(split["test_idx"]),
                    **measured,
                }
                rows.append(row)
                print(
                    f"[{tag} chain {chain}] {label:<22} {measured['elapsed_seconds']:10.1f}s  "
                    f"peak workers={measured['peak_worker_processes']}",
                    flush=True,
                )
                _write_csv(per_chain_path, PER_CHAIN_FIELDS,
                           [{**r, "elapsed_seconds": f"{r['elapsed_seconds']:.3f}"} for r in rows])
                _write_csv(summary_path, SUMMARY_FIELDS,
                           summarize(rows, search_row if args.search_ladder else None))
                write_markdown(output_dir / "summary.md", sections_from_dir(output_dir), args)

    print("\n" + (output_dir / "summary.md").read_text(encoding="utf-8"))
    return 1 if failures else 0


def main() -> int:
    global LADDER_STORE
    args = parse_args()
    LADDER_STORE = args.ladder_store.expanduser().resolve() if args.ladder_store else None
    return run_benchmark(args)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
