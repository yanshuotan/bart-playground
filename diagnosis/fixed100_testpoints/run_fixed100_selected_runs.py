from __future__ import annotations

"""
Run selected fixed100 run IDs without rerunning earlier run IDs.

Use case:
  - You already have run000/run001 and want to add run002-run004.
  - This wrapper keeps the original fixed100 pipeline intact, but monkey-patches
    make_fixed100_splits at runtime so run_fixed100_dataset only receives the
    requested run IDs.

This file does not modify experiment_fixed100.py or run_fixed100.py.
Put it in diagnosis/fixed100_testpoints/ and run from that directory.
"""

import argparse
import threading
import time
from pathlib import Path

import experiment_fixed100 as exp
from run_fixed100 import (
    DATASET_CONFIGS,
    GLOBAL_BASE_CHAIN_SEED,
    GLOBAL_BASE_TRAIN_SEED,
    GLOBAL_FIXED_TEST_SEED,
    load_dataset,
)


def _parse_run_ids(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                start = int(a)
                end = int(b)
                if end < start:
                    raise ValueError(f"Bad run range: {part}")
                out.extend(range(start, end + 1))
            else:
                out.append(int(part))
    out = sorted(set(out))
    if not out:
        raise ValueError("No run IDs were provided.")
    if min(out) < 0:
        raise ValueError("Run IDs must be non-negative.")
    return out


class SelectedRunsPatch:
    """Temporarily replace exp.make_fixed100_splits to return only requested runs."""

    def __init__(self, selected_run_ids: list[int]):
        self.selected_run_ids = sorted(set(int(x) for x in selected_run_ids))
        self.selected_set = set(self.selected_run_ids)
        self.original = exp.make_fixed100_splits

    def __enter__(self):
        selected_set = self.selected_set
        original = self.original
        required_n_runs = max(self.selected_run_ids) + 1

        def make_selected_splits(
            X,
            y,
            *,
            n_runs: int,
            n_fixed_test_points: int = 100,
            train_fraction: float = 0.75,
            fixed_test_seed: int = GLOBAL_FIXED_TEST_SEED,
            base_train_seed: int = GLOBAL_BASE_TRAIN_SEED,
        ):
            # Generate all splits up to max selected run so that run_id -> seed mapping
            # remains exactly the same as in the original pipeline.
            all_splits = original(
                X,
                y,
                n_runs=max(n_runs, required_n_runs),
                n_fixed_test_points=n_fixed_test_points,
                train_fraction=train_fraction,
                fixed_test_seed=fixed_test_seed,
                base_train_seed=base_train_seed,
            )
            selected = [s for s in all_splits if int(s["run_id"]) in selected_set]
            if len(selected) != len(selected_set):
                got = sorted(int(s["run_id"]) for s in selected)
                raise RuntimeError(f"Expected run IDs {sorted(selected_set)}, got {got}")
            return selected

        exp.make_fixed100_splits = make_selected_splits
        return self

    def __exit__(self, exc_type, exc, tb):
        exp.make_fixed100_splits = self.original
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run only selected fixed100 run IDs, e.g. run002-run004."
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_CONFIGS), default=["abalone", "concrete", "friedman"])
    parser.add_argument("--run-ids", nargs="+", required=True, help="Run IDs to execute, e.g. 2 3 4 or 2-4")
    parser.add_argument("--n-chains", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--store-dir", type=str, default="store_selected_runs")
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fixed-test-seed", type=int, default=GLOBAL_FIXED_TEST_SEED)
    parser.add_argument("--base-train-seed", type=int, default=GLOBAL_BASE_TRAIN_SEED)
    parser.add_argument("--base-chain-seed", type=int, default=GLOBAL_BASE_CHAIN_SEED)
    parser.add_argument("--short-ndpost", type=int, default=2000)
    parser.add_argument("--short-nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--long-chunk-size", type=int, default=10000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 100.0])
    parser.add_argument("--ladder-target-rate", type=float, default=0.4)
    parser.add_argument("--ladder-max-rounds", type=int, default=10)
    parser.add_argument("--ladder-ndpost", type=int, default=500)
    parser.add_argument("--ladder-nskip", type=int, default=500)
    parser.add_argument("--ladder-repeats", type=int, default=3)
    parser.add_argument("--ladder-search-points", type=int, default=1000)
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--skip-long", action="store_true", help="Run only the four short-chain methods.")
    parser.add_argument("--skip-short", action="store_true", help="Run only default_long.")
    parser.add_argument("--store-preds", action="store_true", default=True)
    args = parser.parse_args()

    if args.skip_long and args.skip_short:
        parser.error("--skip-long and --skip-short cannot be used together.")

    run_ids = _parse_run_ids(args.run_ids)
    n_runs_for_metadata = max(run_ids) + 1

    here = Path(__file__).resolve().parent
    store_root = here / args.store_dir
    print(f"Writing outputs to: {store_root}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Selected run IDs: {run_ids}", flush=True)
    print(f"n_runs passed to original pipeline: {n_runs_for_metadata}", flush=True)
    print(f"n_chains={args.n_chains}, n_jobs={args.n_jobs}", flush=True)
    print(f"skip_short={args.skip_short}, skip_long={args.skip_long}", flush=True)
    print(f"temperatures={args.temperatures}", flush=True)

    start = time.time()
    with SelectedRunsPatch(run_ids):
        for name in args.datasets:
            cfg = DATASET_CONFIGS[name]
            X, y = load_dataset(name)
            print(f"\n=== DATASET {name} ===", flush=True)
            print(f"X={X.shape}, y={y.shape}", flush=True)
            print(
                f"long_ndpost={cfg['long_ndpost']}, long_store_every={cfg['long_store_every']}",
                flush=True,
            )
            exp.run_fixed100_dataset(
                X=X,
                y=y,
                dataset_tag=cfg["dataset_tag"],
                store_dir=store_root,
                n_runs=n_runs_for_metadata,
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
                ladder_target_rate=args.ladder_target_rate,
                ladder_max_rounds=args.ladder_max_rounds,
                ladder_ndpost=args.ladder_ndpost,
                ladder_nskip=args.ladder_nskip,
                ladder_repeats=args.ladder_repeats,
                ladder_search_points=args.ladder_search_points,
                swap_interval=args.swap_interval,
                multi_tries=args.multi_tries,
                store_preds=args.store_preds,
                progress_print=True,
                run_short=not args.skip_short,
                run_long=not args.skip_long,
                dirichlet_prior=cfg.get("dirichlet_prior", False),
                s_alpha=float(cfg.get("s_alpha", 1.0)),
            )

    runtime_min = (time.time() - start) / 60
    print(f"DONE selected fixed100 runs in {runtime_min:.2f} min", flush=True)


if __name__ == "__main__":
    main()
