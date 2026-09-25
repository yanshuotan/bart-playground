"""Run the retained fixed-100 BART experiments from one command."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np

import experiment_fixed100 as exp
from fixed100_support import (
    DATASET_CONFIGS,
    GLOBAL_BASE_CHAIN_SEED,
    GLOBAL_BASE_TRAIN_SEED,
    GLOBAL_FIXED_TEST_SEED,
    SPECS,
    SelectedRunsPatch,
    _parse_run_ids,
    fail_fast_live_pipeline_check,
    generate_sparse_variant,
    load_dataset,
    make_capped_harmonic_ladder_search,
    memory_logger,
    repo_nested_friedman1_generator,
    smoke_check,
    verify_friedman_center_matches_repo,
)


def _common_arguments(parser):
    parser.add_argument("--store-dir", default="store", help="Output directory, relative to this script unless absolute.")
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fixed-test-seed", type=int, default=GLOBAL_FIXED_TEST_SEED)
    parser.add_argument("--base-train-seed", type=int, default=GLOBAL_BASE_TRAIN_SEED)
    parser.add_argument(
        "--base-chain-seed",
        type=int,
        default=GLOBAL_BASE_CHAIN_SEED,
        help=(
            f"Chain seed base (default {GLOBAL_BASE_CHAIN_SEED}); chain_seed = this + "
            f"run_id*1000 + chain_id, plus 100000 for long chains. diagnosis/store used "
            f"the default throughout."
        ),
    )
    parser.add_argument("--short-ndpost", type=int, default=10_000)
    parser.add_argument("--short-nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--swap-interval", type=int, default=50)
    parser.add_argument("--multi-tries", type=int, default=10)
    parser.add_argument("--ladder-tmax", type=float, default=100.0)
    parser.add_argument("--ladder-init-size", type=int, default=10)
    parser.add_argument("--ladder-max-temperatures", type=int, default=0)
    parser.add_argument("--ladder-target-rate", type=float, default=0.4)
    parser.add_argument("--ladder-max-rounds", type=int, default=10)
    parser.add_argument("--ladder-ndpost", type=int, default=500)
    parser.add_argument("--ladder-nskip", type=int, default=500)
    parser.add_argument("--ladder-repeats", type=int, default=3)
    parser.add_argument("--ladder-search-points", type=int, default=1000)
    parser.add_argument(
        "--parallel-methods",
        action="store_true",
        help=(
            "Run the four short methods as separate tasks (n_chains*4 in total) "
            "instead of one sequential task per chain. Same draws; raise --n-jobs to use it."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true", help="Validate inputs without fitting or writing results.")


def parse_args(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Fixed-100 BART experiments")
    sub = parser.add_subparsers(dest="experiment", required=True)
    fixed = sub.add_parser("fixed", help="Run the retained named datasets")
    _common_arguments(fixed)
    fixed.add_argument("--datasets", nargs="+", choices=("abalone", "calhousing", "ccpp", "concrete", "friedman", "friedman_sparse_dir", "seoul_bike"), default=["abalone", "concrete", "friedman"])
    fixed.add_argument("--run-ids", nargs="+", help="Only these run IDs, e.g. 2 3 4 or 2-4")
    fixed.add_argument("--ladder", choices=("harmonic", "original"), default="harmonic")
    fixed.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 100.0], help="Initial temperatures when --ladder=original")
    fixed.add_argument("--skip-short", action="store_true")
    fixed.add_argument("--skip-long", action="store_true")
    fixed.add_argument(
        "--long-chunk-size",
        type=int,
        default=10_000,
        help="Long-chain iterations held in memory at once; a memory/speed knob that leaves the draws unchanged.",
    )
    fixed.add_argument("--enable-memory-log", action="store_true")
    fixed.add_argument("--memory-log-interval", type=int, default=60)
    sparse = sub.add_parser("sparse", help="Run the retained p20/p200 sparse variants")
    _common_arguments(sparse)
    sparse.add_argument("--variants", nargs="+", choices=("friedman_p20_k5", "friedman_p200_k5"), required=True)
    sparse.add_argument("--n-samples", type=int, default=2000)
    sparse.add_argument("--data-seed", type=int, default=42)
    sparse.add_argument("--noise-sd", type=float, default=1.0)
    sparse.add_argument("--s-alpha", type=float, default=1.0)
    args = parser.parse_args(supplied)
    if args.experiment == "fixed" and args.ladder == "original":
        if "--short-ndpost" not in supplied:
            args.short_ndpost = 2000
        if args.run_ids:
            if "--n-chains" not in supplied:
                args.n_chains = 1
            if "--n-jobs" not in supplied:
                args.n_jobs = 1
        else:
            if "--n-jobs" not in supplied:
                args.n_jobs = 4
            if "--temperatures" not in supplied:
                args.temperatures = [1.0, 1_000_000.0]
    if args.ladder_init_size < 2 or args.ladder_tmax <= 1.0:
        parser.error("The ladder needs at least two temperatures and Tmax > 1.")
    if args.n_runs < 1 or args.n_chains < 1 or args.n_jobs < 1:
        parser.error("--n-runs, --n-chains, and --n-jobs must be positive.")
    if args.short_ndpost < 1 or args.n_trees < 1:
        parser.error("--short-ndpost and --n-trees must be positive.")
    if args.experiment == "fixed" and args.skip_short and args.skip_long:
        parser.error("--skip-short and --skip-long cannot be used together.")
    if args.experiment == "sparse":
        if args.s_alpha <= 0:
            parser.error("--s-alpha must be positive.")
    return args


def _store_root(args):
    return Path(__file__).resolve().parent / args.store_dir


def _initial_temperatures(args):
    if args.experiment == "fixed" and args.ladder == "original":
        return tuple(args.temperatures)
    return tuple(np.geomspace(1.0, args.ladder_tmax, args.ladder_init_size).tolist())


def _fit_arguments(args, X, y, dataset_tag, store_root, *, long_ndpost, long_store_every,
                   run_short, run_long, dirichlet_prior, s_alpha):
    return dict(
        X=X, y=y, dataset_tag=dataset_tag, store_dir=store_root,
        n_runs=args.n_runs, n_chains=args.n_chains, n_jobs=args.n_jobs,
        n_trees=args.n_trees, short_ndpost=args.short_ndpost,
        short_nskip=args.short_nskip, long_ndpost=long_ndpost,
        long_store_every=long_store_every,
        long_chunk_size=getattr(args, "long_chunk_size", 10_000),
        n_fixed_test_points=args.n_fixed_test_points,
        train_fraction=args.train_fraction, fixed_test_seed=args.fixed_test_seed,
        base_train_seed=args.base_train_seed, base_chain_seed=args.base_chain_seed,
        temperatures=_initial_temperatures(args),
        ladder_target_rate=args.ladder_target_rate,
        ladder_max_rounds=args.ladder_max_rounds,
        ladder_ndpost=args.ladder_ndpost, ladder_nskip=args.ladder_nskip,
        ladder_repeats=args.ladder_repeats,
        ladder_search_points=args.ladder_search_points,
        swap_interval=args.swap_interval, multi_tries=args.multi_tries,
        store_preds=True, progress_print=True, run_short=run_short,
        run_long=run_long, dirichlet_prior=dirichlet_prior,
        s_alpha=float(s_alpha), parallel_methods=args.parallel_methods,
    )


def _use_ladder(args):
    if args.experiment == "sparse" or args.ladder == "harmonic":
        cap = None if args.ladder_max_temperatures <= 0 else args.ladder_max_temperatures
        exp.quick_ladder_search = make_capped_harmonic_ladder_search(cap)


def _run_fixed(args, store_root):
    run_ids = _parse_run_ids(args.run_ids) if args.run_ids else None
    if run_ids:
        args.n_runs = max(run_ids) + 1
    if args.preflight_only:
        for name in args.datasets:
            X, y = load_dataset(name)
            if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
                raise ValueError(f"Invalid shape for {name}: X={X.shape}, y={y.shape}")
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"Non-finite data in {name}")
            print(f"{name}: X={X.shape}, y={y.shape}, tag={DATASET_CONFIGS[name]['dataset_tag']}")
        print("PREFLIGHT PASSED; no model fitted or result written")
        return
    stop_event = None
    mem_thread = None
    if args.enable_memory_log:
        stop_event = threading.Event()
        mem_thread = threading.Thread(
            target=memory_logger,
            args=(store_root / "memory_log.csv", stop_event, args.memory_log_interval),
            daemon=True,
        )
        mem_thread.start()
    try:
        with SelectedRunsPatch(run_ids) if run_ids else nullcontext():
            for name in args.datasets:
                cfg = DATASET_CONFIGS[name]
                X, y = load_dataset(name)
                kwargs = _fit_arguments(
                    args, X, y, cfg["dataset_tag"], store_root,
                    long_ndpost=cfg["long_ndpost"],
                    long_store_every=cfg["long_store_every"],
                    run_short=not args.skip_short, run_long=not args.skip_long,
                    dirichlet_prior=cfg.get("dirichlet_prior", False),
                    s_alpha=cfg.get("s_alpha", 1.0),
                )
                exp.run_fixed100_dataset(**kwargs)
    finally:
        if stop_event is not None:
            stop_event.set()
        if mem_thread is not None:
            mem_thread.join(timeout=5)


def _run_sparse(args, store_root):
    smoke_check(friedman_generator=repo_nested_friedman1_generator)
    fail_fast_live_pipeline_check()
    if any(name.startswith("friedman_") for name in args.variants):
        verify_friedman_center_matches_repo()
    if args.preflight_only:
        for name in args.variants:
            X, y, metadata = generate_sparse_variant(
                name, n_samples=args.n_samples, seed=args.data_seed,
                noise_sd=args.noise_sd,
                friedman_generator=repo_nested_friedman1_generator,
            )
            print(f"{name}: X={X.shape}, y={y.shape}, SNR={metadata['target_snr_variance_ratio']:.6g}")
        print("PREFLIGHT PASSED; no model fitted or result written")
        return
    store_root.mkdir(parents=True, exist_ok=True)
    for name in args.variants:
        spec = SPECS[name]
        X, y, metadata = generate_sparse_variant(
            name, n_samples=args.n_samples, seed=args.data_seed,
            noise_sd=args.noise_sd,
            friedman_generator=repo_nested_friedman1_generator,
        )
        metadata["s_alpha"] = float(args.s_alpha)
        if name.startswith("friedman_"):
            metadata["generator_backend"] = "repo_DataGenerator_p100_center_with_nested_feature_sets"
            metadata["nested_design_rule"] = (
                "p20=X100[:,:20]; p200=[X100, 100 deterministic independent U(0,1) nuisances]; y unchanged"
            )
        kwargs = _fit_arguments(
            args, X, y, spec.dataset_tag, store_root,
            long_ndpost=1, long_store_every=1, run_short=True,
            run_long=False, dirichlet_prior=True, s_alpha=args.s_alpha,
        )
        exp.run_fixed100_dataset(**kwargs)
        metadata_dir = store_root / spec.dataset_tag / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with (metadata_dir / f"{spec.dataset_tag}__generator_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


def main(argv=None):
    args = parse_args(argv)
    store_root = _store_root(args)
    print(f"Writing outputs to: {store_root}", flush=True)
    # Echo the seeds: base_chain_seed is not saved into short_metadata.csv, so
    # the run log is the only place it is recorded.
    print(
        f"Seeds: fixed_test={args.fixed_test_seed} base_train={args.base_train_seed} "
        f"base_chain={args.base_chain_seed}"
        f"{'' if args.base_chain_seed == GLOBAL_BASE_CHAIN_SEED else ' (overridden)'}",
        flush=True,
    )
    _use_ladder(args)
    if args.experiment == "fixed":
        _run_fixed(args, store_root)
    else:
        _run_sparse(args, store_root)


if __name__ == "__main__":
    main()
