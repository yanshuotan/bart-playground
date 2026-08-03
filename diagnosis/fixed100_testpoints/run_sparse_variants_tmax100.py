#!/usr/bin/env python3
"""Run the controlled sparse variants through the existing short-only pipeline.

This file deliberately has no long-chain option.  It imports the already-tested
Tmax100 harmonic ladder factory from ``run_fixed100_tmax100_search_v3.py`` and
the live output pipeline from ``experiment_fixed100.py``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import time
from pathlib import Path

import numpy as np

import experiment_fixed100 as exp
from run_fixed100 import (
    GLOBAL_BASE_CHAIN_SEED,
    GLOBAL_BASE_TRAIN_SEED,
    GLOBAL_FIXED_TEST_SEED,
    _load_generator_dataset,
    load_dataset,
)
from run_fixed100_tmax100_search_v3 import make_capped_harmonic_ladder_search
from sparse_variants import SPECS, generate_sparse_variant, smoke_check


def fail_fast_live_pipeline_check() -> None:
    """Stop before fitting if the live server pipeline lacks required fixes."""
    source = inspect.getsource(exp)
    required = {
        "prediction chains": '"preds"',
        "predictive draws": '"pred_samples"',
        "R2 output": "subsample_r2",
    }
    missing = [label for label, token in required.items() if token not in source]
    short_summary_source = inspect.getsource(exp.summarize_model_outputs)
    short_save_source = inspect.getsource(exp.save_short_run)
    if 'result["splitting_weights"]' not in short_summary_source:
        missing.append("short-chain Dirichlet splitting-weight extraction")
    if 'r[method]["splitting_weights"]' not in short_save_source:
        missing.append("short-chain Dirichlet splitting-weight save")

    bad_seed_patterns = {
        "fixed test seed is overridden by a global":
            r"fixed_test_seed\s*=\s*GLOBAL_FIXED_TEST_SEED",
        "base train seed is overridden by a global":
            r"base_train_seed\s*=\s*GLOBAL_BASE_TRAIN_SEED",
    }
    bad = [label for label, pattern in bad_seed_patterns.items() if re.search(pattern, source)]
    if missing or bad:
        details = {"missing_required_outputs": missing, "bad_seed_patterns": bad}
        raise RuntimeError("LIVE PIPELINE PRE-FLIGHT FAILED:\n" + json.dumps(details, indent=2))


def repo_nested_friedman1_generator(
    n_samples: int,
    n_features: int,
    seed: int,
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nest p=20/200 around the exact live-repo p=100 Friedman dataset.

    This deliberately keeps x1,...,x5 and y identical across p.  Generating
    separate (n, p) random matrices with the same seed would change row-wise
    active-feature values when p changes, confounding sparsity with a new data
    realization.
    """
    if n_features < 5:
        raise ValueError("Friedman #1 requires at least five features")
    cfg = {
        "n_samples": int(n_samples),
        "n_features": 100,
        "seed": int(seed),
        "noise": float(noise_sd),
    }
    center_X, center_y = _load_generator_dataset(cfg, "friedman1")
    if n_features <= 100:
        return center_X[:, :n_features].copy(), center_y.copy()

    # Append only response-independent U(0,1) nuisance variables.  A separate
    # seed makes this block reproducible without perturbing the p=100 center.
    nuisance_rng = np.random.default_rng(int(seed) + 200_000)
    extra = nuisance_rng.uniform(
        0.0,
        1.0,
        size=(int(n_samples), int(n_features) - 100),
    )
    return np.column_stack([center_X, extra]), center_y.copy()


def verify_friedman_center_matches_repo() -> None:
    """Ensure the injected p=100 generator exactly reproduces the completed run."""

    repo_X, repo_y = load_dataset("friedman_sparse_dir")
    generated_X, generated_y = repo_nested_friedman1_generator(
        2000, 100, 42, 1.0
    )
    x_match = repo_X.shape == generated_X.shape and np.array_equal(
        repo_X, generated_X
    )
    y_match = repo_y.shape == generated_y.shape and np.array_equal(
        np.asarray(repo_y).reshape(-1), np.asarray(generated_y).reshape(-1)
    )
    if not (x_match and y_match):
        raise RuntimeError(
            "Injected repo DataGenerator does not exactly reproduce "
            "load_dataset('friedman_sparse_dir'). "
            f"x_match={x_match}, y_match={y_match}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=sorted(SPECS), required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate live code and generated datasets, print settings, then exit without fitting.",
    )
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--noise-sd", type=float, default=1.0)
    parser.add_argument("--n-fixed-test-points", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--fixed-test-seed", type=int, default=GLOBAL_FIXED_TEST_SEED)
    parser.add_argument("--base-train-seed", type=int, default=GLOBAL_BASE_TRAIN_SEED)
    parser.add_argument("--base-chain-seed", type=int, default=GLOBAL_BASE_CHAIN_SEED)
    parser.add_argument("--short-ndpost", type=int, default=10_000)
    parser.add_argument("--short-nskip", type=int, default=0)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--s-alpha", type=float, default=1.0)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.short_ndpost != 10_000:
        raise ValueError("Formal sparse attempts require --short-ndpost 10000 per chain")
    if args.n_chains != 4:
        raise ValueError("Formal sparse attempts require exactly 4 chains")
    if args.ladder_tmax != 100.0 or args.ladder_init_size != 10:
        raise ValueError("Formal sparse attempts require np.geomspace(1, 100, 10)")
    if args.s_alpha <= 0:
        raise ValueError("s_alpha must be positive")

    smoke_check(friedman_generator=repo_nested_friedman1_generator)
    fail_fast_live_pipeline_check()
    if any(name.startswith("friedman_") for name in args.variants):
        verify_friedman_center_matches_repo()

    if args.preflight_only:
        print("LIVE PIPELINE AND GENERATOR PRE-FLIGHT PASSED", flush=True)
        for variant in args.variants:
            X, y, metadata = generate_sparse_variant(
                variant,
                n_samples=args.n_samples,
                seed=args.data_seed,
                noise_sd=args.noise_sd,
                friedman_generator=repo_nested_friedman1_generator,
            )
            print(
                variant,
                f"X={X.shape}",
                f"y={y.shape}",
                f"SNR={metadata['target_snr_variance_ratio']:.6g}",
                flush=True,
            )
        print("NO MODEL WAS FITTED", flush=True)
        return

    max_temperatures = (
        None if args.ladder_max_temperatures <= 0 else int(args.ladder_max_temperatures)
    )
    exp.quick_ladder_search = make_capped_harmonic_ladder_search(max_temperatures)
    initial_temperatures = tuple(
        np.geomspace(1.0, args.ladder_tmax, args.ladder_init_size).tolist()
    )

    here = Path(__file__).resolve().parent
    store_root = here / args.store_dir
    store_root.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {store_root}", flush=True)
    print(f"SHORT ONLY: n_runs={args.n_runs}, n_chains={args.n_chains}, n_jobs={args.n_jobs}", flush=True)
    print(f"short_ndpost={args.short_ndpost} PER CHAIN; short_nskip={args.short_nskip}", flush=True)
    print(f"initial_temperatures={np.round(initial_temperatures, 6).tolist()}", flush=True)
    print(
        "ladder="
        f"Tmax={args.ladder_tmax}, cap={max_temperatures or 'disabled'}, "
        f"target={args.ladder_target_rate}, rounds={args.ladder_max_rounds}, "
        f"repeats={args.ladder_repeats}, nskip={args.ladder_nskip}, "
        f"ndpost={args.ladder_ndpost}, search_points={args.ladder_search_points}",
        flush=True,
    )

    total_start = time.time()
    for variant in args.variants:
        spec = SPECS[variant]
        X, y, generator_metadata = generate_sparse_variant(
            variant,
            n_samples=args.n_samples,
            seed=args.data_seed,
            noise_sd=args.noise_sd,
            friedman_generator=repo_nested_friedman1_generator,
        )
        generator_metadata["s_alpha"] = float(args.s_alpha)
        if variant.startswith("friedman_"):
            generator_metadata["generator_backend"] = (
                "repo_DataGenerator_p100_center_with_nested_feature_sets"
            )
            generator_metadata["nested_design_rule"] = (
                "p20=X100[:,:20]; p200=[X100, 100 deterministic independent U(0,1) nuisances]; y unchanged"
            )
        print(f"\n=== SPARSE VARIANT {variant} ===", flush=True)
        print(spec.description, flush=True)
        print(f"X={X.shape}, y={y.shape}", flush=True)

        start = time.time()
        exp.run_fixed100_dataset(
            X=X,
            y=y,
            dataset_tag=spec.dataset_tag,
            store_dir=store_root,
            n_runs=args.n_runs,
            n_chains=args.n_chains,
            n_jobs=args.n_jobs,
            n_trees=args.n_trees,
            short_ndpost=args.short_ndpost,
            short_nskip=args.short_nskip,
            long_ndpost=1,
            long_store_every=1,
            long_chunk_size=1,
            n_fixed_test_points=args.n_fixed_test_points,
            train_fraction=args.train_fraction,
            fixed_test_seed=args.fixed_test_seed,
            base_train_seed=args.base_train_seed,
            base_chain_seed=args.base_chain_seed,
            temperatures=initial_temperatures,
            ladder_target_rate=args.ladder_target_rate,
            ladder_max_rounds=args.ladder_max_rounds,
            ladder_ndpost=args.ladder_ndpost,
            ladder_nskip=args.ladder_nskip,
            ladder_repeats=args.ladder_repeats,
            ladder_search_points=args.ladder_search_points,
            swap_interval=args.swap_interval,
            multi_tries=args.multi_tries,
            store_preds=True,
            progress_print=True,
            run_short=True,
            run_long=False,
            dirichlet_prior=True,
            s_alpha=args.s_alpha,
        )

        metadata_dir = store_root / spec.dataset_tag / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with (metadata_dir / f"{spec.dataset_tag}__generator_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(generator_metadata, handle, indent=2, sort_keys=True)
        print(
            f"DONE {variant} in {(time.time() - start) / 60:.2f} min",
            flush=True,
        )

    print(
        f"DONE ALL SHORT-ONLY SPARSE VARIANTS in {(time.time() - total_start) / 60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
