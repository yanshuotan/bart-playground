#!/usr/bin/env python3
"""Long-Default records scored as methods: the `long` and `tail` columns.

Three records of Default are scored with the definitions of sections 6 and 7:

* ``default``: the production chains, iterations 3,001--10,000;
* ``default_long``: the 10^7-iteration chains read through their stride-1,000
  record, the last 7,000 of 10,000 stored draws;
* ``default_long_tail``: the same chains read through the contiguous record of
  their final 10,000 iterations, the last 7,000 of them.

Every record keeps 7,000 draws per chain after the common 3,000-draw warm-up.
Pointwise and segment R-hat, R_B/W, relative RMSE and CRPS come from
section6_methods; ESS, the chain-specific MSE share, the between-chain variance
share and latent-f containment come from section7_costs.score; the scalar
R-hat values apply section7_costs.scalar_summary to the panel-average
prediction and the response-scale noise variance. Results are written as
long_tail_as_method_* (standard BART) and dart_tail_as_method_* (DART).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG))
from core import ResultCache, run_inputs  # noqa: E402
from section6_methods import (PREDICTIVE_NOISE_SEED, chain_geometry,  # noqa: E402
                              full_draw_rhat, full_target_vector, predictions,
                              predictive_baselines, predictive_metrics,
                              residual_variances, training_targets, y_test)
from section7_costs import SYNTHETIC, TIMING_TAG, scalar_summary, score, targets  # noqa: E402

# Bump when a definition below changes, so cached entries are recomputed.
CACHE_VERSION = 1
METHODS = ("default", "default_long", "default_long_tail")
COLUMNS = ["cross_rhat_median", "cross_rhat_q90", "within_rhat_median",
           "between_within_ratio", "relative_rmse", "relative_crps",
           "rel_ess_single", "rel_ess_joint", "between_share",
           "chain_specific_share", "scalar_rhat_panel", "scalar_rhat_noise",
           "relative_rmse_mean_chainwise",
           "cover_f_single", "cover_f_pooled", "width_single", "width_pooled"]


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(TIMING_TAG))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--n-segments", type=int, default=4)
    parser.add_argument("--interval", type=float, default=0.9)
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute every (dataset, run, method) and rewrite its cache entry.")
    return parser.parse_args(supplied)


def method_row(args, dataset: str, run: int, method: str, y: np.ndarray,
               truth: np.ndarray | None, baseline: dict, y_scale: float) -> dict:
    """Every tail-table column for one record of one run."""
    draws = predictions(args.store_root, dataset, run, method, args.burn)
    if draws.shape[1] != 7000:
        raise ValueError(f"{dataset} run {run} {method}: {draws.shape[1]} retained draws, expected 7000")
    variances = residual_variances(args.store_root, dataset, run, method, args.burn)
    rmse, crps = predictive_metrics(draws, variances, y, y_scale, PREDICTIVE_NOISE_SEED + run)
    chain_mse = ((draws.mean(axis=1) - y) ** 2).mean(axis=1)
    costs = score(draws, y, truth, args.interval)
    # The two scalar series of section7_costs.scalar_results, after warm-up.
    scalar = {"panel_mean_f": scalar_summary(draws.mean(axis=2))["rhat"],
              "noise_variance_y": scalar_summary(variances * y_scale**2)["rhat"]}
    row = {
        **{k: v for k, v in full_draw_rhat(draws, args.n_segments).items()
           if k in ("cross_rhat_median", "cross_rhat_q90", "within_rhat_median")},
        "between_within_ratio": chain_geometry(draws)["between_within_ratio"],
        "relative_rmse": rmse / baseline["naive_rmse"],
        "relative_crps": crps / baseline["climatology_crps"],
        "rel_ess_single": costs["rel_ess_bulk_single_median"],
        "rel_ess_joint": costs["rel_ess_bulk_multi_median"],
        "between_share": costs["between_share_median"],
        "chain_specific_share": costs["chain_share_of_mse"],
        "scalar_rhat_panel": scalar["panel_mean_f"],
        "scalar_rhat_noise": scalar["noise_variance_y"],
        "relative_rmse_mean_chainwise": float(np.sqrt(chain_mse).mean()) / baseline["naive_rmse"],
    }
    for key in ("cover_f_single", "cover_f_pooled", "width_single", "width_pooled"):
        row[key] = costs.get(key, np.nan)
    return row


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache("long_tail_tables", {
        "version": CACHE_VERSION, "burn": args.burn, "n_segments": args.n_segments,
        "interval": args.interval, "noise_seed": PREDICTIVE_NOISE_SEED,
    }, enabled=not args.no_cache)

    rows = []
    for dataset in args.datasets:
        full_targets = full_target_vector(args.store_root, dataset)
        for run in args.runs:
            y = y_test(args.store_root, dataset, run)
            truth = None
            if dataset in SYNTHETIC:
                # Rebuild the generated design to recover the noise-free f.
                observed, truth = targets(dataset, args.store_root, run)
                if not np.allclose(observed, y):
                    raise ValueError(f"{dataset} run {run}: rebuilt responses differ from the stored test set")
            train = training_targets(args.store_root, dataset, run, full_targets, y)
            baseline = predictive_baselines(train, y)
            # DefaultPreprocessor min-max scales y on the training targets.
            y_scale = float(np.max(train) - np.min(train))
            for method in METHODS:
                inputs = run_inputs(args.store_root, dataset, run, [method])
                key = (dataset, f"run{run:03d}", method)
                row = cache.get(key, inputs)
                if row is None:
                    row = method_row(args, dataset, run, method, y, truth, baseline, y_scale)
                    cache.put(key, inputs, row)
                rows.append({"dataset": dataset, "run": run, "method": method, **row})
            print(f"[long tail] {dataset} run {run:03d}", flush=True)
    print(f"[long tail] cache: {cache.hits} reused, {cache.misses} computed", flush=True)

    frame = pd.DataFrame(rows)[["dataset", "run", "method", *COLUMNS]]
    dart = frame.dataset.str.startswith("fixed100_FriedmanSparseDir")
    for name, part in (("long_tail_as_method", frame[~dart]), ("dart_tail_as_method", frame[dart])):
        if part.empty:
            continue
        part.to_csv(args.table_dir / f"{name}_run_level.csv", index=False)
        (part.drop(columns="run").groupby(["dataset", "method"], sort=False).mean()
         .reset_index().to_csv(args.table_dir / f"{name}_summary.csv", index=False))
    print(f"[long tail] wrote tables to {args.table_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
