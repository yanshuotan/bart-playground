#!/usr/bin/env python3
"""Feature use: do independent chains agree on which predictors the trees split on?

Each stored draw carries the ensemble's split counts per predictor,
``vars_histogram`` with shape (chain, draw, predictor). Two per-draw summaries
follow the package's own definitions (``feature_inclusion_frequency`` with
``normalize='per_draw'`` and ``feature_inclusion_probability``):

* the split-use share of predictor v, its count divided by the draw's total
  number of splits;
* the inclusion indicator, whether predictor v is used at least once.

For DART the splitting proportions satisfy
``s | counts ~ Dirichlet(a/d + c_1, ..., a/d + c_d)``, so the stored counts
give the conditional posterior mean selection weight ``(a/d + c_v)/(a + sum c)``
for every draw; ``a = 1`` in every DART setting.

Per (dataset, run, record), with the common 3,000-draw warm-up:

* rank-normalized split/folded R-hat of each predictor's share series, across
  the four chains, summarized over predictors (median, 90th percentile, max);
* chain-level posterior mean shares and inclusion probabilities, and the
  largest between-chain range of each across predictors;
* the number of predictors whose chain-level inclusion probability lies on
  both sides of 0.5 (a median-probability selection that depends on the chain);
* on the Friedman designs, where x1--x5 carry the signal, the share of splits
  on those five predictors.

Records: the four production methods, the stride-1,000 long Default record and
the contiguous long-Default tail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG))
from core import ResultCache, load_with_shape  # noqa: E402
from section7_costs import DISPLAY, TIMING_TAG  # noqa: E402

# Bump when a definition below changes, so cached entries are recomputed.
CACHE_VERSION = 1
METHODS = ("default", "mtmh", "default_pt", "mtmh_pt", "default_long", "default_long_tail")
METHOD_NAMES = {"default": "Default", "mtmh": "MTMH", "default_pt": "Default+PT",
                "mtmh_pt": "MTMH+PT", "default_long": "Default (long)",
                "default_long_tail": "Default (tail)"}
DART_ALPHA = 1.0
# Friedman #1 uses x1--x5; the remaining inputs are independent nuisance variables.
FRIEDMAN = {"fixed100_Friedman", "fixed100_FriedmanSparseDir_p20",
            "fixed100_FriedmanSparseDir_p100", "fixed100_FriedmanSparseDir_p200"}
ACTIVE = 5


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(TIMING_TAG))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute every (dataset, run, record) and rewrite its cache entry.")
    return parser.parse_args(supplied)


def histogram_path(store_root: Path, dataset: str, run: int, method: str) -> Path | None:
    hits = sorted((store_root / dataset / "vars_histogram").glob(
        f"*__run{run:03d}__{method}__vars_histogram.csv"))
    return hits[0] if hits else None


def rhat_over_predictors(values: np.ndarray) -> np.ndarray:
    """Rank R-hat per predictor for (chain, draw, predictor); NaN where a series is constant."""
    data = xr.DataArray(values, dims=("chain", "draw", "predictor"), name="x")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(az.rhat(data, method="rank")["x"].values, dtype=float)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {f"{prefix}_median": np.nan, f"{prefix}_q90": np.nan, f"{prefix}_max": np.nan}
    return {f"{prefix}_median": float(np.median(finite)),
            f"{prefix}_q90": float(np.quantile(finite, 0.9)),
            f"{prefix}_max": float(np.max(finite))}


def feature_use(counts: np.ndarray, dataset: str) -> dict[str, float]:
    """All feature-use summaries for one record's (chain, draw, predictor) counts."""
    counts = np.asarray(counts, dtype=float)
    chains, draws, predictors = counts.shape
    totals = counts.sum(axis=2, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("a retained draw has no splits")
    share = counts / totals
    included = (counts > 0).astype(float)

    chain_share = share.mean(axis=1)                      # (chain, predictor)
    chain_pip = included.mean(axis=1)                     # (chain, predictor)
    share_range = chain_share.max(axis=0) - chain_share.min(axis=0)
    pip_range = chain_pip.max(axis=0) - chain_pip.min(axis=0)
    straddles = (chain_pip.max(axis=0) >= 0.5) & (chain_pip.min(axis=0) < 0.5)

    out = {"predictors": predictors, "draws": draws,
           **summarize(rhat_over_predictors(share), "share_rhat"),
           "share_range_max": float(share_range.max()),
           "share_range_median": float(np.median(share_range)),
           "pip_range_max": float(pip_range.max()),
           "pip_straddle_count": int(straddles.sum()),
           "pip_constant_count": int(np.sum(np.all(included == included[:1, :1, :], axis=(0, 1))))}

    if dataset.startswith("fixed100_FriedmanSparseDir"):
        weight = (DART_ALPHA / predictors + counts) / (DART_ALPHA + totals)
        out.update(summarize(rhat_over_predictors(weight), "weight_rhat"))
    if dataset in FRIEDMAN:
        active = share[:, :, :ACTIVE].sum(axis=2)          # (chain, draw)
        chain_active = active.mean(axis=1)
        out["active_share_mean"] = float(chain_active.mean())
        out["active_share_range"] = float(chain_active.max() - chain_active.min())
        data = xr.DataArray(active, dims=("chain", "draw"), name="x")
        out["active_share_rhat"] = float(az.rhat(data, method="rank")["x"])
        nuisance_pip = chain_pip[:, ACTIVE:]
        out["nuisance_pip_max"] = float(nuisance_pip.max()) if nuisance_pip.size else np.nan
    return out


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache("section7_feature_use", {"version": CACHE_VERSION, "burn": args.burn,
                                                 "dart_alpha": DART_ALPHA},
                        enabled=not args.no_cache)
    rows = []
    for dataset in args.datasets:
        for run in args.runs:
            for method in METHODS:
                path = histogram_path(args.store_root, dataset, run, method)
                if path is None:
                    continue
                key = (dataset, f"run{run:03d}", method)
                result = cache.get(key, [path])
                if result is None:
                    counts = load_with_shape(path)[:, args.burn:, :]
                    if counts.shape[1] != 7000:
                        raise ValueError(f"{path.name}: {counts.shape[1]} retained draws, expected 7000")
                    result = feature_use(counts, dataset)
                    cache.put(key, [path], result)
                rows.append({"dataset": dataset, "setting": DISPLAY.get(dataset, dataset),
                             "run": run, "method": METHOD_NAMES[method], **result})
            print(f"[feature use] {dataset} run {run:03d}", flush=True)
    print(f"[feature use] cache: {cache.hits} reused, {cache.misses} computed", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.table_dir / "feature_use_run_level.csv", index=False)
    numeric = frame.drop(columns=["run"]).select_dtypes("number").columns
    summary = frame.groupby(["dataset", "setting", "method"], sort=False)[list(numeric)].agg(["mean", "min", "max"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(args.table_dir / "feature_use_summary.csv", index=False)
    print(f"[feature use] wrote tables to {args.table_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
