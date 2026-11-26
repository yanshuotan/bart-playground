#!/usr/bin/env python
"""Particle Gibbs with the optimal proposal against bootstrap PG and the paper's methods.

Reads the paper store (default, default_pt, mtmh, mtmh_pt, default_long) and
the store of the pg_opt trial, which holds default, pg and pg_opt fitted on
the same node. The Default draws there must equal the paper store's; Default's
fit time is the cost baseline. Chain statistics are those of section 7, on the
same post-burn draws.

    python diagnosis/pg/compare_pg_opt.py --opt-store /scratch/$USER/bart_store_pgopt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG / "paper"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import load_with_shape, worst_direction_summary  # noqa: E402
from section7_methods import (PREDICTIVE_NOISE_SEED, chain_geometry, full_draw_rhat,  # noqa: E402
                              full_target_vector, predictive_baselines, predictive_metrics,
                              training_targets, y_test)

PAPER = ("default", "default_pt", "mtmh", "mtmh_pt")
PG = ("pg", "pg_opt")
NAMES = {"default": "Default", "default_pt": "Default+PT", "mtmh": "MTMH", "mtmh_pt": "MTMH+PT",
         "pg": "PG", "pg_opt": "PG-opt"}
COLORS = {"default": "#1b9e77", "pg": "#7570b3", "pg_opt": "#000000"}
DATASETS = ("fixed100_Abalone", "fixed100_SeoulBike")


def stored(store: Path, dataset: str, run: int, method: str, kind: str) -> np.ndarray:
    hits = sorted((store / dataset / kind).glob(f"*__run{run:03d}__{method}__{kind}.csv"))
    if not hits:
        raise FileNotFoundError(f"no {kind} for {method} in {store / dataset} run {run}")
    return load_with_shape(hits[0])


def predictions(store: Path, dataset: str, run: int, method: str, burn: int = 0) -> np.ndarray:
    """(chain, draw, test point) predictions after `burn` draws."""
    return stored(store, dataset, run, method, "preds").transpose(0, 2, 1)[:, burn:, :]


def residual_variances(store: Path, dataset: str, run: int, method: str, burn: int = 0) -> np.ndarray:
    """Stored eps_sigma2, (chain, draw), after `burn` draws."""
    raw = stored(store, dataset, run, method, "sigmas")
    return np.asarray(raw, dtype=float).reshape(raw.shape[0], -1)[:, burn:]


def chain_stats(args, store: Path, dataset: str, run: int, method: str, truth, baseline, y_scale) -> dict:
    post = predictions(store, dataset, run, method, args.short_burn)
    sig = residual_variances(store, dataset, run, method, args.short_burn)
    rmse, crps = predictive_metrics(post, sig, truth, y_scale, PREDICTIVE_NOISE_SEED + run)
    worst = worst_direction_summary(post, args.ridge_fraction)
    return {**full_draw_rhat(post), **chain_geometry(post), "worst_lambda_max": worst["lambda_max"],
            "rmse": rmse, "crps": crps,
            "relative_rmse": rmse / baseline["naive_rmse"],
            "relative_crps": crps / baseline["climatology_crps"]}


def run_inputs(args, dataset: str, run: int):
    full = full_target_vector(args.store_root, dataset)
    truth = y_test(args.store_root, dataset, run)
    train = training_targets(args.store_root, dataset, run, full, truth)
    return truth, predictive_baselines(train, truth), float(np.max(train) - np.min(train))


def timing_row(args, dataset: str, run: int) -> dict:
    sec = {m: float(stored(args.opt_store, dataset, run, m, "fit_seconds").mean())
           for m in ("default", *PG)}
    same = (np.array_equal(residual_variances(args.opt_store, dataset, run, "default"),
                           residual_variances(args.store_root, dataset, run, "default"))
            and np.array_equal(predictions(args.opt_store, dataset, run, "default"),
                               predictions(args.store_root, dataset, run, "default")))
    row = {"dataset": dataset, "run": run, "default_identical_to_store": bool(same)}
    for m in PG:
        changes = stored(args.opt_store, dataset, run, m, "tree_changes")[:, args.short_burn:]
        row[f"{m}_seconds"] = sec[m]
        row[f"{m}_cost_ratio"] = sec[m] / sec["default"]
        row[f"{m}_trees_changed"] = float(changes.mean())
    row["default_seconds"] = sec["default"]
    return row


def sigma2_figure(args, path: Path):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 2.9), squeeze=False)
    edges = np.unique(np.geomspace(1, 10_000, 40).astype(int))
    centers = [np.sqrt(a * b) for a, b in zip(edges[:-1], edges[1:])]
    for ax, dataset in zip(axes.ravel(), DATASETS):
        long = residual_variances(args.store_root, dataset, 0, "default_long", args.long_burn)
        lo, hi = (long.mean() - long.std()) * 1e3, (long.mean() + long.std()) * 1e3
        top = hi + 10 * (hi - lo)
        for m in ("default", *PG):
            trace = residual_variances(args.opt_store, dataset, 0, m).mean(axis=0)
            values = [trace[a - 1:b].mean() * 1e3 for a, b in zip(edges[:-1], edges[1:])]
            # Show every curve from iteration 10 on.
            top = max(top, max(v for x, v in zip(centers, values) if x >= 10))
            ax.plot(centers, values, color=COLORS[m], lw=1.6 if m == "pg_opt" else 1.0, label=NAMES[m])
        ax.axhspan(lo, hi, color="0.85", zorder=0, label="long Default ±1 sd")
        ax.set_xscale("log")
        ax.set_ylim(lo - 3 * (hi - lo), 1.03 * top)
        ax.set_title(dataset.replace("fixed100_", ""), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("iteration (log scale)", fontsize=8)
    axes[0, 0].set_ylabel(r"$\sigma^2\times10^{3}$, mean of 4 chains, run 0", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--opt-store", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "opt")
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=3000)
    parser.add_argument("--ridge-fraction", type=float, default=1e-8)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, timing = [], []
    for dataset in DATASETS:
        for run in args.runs:
            inputs = run_inputs(args, dataset, run)
            for method in (*PAPER, *PG):
                store = args.opt_store if method in PG else args.store_root
                rows.append({"dataset": dataset, "run": run, "method": NAMES[method],
                             **chain_stats(args, store, dataset, run, method, *inputs)})
            timing.append(timing_row(args, dataset, run))
            print(f"[pg_opt] {dataset} run {run:03d}", flush=True)
    runs = pd.DataFrame(rows)
    times = pd.DataFrame(timing)
    runs.to_csv(args.out / "pg_opt_runs.csv", index=False)
    times.to_csv(args.out / "pg_opt_timing.csv", index=False)
    metrics = [c for c in runs.columns if c not in ("dataset", "run", "method")]
    summary = runs.groupby(["dataset", "method"], sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(args.out / "pg_opt_summary.csv", index=False)
    times.groupby("dataset").mean(numeric_only=True).drop(columns="run").reset_index().to_csv(
        args.out / "pg_opt_timing_summary.csv", index=False)
    sigma2_figure(args, args.out / "pg_opt_sigma2.png")
    print(f"[pg_opt] outputs in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
