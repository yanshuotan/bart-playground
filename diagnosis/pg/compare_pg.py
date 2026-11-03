#!/usr/bin/env python
"""Particle Gibbs against the paper's four short methods and the long Default run.

Reads the paper's store (default, default_pt, mtmh, mtmh_pt, default_long) and
the store written by the pg jobs (pg with C = 10, beside a same-node Default
rerun that serves as the timing baseline and must equal the stored Default
draws). Every chain statistic is the one section 6 of the paper computes, on
the same post-burn draws, and the equal-time table uses the paper's budget
b = 2.5. Also summarises the particle-count study and the Hypercube-D study
when their outputs are given.

    python diagnosis/pg/compare_pg.py --pg-store diagnosis/store_pg \
        --pg-c-stores 2=... 5=... 20=... 50=... --hypercube DIR
"""
from __future__ import annotations

import argparse
import json
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

from core import load_with_shape, pointwise_rhat_summary  # noqa: E402
from section6_methods import (DATASETS, PREDICTIVE_NOISE_SEED, chain_geometry,  # noqa: E402
                              full_draw_rhat, full_target_vector, predictive_baselines,
                              predictive_metrics, training_targets, y_test)
from section6_equal_time import TIMING_TAG, cost_ratios, thinned_tail  # noqa: E402

METHODS = ("default", "default_pt", "mtmh", "mtmh_pt", "pg")
NAMES = {"default": "Default", "default_pt": "Default+PT", "mtmh": "MTMH",
         "mtmh_pt": "MTMH+PT", "pg": "PG"}
COLORS = {"default": "#1b9e77", "default_pt": "#d95f02", "mtmh": "#7570b3",
          "mtmh_pt": "#e7298a", "pg": "#000000"}
SPARSE = ("fixed100_FriedmanSparseDir_p20", "fixed100_FriedmanSparseDir_p100",
          "fixed100_FriedmanSparseDir_p200")
STANDARD = tuple(d for d in DATASETS if d not in SPARSE)
SHORT = {d: d.replace("fixed100_", "").replace("_subsample5000", "").replace("FriedmanSparseDir_", "DART ")
         for d in DATASETS}


def stored(store: Path, dataset: str, run: int, method: str, kind: str) -> np.ndarray:
    hits = sorted((store / dataset / kind).glob(f"*__run{run:03d}__{method}__{kind}.csv"))
    if not hits:
        raise FileNotFoundError(f"no {kind} for {method} in {store / dataset} run {run}")
    return load_with_shape(hits[0])


def draws_of(store: Path, dataset: str, run: int, method: str) -> np.ndarray:
    """(chain, draw, test point) predictions."""
    return stored(store, dataset, run, method, "preds").transpose(0, 2, 1)


def sigmas_of(store: Path, dataset: str, run: int, method: str) -> np.ndarray:
    raw = stored(store, dataset, run, method, "sigmas")
    return np.asarray(raw, dtype=float).reshape(raw.shape[0], -1)


def long_sigma2(args, dataset: str, run: int) -> tuple[float, float]:
    """Mean and sd of the long Default run's noise variance."""
    sig = sigmas_of(args.store_root, dataset, run, "default_long")[:, args.long_burn:]
    return float(sig.mean()), float(sig.std())


def run_inputs(args, dataset: str, run: int):
    """Test targets, predictive baselines and response range of one run."""
    full_targets = full_target_vector(args.store_root, dataset)
    truth = y_test(args.store_root, dataset, run)
    train = training_targets(args.store_root, dataset, run, full_targets, truth)
    return truth, predictive_baselines(train, truth), float(np.max(train) - np.min(train))


def chain_stats(args, store: Path, dataset: str, run: int, method: str, truth, baseline, y_scale) -> dict:
    post = draws_of(store, dataset, run, method)[:, args.short_burn:, :]
    sig = sigmas_of(store, dataset, run, method)[:, args.short_burn:]
    rmse, crps = predictive_metrics(post, sig, truth, y_scale, PREDICTIVE_NOISE_SEED + run)
    return {
        **full_draw_rhat(post, args.n_segments),
        **chain_geometry(post),
        "rmse": rmse, "crps": crps,
        "relative_rmse": rmse / baseline["naive_rmse"],
        "relative_crps": crps / baseline["climatology_crps"],
    }


def timing_and_identity(args, dataset: str, run: int) -> dict:
    pg_t = stored(args.pg_store, dataset, run, "pg", "fit_seconds").ravel()
    def_t = stored(args.pg_store, dataset, run, "default", "fit_seconds").ravel()
    same = (np.array_equal(sigmas_of(args.pg_store, dataset, run, "default"),
                           sigmas_of(args.store_root, dataset, run, "default"))
            and np.array_equal(draws_of(args.pg_store, dataset, run, "default"),
                               draws_of(args.store_root, dataset, run, "default")))
    return {"dataset": dataset, "run": run, "pg_seconds": float(pg_t.mean()),
            "default_seconds": float(def_t.mean()), "pg_cost_ratio": float(pg_t.mean() / def_t.mean()),
            "default_identical_to_store": bool(same)}


def extended_default_draws(args, dataset: str, run: int) -> np.ndarray | None:
    path = args.extended_store / TIMING_TAG[dataset] / "preds" / f"{TIMING_TAG[dataset]}__run{run:03d}__default_ext__preds.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        return archive["predictions"].transpose(0, 2, 1)


def equal_time(args, dataset: str, budget: float, pg_ratio: float) -> list[dict]:
    """Run 0 at a per-chain budget of `budget` Default production runs."""
    ratios = cost_ratios(args.timing_dir, dataset)
    ratios["pg"] = pg_ratio
    draws = {m: draws_of(args.pg_store if m == "pg" else args.store_root, dataset, 0, m)
             for m in METHODS}
    ext = extended_default_draws(args, dataset, 0)
    if ext is not None:
        draws["default"] = ext
    iters = {m: int(budget * 10_000 / ratios[m]) for m in METHODS}
    # PG has 10^4 iterations, fewer than its share where it costs under b
    iters = {m: min(n, draws[m].shape[1]) for m, n in iters.items()}
    keep = min(n - int(round(args.warmup_fraction * n)) for n in iters.values())
    rows = []
    for m in METHODS:
        sample = thinned_tail(draws[m], iters[m], args.warmup_fraction, keep)
        summary = pointwise_rhat_summary(sample)
        rows.append({"dataset": dataset, "method": NAMES[m], "budget": budget,
                     "cost_ratio": ratios[m], "iterations": iters[m], "draws_compared": keep,
                     "rhat_median": summary["median"], "rhat_q90": summary["q90"]})
    return rows


def sigma2_figure(args, datasets, path: Path):
    n = len(datasets)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    edges = np.unique(np.geomspace(1, 10_000, 40).astype(int))
    for ax, dataset in zip(axes.ravel(), datasets):
        mean, sd = long_sigma2(args, dataset, 0)
        for m in METHODS:
            store = args.pg_store if m == "pg" else args.store_root
            trace = sigmas_of(store, dataset, 0, m).mean(axis=0)
            centers = [np.sqrt(a * b) for a, b in zip(edges[:-1], edges[1:])]
            values = [trace[a - 1:b].mean() for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(centers, np.asarray(values) * 1e3, color=COLORS[m], lw=1.6 if m == "pg" else 1.0,
                    label=NAMES[m])
        lo, hi = (mean - sd) * 1e3, (mean + sd) * 1e3
        ax.axhspan(lo, hi, color="0.8", zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(lo - 6 * (hi - lo), hi + 10 * (hi - lo))
        ax.set_title(SHORT[dataset], fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("iteration (log scale)", fontsize=9)
    fig.supylabel(r"$\sigma^2\times10^{3}$, mean of 4 chains, run 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def particle_study(args) -> pd.DataFrame:
    rows = []
    stores = dict(args.pg_c_stores)
    stores[10] = args.pg_store
    for C, store in sorted(stores.items()):
        for dataset in ("fixed100_Abalone", "fixed100_SeoulBike"):
            if not (store / dataset).exists():
                continue
            rows.append({"dataset": dataset, "particles": C,
                         "seconds_per_chain": float(stored(store, dataset, 0, "pg", "fit_seconds").mean()),
                         **chain_stats(args, store, dataset, 0, "pg", *run_inputs(args, dataset, 0))})
    return pd.DataFrame(rows)


def hypercube_table(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.rglob("*.json")):
        res = json.loads(path.read_text())
        half = len(res["leaves"]) // 2
        rows.append({"grid": res["grid"], "D": res["D"], "seed": res["data_seed"], "sampler": res["sampler"],
                     "ess": res["ess_loglik"], "ess_per_second": res["ess_per_second"],
                     "seconds": res["seconds"], "leaves": float(np.mean(res["leaves"][half:])),
                     "loglik": float(np.mean(res["loglik"][half:])),
                     "test_mse": float(np.mean(res["test_mse"][half:]))})
    frame = pd.DataFrame(rows)
    return frame.groupby(["grid", "D", "sampler"], as_index=False).mean(numeric_only=True).drop(columns="seed")


def hypercube_figure(directory: Path, path: Path, grid: str = "uniform", dims=(3, 4), dpi: int = 150):
    """Training log-likelihood and leaf count against iteration, every seed."""
    colors = {"cgm": "#1b9e77", "growprune": "#d95f02", "pg": "#000000"}
    names = {"cgm": "CGM", "growprune": "GrowPrune", "pg": "PG"}
    fig, axes = plt.subplots(len(dims), 2, figsize=(9, 2.6 * len(dims)), squeeze=False)
    for row, D in zip(axes, dims):
        for sampler in ("cgm", "growprune", "pg"):
            for k, f in enumerate(sorted(directory.rglob(f"{grid}_D{D}_seed*_{sampler}.json"))):
                res = json.loads(f.read_text())
                kw = dict(color=colors[sampler], lw=0.8, alpha=0.8, label=names[sampler] if k == 0 else None)
                row[0].plot(np.arange(1, len(res["loglik"]) + 1), res["loglik"], **kw)
                row[1].plot(np.arange(1, len(res["leaves"]) + 1), res["leaves"], **kw)
        row[1].axhline(2 ** D, color="0.5", ls=":", lw=1)
        row[0].set_ylabel(f"D = {D}\ntraining log-likelihood", fontsize=8)
        row[1].set_ylabel("leaves", fontsize=8)
        for ax in row:
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("iteration", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def parse_c_store(text: str):
    c, path = text.split("=", 1)
    return int(c), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--pg-store", type=Path, default=DIAG / "store_pg")
    parser.add_argument("--extended-store", type=Path, default=DIAG / "store_default_ext")
    parser.add_argument("--timing-dir", type=Path, default=DIAG / "timing")
    parser.add_argument("--pg-c-stores", nargs="*", type=parse_c_store, default=[])
    parser.add_argument("--hypercube", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=3000)
    parser.add_argument("--n-segments", type=int, default=4)
    parser.add_argument("--budget", type=float, default=2.5)
    parser.add_argument("--warmup-fraction", type=float, default=0.3)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    tables, figures = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rows, timing = [], []
    for dataset in args.datasets:
        for run in args.runs:
            inputs = run_inputs(args, dataset, run)
            for method in METHODS:
                store = args.pg_store if method == "pg" else args.store_root
                rows.append({"dataset": dataset, "run": run, "method": NAMES[method],
                             **chain_stats(args, store, dataset, run, method, *inputs)})
            timing.append(timing_and_identity(args, dataset, run))
            print(f"[pg] {dataset} run {run:03d}", flush=True)
    run_frame = pd.DataFrame(rows)
    timing_frame = pd.DataFrame(timing)
    run_frame.to_csv(tables / "pg_run_summary.csv", index=False)
    timing_frame.to_csv(tables / "pg_timing.csv", index=False)

    metrics = [c for c in run_frame.columns if c not in ("dataset", "run", "method")]
    by_dataset = run_frame.groupby(["dataset", "method"], sort=False)[metrics].mean().reset_index()
    by_dataset.to_csv(tables / "pg_dataset_summary.csv", index=False)
    for label, group in (("standard", STANDARD), ("sparse", SPARSE)):
        sub = by_dataset[by_dataset.dataset.isin(group)]
        if sub.empty:
            continue
        agg = sub.groupby("method", sort=False)[metrics].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.reset_index().to_csv(tables / f"pg_across_{label}.csv", index=False)

    budget_rows = []
    for dataset in args.datasets:
        pg_ratio = float(timing_frame[(timing_frame.dataset == dataset) & (timing_frame.run == 0)].pg_cost_ratio.iloc[0])
        budget_rows += equal_time(args, dataset, args.budget, pg_ratio)
    pd.DataFrame(budget_rows).to_csv(tables / "pg_equal_time.csv", index=False)

    sigma2_figure(args, args.datasets, figures / "pg_sigma2_traces.png")
    if args.pg_c_stores:
        particle_study(args).to_csv(tables / "pg_particles.csv", index=False)
    if args.hypercube is not None:
        hypercube_table(args.hypercube).to_csv(tables / "pg_hypercube.csv", index=False)
        hypercube_figure(args.hypercube, figures / "pg_hypercube_traces.png", dpi=args.dpi)
    print(f"[pg] tables in {tables}, figures in {figures}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
