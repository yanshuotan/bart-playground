#!/usr/bin/env python3
"""Chain agreement at equal production wall time, from the stored draws.

Budgets are expressed in units of one Default production run: budget $b$ is the
wall time Default needs for $b\\times10^4$ iterations on the same node. A method
whose measured cost ratio is $C_m$ therefore completes $N=b\\times10^4/C_m$
iterations in that time, and the agreement it reaches with those iterations is
what the budget buys.

No sampler is rerun. The transition kernel does not depend on ``ndpost``, the
ladder is read from the store and no stage adapts, so the first $N$ retained
draws of a stored run are exactly what a run of $N$ iterations would have
produced from the same seed.

Two conventions keep the comparison fair at a budget point:

* a warm-up fraction is discarded from every prefix, not a fixed draw count,
  because a compressed budget cannot afford the production burn-in; and
* the retained draws are thinned to a common count $K$, chosen per budget point
  as the smallest any method has there. Rank-split $\\Rhat$ depends on how many
  draws estimate the within-chain variance, so methods compared at one budget
  must be compared at one draw count.

One run is analysed, not an average over runs, because the cost ratios that
set Equation's right-hand side were measured on one run and the ladder width
varies between runs.

Default's own stored short run caps the budget at one. ``extend_default.py``
reruns Default alone at 10^5 iterations under the short-chain seeds, and this
script reads that output when it is present, checks the overlap against the
stored short run and uses it in place of it, which lifts the ceiling to the
point where some other method runs out of draws instead.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
from core import load_with_shape, pointwise_rhat_summary  # noqa: E402
from section6_methods import (full_target_vector, predictive_baselines,  # noqa: E402
                              predictive_metrics, residual_variances, split_indices)
from section7_costs import scalar_summary, targets  # noqa: E402

METHODS = ["default", "default_pt", "mtmh", "mtmh_pt"]
METHOD_NAMES = {"default": "Default", "default_pt": "Default+PT",
                "mtmh": "MTMH", "mtmh_pt": "MTMH+PT"}
TIMING_LABEL = {"default": "default", "mtmh": "mtmh",
                "default_pt": "default_pt (parallel)",
                "mtmh_pt": "mtmh_pt (parallel)"}
# store directory -> the tag the timing summaries are filed under
TIMING_TAG = {
    "fixed100_Abalone": "fixed100_Abalone",
    "fixed100_Airfoil": "fixed100_Airfoil",
    "fixed100_CCPP": "fixed100_CCPP",
    "fixed100_CPUAct": "fixed100_CPUAct",
    "fixed100_CalHousing_subsample5000": "fixed100_CalHousing_subsample5000",
    "fixed100_Concrete": "fixed100_Concrete",
    "fixed100_Friedman": "fixed100_Friedman",
    "fixed100_FriedmanSparseDir_p20": "fixed100_FriedmanSparseDirP20K5",
    "fixed100_FriedmanSparseDir_p100": "fixed100_FriedmanSparseDir",
    "fixed100_FriedmanSparseDir_p200": "fixed100_FriedmanSparseDirP200K5",
    "fixed100_SeoulBike": "fixed100_SeoulBike",
}
DISPLAY = {
    "fixed100_Abalone": "Abalone", "fixed100_Airfoil": "Airfoil",
    "fixed100_CCPP": "CCPP", "fixed100_CPUAct": "CPU activity",
    "fixed100_CalHousing_subsample5000": "CalHousing",
    "fixed100_Concrete": "Concrete", "fixed100_Friedman": "Friedman",
    "fixed100_FriedmanSparseDir_p20": "DART $d=20$",
    "fixed100_FriedmanSparseDir_p100": "DART $d=100$",
    "fixed100_FriedmanSparseDir_p200": "DART $d=200$",
    "fixed100_SeoulBike": "SeoulBike",
}
PRODUCTION_ITERS = 10_000
# the resolving group of settings first, then the persistent one
ORDER = ["fixed100_FriedmanSparseDir_p20", "fixed100_FriedmanSparseDir_p200",
         "fixed100_FriedmanSparseDir_p100", "fixed100_Abalone",
         "fixed100_Friedman", "fixed100_Concrete",
         "fixed100_CalHousing_subsample5000", "fixed100_Airfoil",
         "fixed100_CCPP", "fixed100_CPUAct", "fixed100_SeoulBike"]
NAMES = ["Default", "MTMH", "Default+PT", "MTMH+PT"]
STYLE = {"Default": dict(color="#1f77b4", marker="o"),
         "MTMH": dict(color="#2ca02c", marker="s"),
         "Default+PT": dict(color="#d62728", marker="^"),
         "MTMH+PT": dict(color="#9467bd", marker="D")}


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--extended-store", type=Path,
                        default=DIAG / "store_default_ext",
                        help="Default runs longer than 10^4 iterations, used to raise the budget ceiling. Ignored when absent.")
    parser.add_argument("--extended-method", default="default_ext")
    parser.add_argument("--timing-dir", type=Path, default=DIAG / "timing")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures" / "equal_time")
    parser.add_argument("--datasets", nargs="+", default=list(TIMING_TAG))
    parser.add_argument("--run", type=int, default=0,
                        help="The single run to analyse. Its cost ratios come from the "
                             "timing benchmark of the same run, so the two must match; "
                             "timing exists for run 0 only.")
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--warmup-fraction", type=float, default=0.3)
    parser.add_argument("--min-draws", type=int, default=150,
                        help="Skip a budget point if any method retains fewer draws.")
    parser.add_argument("--long-fractions", nargs="+", type=float,
                        default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--long-draws", type=int, default=350)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(supplied)


def cost_ratios(timing_dir: Path, dataset: str) -> dict[str, float]:
    path = timing_dir / f"{TIMING_TAG[dataset]}_run000_summary.csv"
    rows = {r["label"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    missing = [m for m in METHODS if TIMING_LABEL[m] not in rows]
    if missing:
        raise ValueError(f"{path.name} has no timing for {missing}")
    return {m: float(rows[TIMING_LABEL[m]]["relative_to_default"]) for m in METHODS}


def stored_draws(store_root: Path, dataset: str, run: int, method: str) -> np.ndarray:
    folder = store_root / dataset / "preds"
    hits = sorted(folder.glob(f"*__run{run:03d}__{method}__preds.csv"))
    if not hits:
        raise FileNotFoundError(f"no {method} draws for {dataset} run {run}")
    return load_with_shape(hits[0]).transpose(0, 2, 1)


def extended_default(args, dataset: str, run: int, short: np.ndarray) -> np.ndarray:
    """The longer Default run if stored, checked against the short run it extends."""
    # the extension stores under the dataset tag, which for the DART variants
    # is not the folder name the main store uses
    folder = args.extended_store / TIMING_TAG[dataset] / "preds"
    packed = folder / f"{TIMING_TAG[dataset]}__run{run:03d}__{args.extended_method}__preds.npz"
    hits = sorted(folder.glob(f"*__run{run:03d}__{args.extended_method}__preds.csv"))
    if not hits and not packed.exists():
        return short
    if packed.exists():
        with np.load(packed) as archive:
            draws = archive['predictions'].transpose(0, 2, 1)
    else:
        draws = load_with_shape(hits[0]).transpose(0, 2, 1)
    overlap = min(short.shape[1], draws.shape[1])
    gap = float(np.max(np.abs(short[:, :overlap, :] - draws[:, :overlap, :])))
    if gap > 1e-9:
        raise ValueError(f"{dataset} run {run}: the longer Default run departs from "
                         f"the stored short run by {gap:.3e}, so it is not a prefix "
                         f"extension of it")
    return draws


def thinned_tail(draws: np.ndarray, n_iters: int, warmup: float, keep: int) -> np.ndarray:
    """Retained part of the first `n_iters` draws, thinned to `keep` of them."""
    prefix = draws[:, :n_iters, :]
    start = int(round(warmup * n_iters))
    retained = prefix[:, start:, :]
    if retained.shape[1] < keep:
        raise ValueError(f"only {retained.shape[1]} retained draws, need {keep}")
    index = np.linspace(0, retained.shape[1] - 1, keep).round().astype(int)
    return retained[:, index, :]


def curves_main(args) -> int:
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows, long_rows = [], []
    for dataset in args.datasets:
        ratios = cost_ratios(args.timing_dir, dataset)
        for run in [args.run]:
            draws = {m: stored_draws(args.store_root, dataset, run, m) for m in METHODS}
            draws["default"] = extended_default(args, dataset, run, draws["default"])

            for budget in args.budgets:
                iters = {m: int(budget * PRODUCTION_ITERS / ratios[m]) for m in METHODS}
                if any(iters[m] > draws[m].shape[1] for m in METHODS):
                    continue
                retained = {m: iters[m] - int(round(args.warmup_fraction * iters[m]))
                            for m in METHODS}
                keep = min(retained.values())
                if keep < args.min_draws:
                    continue
                for method in METHODS:
                    sample = thinned_tail(draws[method], iters[method],
                                          args.warmup_fraction, keep)
                    summary = pointwise_rhat_summary(sample)
                    rows.append({
                        "dataset": dataset, "run": run,
                        "method": METHOD_NAMES[method], "budget": budget,
                        "cost_ratio": ratios[method], "iterations": iters[method],
                        "draws_compared": keep,
                        "rhat_median": summary["median"], "rhat_q90": summary["q90"],
                    })
            del draws

            long_hits = sorted((args.store_root / dataset / "preds").glob(
                f"*__run{run:03d}__default_long__preds.csv"))
            if long_hits:
                long_draws = load_with_shape(long_hits[0]).transpose(0, 2, 1)
                stored = long_draws.shape[1]
                for fraction in args.long_fractions:
                    count = int(round(fraction * stored))
                    if count - int(round(args.warmup_fraction * count)) < args.long_draws:
                        continue
                    sample = thinned_tail(long_draws, count, args.warmup_fraction,
                                          args.long_draws)
                    summary = pointwise_rhat_summary(sample)
                    long_rows.append({
                        "dataset": dataset, "run": run, "method": "Default (long)",
                        "budget": fraction * 1000.0, "cost_ratio": 1.0,
                        "iterations": int(fraction * 10_000_000),
                        "draws_compared": args.long_draws,
                        "rhat_median": summary["median"], "rhat_q90": summary["q90"],
                    })
                del long_draws
            print(f"[equal-time] {dataset} run {run:03d}", flush=True)

    frame = pd.DataFrame(rows + long_rows)
    frame.to_csv(args.table_dir / "equal_time_budget.csv", index=False)

    short = pd.DataFrame(rows)
    reached = set.intersection(*(set(g.budget) for _, g in short.groupby("dataset")))
    headline = max(reached)
    summary = short[short.budget.eq(headline)][
        ["dataset", "method", "budget", "cost_ratio", "iterations",
         "draws_compared", "rhat_median", "rhat_q90"]].reset_index(drop=True)
    summary.to_csv(args.table_dir / "equal_time_summary.csv", index=False)

    order = [d for d in ORDER if d in set(short.dataset)]
    median = summary.pivot(index="dataset", columns="method", values="rhat_median")
    q90 = summary.pivot(index="dataset", columns="method", values="rhat_q90")
    draw_curves(args, frame, order)

    pd.set_option("display.width", 200)
    wide = median.loc[order]
    wide.index = [DISPLAY[d] for d in order]
    print()
    print(f"run {args.run:03d}, budget b={headline:g}, draws compared "
          f"{summary.draws_compared.min():.0f} to {summary.draws_compared.max():.0f}")
    print("Pointwise rank-split R-hat median at the largest shared budget")
    print(wide[NAMES].round(3).to_string())
    for label, table in (("median", median), ("q90", q90)):
        beats = {m: int((table.loc[order, m] < table.loc[order, "Default"]).sum())
                 for m in NAMES[1:]}
        print(f"lower than Default, {label}: "
              + ", ".join(f"{m} {n}/{len(order)}" for m, n in beats.items()))
    print()
    print(f"tables: {args.table_dir}")
    print(f"figures: {args.figure_dir}")
    return 0


def draw_curves(args: argparse.Namespace, frame: pd.DataFrame, order) -> None:
    """One panel per setting: agreement against wall time, all on one grid."""
    short = frame[frame.method.ne("Default (long)")]
    floor = (frame[frame.method.eq("Default (long)") & frame.budget.eq(1000.0)]
             .set_index("dataset")["rhat_median"])
    fig, axes = plt.subplots(3, 4, figsize=(11.2, 7.4), sharex=True)
    flat = axes.ravel()
    for ax, dataset in zip(flat, order):
        part = short[short.dataset.eq(dataset)]
        for method in NAMES:
            line = part[part.method.eq(method)].sort_values("budget")
            ax.plot(line.budget, line.rhat_median, label=method, linewidth=1.4,
                    markersize=3.4, **STYLE[method])
        if dataset in floor.index:
            ax.axhline(floor[dataset], color="0.35", linestyle=(0, (4, 2)),
                       linewidth=1.1,
                       label=r"Default at $1000\times$" if dataset == order[0] else None)
        ax.set_title(DISPLAY[dataset], fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    for ax in flat[len(order):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("wall time (Default runs)", fontsize=8)
    for row in axes:
        row[0].set_ylabel(r"median $\widehat{R}$", fontsize=8)
    flat[0].legend(fontsize=6.6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.figure_dir / "equal_time_curves.png", dpi=args.dpi,
                bbox_inches="tight")
    plt.close(fig)


def budget_configurations(budget, ratios, chains=4):
    if budget <= 0 or chains < 2 or any(v <= 0 for v in ratios.values()):
        raise ValueError('Positive budget/cost ratios and at least two pooled chains required')
    result=[(m,METHOD_NAMES[m],False,int(budget*10000/ratios[m])) for m in METHODS]
    result.append(('default',f'Pooled Default ({chains} sequential)',True,int(budget*10000/chains)))
    return result

def evaluate(draws, y, truth, pooled):
    means = draws.mean(axis=1)
    distributions = [draws.reshape(-1, draws.shape[-1])] if pooled else list(draws)
    estimates = [means.mean(axis=0)] if pooled else list(means)
    mse = float(np.mean([np.mean((f-y)**2) for f in estimates]))
    out = {'mse': mse, 'rmse_root_mean_mse': np.sqrt(mse),
           'rmse_mean_chainwise': float(np.mean([np.sqrt(np.mean((f-y)**2)) for f in estimates]))}
    intervals = [np.quantile(d, [.05,.95], axis=0) for d in distributions]
    out['interval_width_median'] = float(np.mean([np.median(hi-lo) for lo,hi in intervals]))
    if truth is not None:
        out['latent_f_containment'] = float(np.mean([np.mean((lo<=truth)&(truth<=hi)) for lo,hi in intervals]))
        out['rmse_f_root_mean_mse'] = float(np.sqrt(np.mean([np.mean((f-truth)**2) for f in estimates])))
    data = xr.DataArray(draws, dims=('chain','draw','point'), name='f')
    rhat = az.rhat(data, method='rank')['f'].values
    out.update(rhat_median=float(np.median(rhat)), rhat_q90=float(np.quantile(rhat,.9)))
    average = scalar_summary(draws.mean(axis=2))
    out['panel_mean_rhat'] = average['rhat']
    # These R-hats compare the four constituent chains / repeated realizations;
    # pooling their draws does not itself improve their agreement.
    return out

def baselines_main(argv=None):
    args = parse_args(argv)
    if args.run != 0:
        raise ValueError('Wall-time benchmark is available for run 0 only')
    rows=[]
    for dataset in args.datasets:
        ratios=cost_ratios(args.timing_dir,dataset)
        all_draws={m:stored_draws(args.store_root,dataset,0,m) for m in METHODS}
        all_draws['default']=extended_default(args,dataset,0,all_draws['default'])
        y,truth=targets(dataset,args.store_root,0)
        full_y=full_target_vector(args.store_root,dataset)
        train=full_y[split_indices(args.store_root,dataset,0,'train_idx')]
        reference=predictive_baselines(train,y)
        y_scale=float(np.ptp(train))
        sigma={m:residual_variances(args.store_root,dataset,0,m) for m in METHODS}
        extension=args.extended_store/TIMING_TAG[dataset]/'sigmas'/f'{TIMING_TAG[dataset]}__run000__{args.extended_method}__sigmas.csv'
        if all_draws['default'].shape[1] > sigma['default'].shape[1]:
            raw=load_with_shape(extension)
            extended=raw.reshape(raw.shape[0],-1)
            np.testing.assert_allclose(extended[:,:sigma['default'].shape[1]],sigma['default'],rtol=0,atol=1e-9)
            sigma['default']=extended
        for budget in args.budgets:
            configurations=budget_configurations(budget,ratios)
            if any(n>all_draws[m].shape[1] or n-int(round(args.warmup_fraction*n))<args.min_draws for m,_,_,n in configurations):
                continue
            for method,label,pooled,n in configurations:
                start=int(round(args.warmup_fraction*n))
                sample=all_draws[method][:,start:n,:]
                scored=evaluate(sample,y,truth,pooled)
                # Score posterior predictive y at the headline budget. Noise
                # uses the matching draw's variance and training-only scale.
                if budget == max(args.budgets):
                    variance=sigma[method][:,start:n]
                    f_for_score=sample.reshape(1,-1,sample.shape[-1]) if pooled else sample
                    v_for_score=variance.reshape(1,-1) if pooled else variance
                    _,crps=predictive_metrics(f_for_score,v_for_score,y,y_scale,20261014)
                    scored.update(crps=crps,relative_crps=crps/reference['climatology_crps'])
                scored['relative_rmse']=scored['rmse_root_mean_mse']/reference['naive_rmse']
                rows.append({'dataset':dataset,'run':0,'method':label,'budget':budget,
                             'estimator_chains':4 if pooled else 1,'repeated_single_estimators':1 if pooled else 4,
                             'iterations_per_chain':n,'warmup_per_chain':start,
                             'retained_per_chain':n-start,'charged_wall_units':n/10000*ratios[method]*(4 if pooled else 1),
                             **scored})
        print(f'[wall baseline] {dataset}',flush=True)
    frame=pd.DataFrame(rows)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table_dir/'wall_time_baselines.csv',index=False)
    headline=max(set.intersection(*(set(g.budget) for _,g in frame.groupby('dataset'))))
    frame[frame.budget.eq(headline)].to_csv(args.table_dir/'wall_time_baselines_summary.csv',index=False)



def main(supplied: list[str] | None = None) -> int:
    """`curves` draws the equal-wall-time figure; `baselines` scores the estimators."""
    argv = list(sys.argv[1:] if supplied is None else supplied)
    part = argv[0] if argv and not argv[0].startswith("-") else "curves"
    rest = argv[1:] if argv and not argv[0].startswith("-") else argv
    if part == "baselines":
        baselines_main(rest)
        return 0
    if part != "curves":
        raise SystemExit(f"unknown part {part!r}; expected 'curves' or 'baselines'")
    return curves_main(parse_args(rest))


if __name__ == "__main__":
    raise SystemExit(main())
