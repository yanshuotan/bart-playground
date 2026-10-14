#!/usr/bin/env python3
"""What the contiguous tail of the long chains shows that the stored trace cannot.

The long Default runs are stored at stride 1,000, so every diagnostic computed
from them describes a sequence whose neighbouring draws are already a thousand
iterations apart. The closing 10,000 iterations of the same runs were also kept
at stride 1, and that is the only record here of how a BART chain moves from one
iteration to the next after a long warm-up.

Three comparisons are drawn, all on 7,000 draws from four chains so that the
draw count never explains a difference:

* the contiguous tail, which is Default at stride 1 after :math:`10^7`
  iterations;
* the stored long trace, the same runs at stride 1,000; and
* the four production arms, which are also stride 1 but start cold and run for
  :math:`10^4` iterations.

The tail and the production runs therefore differ only in how long the sampler
had already been running, and the tail and the stored trace differ only in
thinning. Comparing the integrated autocorrelation time measured on the tail
with the storage stride says whether that stride was long enough to have
produced near-independent draws.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG / "longchain"))
from chains import load_with_shape, pointwise_rhat_summary  # noqa: E402

TAIL = "default_long_tail"
SPARSE = "default_long"
SHORT = ["default", "mtmh", "default_pt", "mtmh_pt"]
LABEL = {TAIL: "Default tail, stride 1", SPARSE: "Default long, stride 1000",
         "default": "Default", "mtmh": "MTMH",
         "default_pt": "Default+PT", "mtmh_pt": "MTMH+PT"}
DISPLAY = {"fixed100_Abalone": "Abalone", "fixed100_SeoulBike": "SeoulBike"}
STORAGE_STRIDE = 1000


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(r"C:\Users\ztykk\Phd\BART\paper\long_tail"))
    parser.add_argument("--datasets", nargs="+", default=list(DISPLAY))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--draws", type=int, default=7000,
                        help="Draws kept from the end of every sequence.")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(supplied)


def sequence(store_root: Path, dataset: str, run: int, method: str,
             draws: int) -> np.ndarray | None:
    """The last `draws` stored draws, as (chain, draw, test point)."""
    hits = sorted((store_root / dataset / "preds").glob(
        f"*__run{run:03d}__{method}__preds.csv"))
    if not hits:
        return None
    stored = load_with_shape(hits[0]).transpose(0, 2, 1)
    if stored.shape[1] < draws:
        raise ValueError(f"{dataset} {method} has only {stored.shape[1]} draws")
    return stored[:, -draws:, :]


def efficiency(block: np.ndarray) -> dict[str, float]:
    """Agreement, and how much independent information one chain carries."""
    chains, retained, points = block.shape
    alone = xr.DataArray(block.transpose(1, 0, 2).reshape(1, -1, chains * points),
                         dims=("chain", "draw", "point"), name="prediction")
    single = az.ess(alone, method="bulk")["prediction"].values.reshape(chains, points)
    relative = single / retained
    summary = pointwise_rhat_summary(block)
    # the integrated autocorrelation time in units of stored draws; one stored
    # draw is one iteration in the tail and a thousand in the stored trace
    tau = retained / np.clip(single, 1e-9, None)
    return {
        "rhat_median": summary["median"],
        "rhat_q90": summary["q90"],
        "rel_ess_median": float(np.median(relative)),
        "rel_ess_q10": float(np.quantile(relative, 0.1)),
        "tau_draws_median": float(np.median(tau)),
        "tau_draws_q90": float(np.quantile(tau, 0.9)),
    }



def autocorrelation(block: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean autocorrelation over chains and inputs, by lag."""
    centred = block - block.mean(axis=1, keepdims=True)
    variance = (centred ** 2).mean(axis=1)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        cov = (centred[:, :-lag, :] * centred[:, lag:, :]).mean(axis=1)
        out[lag] = float(np.mean(cov / np.clip(variance, 1e-12, None)))
    return out


def draw(args, summary: pd.DataFrame, acf: dict) -> None:
    """Agreement at two summaries, and the decay only the tail can show."""
    order = [LABEL[m] for m in [TAIL, SPARSE] + SHORT]
    colours = dict(zip(order, ["#d62728", "#7f3f9f", "#1f77b4", "#2ca02c",
                               "#ff7f0e", "#8c564b"]))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), layout="constrained")

    panels = (("rhat_median", r"(a) Pointwise $\widehat{R}$, median over inputs"),
              ("rhat_q90", r"(b) Pointwise $\widehat{R}$, 90th percentile"))
    for ax, (column, title) in zip(axes[:2], panels):
        wide = summary.pivot(index="dataset", columns="method", values=column)
        names = [DISPLAY[d] for d in wide.index]
        width, offsets = 0.13, np.arange(len(order)) - (len(order) - 1) / 2
        for k, method in enumerate(order):
            if method not in wide.columns:
                continue
            ax.bar(np.arange(len(names)) + offsets[k] * width, wide[method],
                   width=width, color=colours[method], label=method)
        ax.axhline(1.0, color="0.5", linewidth=0.8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[2]
    for dataset, series in acf.items():
        for method, values in series.items():
            ax.plot(np.arange(len(values)), values, linewidth=1.4,
                    color=colours[LABEL[method]],
                    linestyle="-" if dataset == args.datasets[0] else (0, (3, 2)),
                    label=f"{DISPLAY[dataset]}, {LABEL[method]}")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set(xlabel="lag, iterations", ylabel="autocorrelation",
           title="(c) Decay at stride 1")
    ax.legend(fontsize=6.6, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)

    figures = args.out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "long_tail.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {figures / 'long_tail.png'}")


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        for run in args.runs:
            for method in [TAIL, SPARSE] + SHORT:
                block = sequence(args.store_root, dataset, run, method, args.draws)
                if block is None:
                    continue
                row = {"dataset": dataset, "run": run, "method": LABEL[method],
                       "stride": STORAGE_STRIDE if method == SPARSE else 1,
                       "draws": block.shape[1]}
                row.update(efficiency(block))
                # one stored draw is `stride` iterations, so this is the
                # autocorrelation time on the sampler's own clock
                row["tau_iterations_median"] = row["tau_draws_median"] * row["stride"]
                rows.append(row)
                del block
            print(f"[long-tail] {dataset} run {run:03d}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "long_tail_run_level.csv", index=False)
    summary = frame.drop(columns=["run"]).groupby(
        ["dataset", "method"], sort=False).mean().reset_index()
    summary.to_csv(tables / "long_tail_summary.csv", index=False)

    acf = {}
    for dataset in args.datasets:
        acf[dataset] = {}
        for method in (TAIL, "default"):
            block = sequence(args.store_root, dataset, args.runs[0], method, args.draws)
            if block is not None:
                acf[dataset][method] = autocorrelation(block, 2000)
                del block
    draw(args, summary, acf)

    pd.set_option("display.width", 220)
    order = [LABEL[m] for m in [TAIL, SPARSE] + SHORT]
    for column in ("rhat_median", "rel_ess_median", "tau_draws_median",
                   "tau_iterations_median"):
        wide = summary.pivot(index="dataset", columns="method", values=column)
        wide.index = [DISPLAY[d] for d in wide.index]
        print()
        print(column)
        print(wide[[c for c in order if c in wide.columns]].round(4).to_string())
    print()
    print(f"tables: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
