#!/usr/bin/env python3
"""Late-window and long-span prediction diagnostics for Abalone and SeoulBike.

Each sequence contributes 7,000 draws from four chains. The contiguous Default
tail covers 7,000 iterations near the end of a 10^7-iteration run; the stride-
1,000 trace covers approximately seven million iterations of that run. The
production arms cover iterations 3,001--10,000 under their respective kernels.
ESS describes each recorded sequence. Figure bars average five paired runs.
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
from core import load_with_shape, pointwise_rhat_summary, prediction_variance_terms  # noqa: E402

TAIL = "default_long_tail"
SPARSE = "default_long"
SHORT = ["default", "mtmh", "default_pt", "mtmh_pt"]
LABEL = {TAIL: "Default tail, stride 1", SPARSE: "Default long, stride 1000",
         "default": "Default", "mtmh": "MTMH",
         "default_pt": "Default+PT", "mtmh_pt": "MTMH+PT"}
DISPLAY = {"fixed100_Abalone": "Abalone", "fixed100_SeoulBike": "SeoulBike"}
STORAGE_STRIDE = 1000


def display(dataset: str) -> str:
    """Axis label for a store directory, for the settings DISPLAY omits."""
    return DISPLAY.get(dataset, dataset.removeprefix("fixed100_"))


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "figures" / "long_tail")
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
    # Each chain is measured on its own and the four are then averaged, giving
    # one value per input before any summary across inputs. That is the
    # convention the outline's relative-ESS table uses, so the two agree.
    single = az.ess(alone, method="bulk")["prediction"].values.reshape(chains, points)
    single = single.mean(axis=0)
    relative = single / retained
    summary = pointwise_rhat_summary(block)
    within, between = prediction_variance_terms(block)
    return {
        "rhat_median": summary["median"],
        "rhat_q90": summary["q90"],
        "rel_ess_median": float(np.median(relative)),
        "rel_ess_q10": float(np.quantile(relative, 0.1)),
        "within_variance_median": float(np.median(within)),
        "between_variance_median": float(np.median(between)),
    }



def draw(args, summary: pd.DataFrame) -> None:
    """Agreement at the median pointwise R-hat, as a standalone figure."""
    order = [LABEL[m] for m in [TAIL, SPARSE] + SHORT]
    colours = dict(zip(order, ["#d62728", "#7f3f9f", "#1f77b4", "#2ca02c",
                               "#ff7f0e", "#8c564b"]))
    figures = args.out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    name = "long_tail_rhat_median.png"
    wide = summary.pivot(index="dataset", columns="method", values="rhat_median")
    names = [display(d) for d in wide.index]
    fig, ax = plt.subplots(figsize=(1.05 * len(names) + 2.6, 4.1),
                           layout="constrained")
    width, offsets = 0.13, np.arange(len(order)) - (len(order) - 1) / 2
    for k, method in enumerate(order):
        if method not in wide.columns:
            continue
        ax.bar(np.arange(len(names)) + offsets[k] * width, wide[method],
               width=width, color=colours[method], label=method)
    ax.axhline(1.0, color="0.5", linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title(r"Pointwise $\widehat{R}$, median over inputs", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.savefig(figures / name, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {figures / name}")


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
                rows.append(row)
                del block
            print(f"[long-tail] {dataset} run {run:03d}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "long_tail_run_level.csv", index=False)
    summary = frame.drop(columns=["run"]).groupby(
        ["dataset", "method"], sort=False).mean().reset_index()
    summary.to_csv(tables / "long_tail_summary.csv", index=False)

    draw(args, summary)

    pd.set_option("display.width", 220)
    order = [LABEL[m] for m in [TAIL, SPARSE] + SHORT]
    for column in ("rhat_median", "rel_ess_median"):
        wide = summary.pivot(index="dataset", columns="method", values=column)
        wide.index = [display(d) for d in wide.index]
        print()
        print(column)
        print(wide[[c for c in order if c in wide.columns]].round(4).to_string())
    print()
    print(f"tables: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
