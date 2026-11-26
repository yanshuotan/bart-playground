#!/usr/bin/env python3
"""Per-chain posterior at the panel input with the largest pointwise R-hat.

For each dataset and run, the input j with the largest R-hat_j of the latent
predictions is chosen on one record (production Default by default). At that
input every requested record is drawn as one row of two panels:

* left, the latent f(x_j*) draws of each chain;
* right, the outcome-predictive draws f + sigma * e on the response scale,
  sigma = sqrt(eps_sigma2) * (max - min of the training targets), as in the
  CRPS of section7_methods.

Each panel shows a kernel density per chain, the pooled density, and each
chain's central 90% interval as a bar below the axis. The true f is marked on
the generated designs and the observed y_j* on the predictive panels. Every
record keeps the draws after the common 3,000-draw warm-up.

``--overview`` also draws the first record of every dataset in one grid.

The paper figures:

* 4.4: ``--datasets fixed100_Friedman --records default default_long --panels latent --suffix _long_latent``
* 7.3: ``--datasets fixed100_Friedman --records default mtmh default_pt mtmh_pt --panels latent --suffix _methods_latent``
* appendix: the same two calls with ``--panels predictive`` and suffixes
  ``_long_predictive`` and ``_methods_predictive``; ``--overview --suffix _default`` and
  ``--overview --records mtmh_pt --suffix _mtmh_pt`` (same inputs, chosen on Default)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG))
from core import OPENML_SETTINGS, pointwise_rhat  # noqa: E402
from section7_methods import (full_target_vector, predictions,  # noqa: E402
                              residual_variances, training_targets, y_test)
from section7_costs import DISPLAY, SYNTHETIC, targets  # noqa: E402

DATASETS = (
    "fixed100_Abalone", "fixed100_Airfoil", "fixed100_CCPP",
    "fixed100_CalHousing_subsample10000", "fixed100_Concrete", "fixed100_CPUAct",
    "fixed100_SeoulBike", "fixed100_Superconduct_subsample10000", "fixed100_Friedman",
    "fixed100_FriedmanSparseDir_p20", "fixed100_FriedmanSparseDir_p100",
    "fixed100_FriedmanSparseDir_p200",
    *OPENML_SETTINGS,
)
RECORDS = {
    "default": "Default",
    "default_long": "Default, thinned $10^7$",
    "default_long_tail": "Default, tail of $10^7$",
    "mtmh": "MTMH",
    "default_pt": "Default+PT",
    "mtmh_pt": "MTMH+PT",
}
CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
NOISE_SEED = 20261117


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures" / "worst_input")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--runs", nargs="+", type=int, default=[0])
    parser.add_argument("--records", nargs="+", default=["default"], choices=list(RECORDS),
                        help="One figure row per record, in this order.")
    parser.add_argument("--select-record", default="default", choices=list(RECORDS),
                        help="Record whose largest R-hat_j picks the input.")
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--interval", type=float, default=0.9)
    parser.add_argument("--suffix", default="",
                        help="Appended to output names, e.g. _methods.")
    parser.add_argument("--overview", action="store_true",
                        help="Also draw the first record of every dataset in one grid.")
    parser.add_argument("--panels", default="both", choices=["both", "latent", "predictive"],
                        help="both: one row per record with latent and predictive panels; "
                             "latent or predictive: one panel per record, two per row.")
    parser.add_argument("--title", action="store_true",
                        help="Add a title naming the dataset, run and input to each dataset figure.")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(supplied)


def run_data(args: argparse.Namespace, dataset: str, run: int) -> dict:
    """The chosen input and, per record, the latent and predictive draws there."""
    y = y_test(args.store_root, dataset, run)
    train = training_targets(args.store_root, dataset, run,
                             full_target_vector(args.store_root, dataset), y)
    # DefaultPreprocessor min-max scales y on the training targets.
    y_scale = float(np.max(train) - np.min(train))
    truth = None
    if dataset in SYNTHETIC:
        observed, truth = targets(dataset, args.store_root, run)
        if not np.allclose(observed, y):
            raise ValueError(f"{dataset} run {run}: rebuilt responses differ from the stored test set")

    select = pointwise_rhat(predictions(args.store_root, dataset, run, args.select_record, args.burn))
    j = int(np.argmax(select))
    out = {"input": j, "rhat_median_select": float(np.median(select)), "y": float(y[j]),
           "truth": None if truth is None else float(truth[j]), "records": {}}
    for k, record in enumerate(args.records):
        f = predictions(args.store_root, dataset, run, record, args.burn)
        latent = f[:, :, j]
        variances = residual_variances(args.store_root, dataset, run, record, args.burn)
        rng = np.random.default_rng(NOISE_SEED + 100 * run + k)
        predictive = latent + np.sqrt(variances) * y_scale * rng.standard_normal(latent.shape)
        out["records"][record] = {
            "latent": latent, "predictive": predictive,
            "rhat_latent": float(pointwise_rhat(latent[:, :, None])[0]),
            "rhat_predictive": float(pointwise_rhat(predictive[:, :, None])[0]),
            "rhat_median": float(np.median(pointwise_rhat(f))),
        }
    return out


def axis_range(samples: list[np.ndarray]) -> np.ndarray:
    pooled = np.concatenate([s.reshape(-1) for s in samples])
    lo, hi = np.percentile(pooled, [0.2, 99.8])
    pad = 0.08 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, 400)


def density_panel(ax, samples: np.ndarray, grid: np.ndarray, interval: float,
                  marker: float | None, marker_label: str, title: str) -> None:
    tail = 50 * (1 - interval)
    densities = [gaussian_kde(chain)(grid) for chain in samples]
    pooled = gaussian_kde(samples.reshape(-1))(grid)
    top = max(max(d.max() for d in densities), pooled.max())
    for c, (chain, density) in enumerate(zip(samples, densities)):
        ax.plot(grid, density, color=CHAIN_COLORS[c], lw=1.3, label=f"chain {c + 1}")
        lo, hi = np.percentile(chain, [tail, 100 - tail])
        level = -0.07 * top * (c + 1)
        ax.plot([lo, hi], [level, level], color=CHAIN_COLORS[c], lw=2.2, solid_capstyle="butt")
    ax.plot(grid, pooled, color="0.2", lw=1.0, ls="--", label="pooled")
    if marker is not None:
        ax.axvline(marker, color="k", lw=1.0, ls=":", label=marker_label)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.07 * top * (len(samples) + 1), 1.08 * top)
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def draw_rows(axes, data: dict, records: list[str], interval: float, label_rows: bool) -> None:
    grids = [axis_range([data["records"][r][key] for r in records])
             for key in ("latent", "predictive")]
    for row, record in enumerate(records):
        result = data["records"][record]
        prefix = f"{RECORDS[record]}: " if label_rows else ""
        density_panel(axes[row][0], result["latent"], grids[0], interval, data["truth"], "true $f$",
                      f"{prefix}latent $f$, $\\hat R={result['rhat_latent']:.2f}$")
        density_panel(axes[row][1], result["predictive"], grids[1], interval, data["y"], "observed $y$",
                      f"{prefix}predictive $f+\\sigma\\varepsilon$, $\\hat R={result['rhat_predictive']:.2f}$")


def single_kind_figure(args: argparse.Namespace, dataset: str, run: int, data: dict) -> Path:
    """One panel per record, latent or predictive only, two panels per row."""
    kind = args.panels
    n = len(args.records)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.9 * nrows), squeeze=False)
    grid = axis_range([data["records"][r][kind] for r in args.records])
    marker, marker_label = (data["truth"], "true $f$") if kind == "latent" else (data["y"], "observed $y$")
    what = "latent $f$" if kind == "latent" else "predictive $f+\\sigma\\varepsilon$"
    for k, record in enumerate(args.records):
        result = data["records"][record]
        density_panel(axes[k // ncols][k % ncols], result[kind], grid, args.interval, marker, marker_label,
                      f"{RECORDS[record]}: {what}, $\\hat R={result[f'rhat_{kind}']:.2f}$")
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = args.figure_dir / f"worst_input_{dataset.removeprefix('fixed100_')}_run{run:03d}{args.suffix}.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def dataset_figure(args: argparse.Namespace, dataset: str, run: int, data: dict) -> Path:
    if args.panels != "both":
        return single_kind_figure(args, dataset, run, data)
    rows = len(args.records)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 2.7 * rows + 0.5), squeeze=False)
    draw_rows(axes, data, args.records, args.interval, label_rows=rows > 1)
    axes[0][0].legend(fontsize=7, frameon=False)
    axes[0][1].legend(fontsize=7, frameon=False)
    if args.title:
        fig.suptitle(f"{DISPLAY[dataset]}, run {run}, input {data['input']}: largest $\\hat R_j$ of "
                     f"{RECORDS[args.select_record]} (panel median {data['rhat_median_select']:.2f})",
                     fontsize=10)
    fig.tight_layout()
    path = args.figure_dir / f"worst_input_{dataset.removeprefix('fixed100_')}_run{run:03d}{args.suffix}.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def overview_figure(args: argparse.Namespace, run: int, collected: dict) -> Path:
    names = [d for d in args.datasets if d in collected]
    nrows = (len(names) + 1) // 2
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 2.6 * nrows), squeeze=False)
    record = args.records[0]
    for k, dataset in enumerate(names):
        row, col = divmod(k, 2)
        pair = [[axes[row][2 * col], axes[row][2 * col + 1]]]
        draw_rows(pair, collected[dataset], [record], args.interval, label_rows=False)
        for ax in pair[0]:
            ax.set_title(f"{DISPLAY[dataset]}, input {collected[dataset]['input']}: {ax.get_title()}",
                         fontsize=8)
    for k in range(len(names), 2 * nrows):
        row, col = divmod(k, 2)
        axes[row][2 * col].axis("off")
        axes[row][2 * col + 1].axis("off")
    handles = [Line2D([], [], color=color, lw=1.3) for color in CHAIN_COLORS]
    handles += [Line2D([], [], color="0.2", lw=1.0, ls="--"), Line2D([], [], color="k", lw=1.0, ls=":")]
    labels = [f"chain {c + 1}" for c in range(len(CHAIN_COLORS))]
    labels += ["pooled", "true $f$ (latent panels, generated designs); observed $y$ (predictive panels)"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    path = args.figure_dir / f"worst_input_overview_run{run:03d}{args.suffix}.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in args.runs:
        collected = {}
        for dataset in args.datasets:
            data = run_data(args, dataset, run)
            collected[dataset] = data
            for record, result in data["records"].items():
                rows.append({"dataset": DISPLAY[dataset], "run": run, "input": data["input"],
                             "select_record": args.select_record, "record": record,
                             "rhat_latent": result["rhat_latent"],
                             "rhat_predictive": result["rhat_predictive"],
                             "rhat_median_record": result["rhat_median"],
                             "true_f": data["truth"], "observed_y": data["y"]})
            print(dataset_figure(args, dataset, run, data), flush=True)
        if args.overview:
            print(overview_figure(args, run, collected), flush=True)
    table = args.table_dir / f"worst_input_density{args.suffix}.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    print(table)


if __name__ == "__main__":
    main()
