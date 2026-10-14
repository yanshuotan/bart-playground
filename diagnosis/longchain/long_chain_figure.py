#!/usr/bin/env python
"""Worst-direction trace and two-direction scatter for stored long Default chains.

Default: the two-dataset figure for Section 3. With `--all`, every dataset that
has long chains, ordered by its long-chain pointwise R-hat and split across
pages for the appendix.

    .venv/Scripts/python diagnosis/longchain/long_chain_figure.py
    .venv/Scripts/python diagnosis/longchain/long_chain_figure.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chains import long_predictions, long_runs, worst_directions  # noqa: E402

CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
PANELS = "abcdefghijklmnopqrstuvwxyz"
DISPLAY = {
    "fixed100_Abalone": "Abalone", "fixed100_Airfoil": "Airfoil", "fixed100_CCPP": "CCPP",
    "fixed100_CPUAct": "CPU activity", "fixed100_CalHousing_subsample5000": "CalHousing",
    "fixed100_Concrete": "Concrete", "fixed100_Friedman": "Friedman",
    "fixed100_FriedmanSparseDir_p20": "Friedman-S p20",
    "fixed100_FriedmanSparseDir_p100": "Friedman-S p100",
    "fixed100_FriedmanSparseDir_p200": "Friedman-S p200",
    "fixed100_SeoulBike": "SeoulBike",
}


def spaced(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


def draw(args, pairs: list[tuple[str, str]], output: Path, first_panel: int = 0) -> None:
    """One figure: a row per dataset, worst-direction trace beside a 2D scatter."""
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(11.0, 3.2 * rows), layout="constrained",
                             squeeze=False)
    for row, (dataset, label) in enumerate(pairs):
        draws = long_predictions(args.store_root, dataset, args.run, args.burn)
        result = worst_directions(draws, n_directions=2)
        projected = result["projected"]
        eigenvalues = result["eigenvalues"]
        rhats = result["rhats"]

        trace_ax, scatter_ax = axes[row]
        iterations = np.arange(args.burn, args.burn + projected.shape[1])
        for chain in range(projected.shape[0]):
            trace_ax.plot(iterations, projected[chain, :, 0], color=CHAIN_COLORS[chain],
                          linewidth=0.5, alpha=0.8, label=f"chain {chain + 1}")
        letter = PANELS[first_panel + 2 * row]
        trace_ax.set(title=(f"({letter}) {label}, worst direction "
                            f"($\\widehat R$={rhats[0]:.3f})"),
                     xlabel="stored draw", ylabel="projected prediction")
        if row == 0:
            trace_ax.legend(ncol=4, fontsize=7.5, loc="upper center")

        coords = projected
        panel_title = (f"({PANELS[first_panel + 2 * row + 1]}) {label}, "
                       "top two between/within directions")
        xlabel = f"direction 1 ($\\lambda_1$={eigenvalues[0]:.3g})"
        ylabel = f"direction 2 ($\\lambda_2$={eigenvalues[1]:.3g})"
        idx = spaced(coords.shape[1], args.plot_draws)
        for chain in range(coords.shape[0]):
            scatter_ax.scatter(coords[chain, idx, 0], coords[chain, idx, 1], s=7, alpha=0.18,
                               color=CHAIN_COLORS[chain], edgecolors="none")
            center = coords[chain].mean(axis=0)
            scatter_ax.scatter(*center, marker="X", s=90, color=CHAIN_COLORS[chain],
                               edgecolor="black", linewidth=0.6, zorder=3)
        scatter_ax.set(title=panel_title, xlabel=xlabel, ylabel=ylabel)
        for ax in (trace_ax, scatter_ax):
            ax.grid(alpha=0.2)
        print(f"[figure] {label}: worst-direction R-hat {rhats[0]:.3f}; "
              f"lambda1/lambda2 {eigenvalues[0]:.3g}/{eigenvalues[1]:.3g}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"[figure] {output}", flush=True)


def ordered_datasets(args) -> list[tuple[str, str]]:
    """Every dataset with long chains, ordered by long-chain q90 R-hat."""
    available = [d.name for d in sorted(args.store_root.glob("fixed100_*"))
                 if long_runs(args.store_root, d.name)]
    # order by the run-level table, which keys on the raw store names
    runs = args.store_root.parent / "longchain" / "tables" / "budget_comparison_runs.csv"
    if runs.is_file():
        ordering = pd.read_csv(runs).groupby("dataset")["long_rhat_q90"].mean()
        available.sort(key=lambda d: ordering.get(d, float("inf")))
    return [(d, DISPLAY.get(d, d.removeprefix("fixed100_"))) for d in available]


def main() -> int:
    script = Path(__file__).resolve()
    figures = Path(r"C:\Users\ztykk\Phd\BART\paper\figures\outline_example")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=script.parent.parent / "store")
    parser.add_argument("--datasets", nargs="+",
                        default=["fixed100_Abalone", "fixed100_SeoulBike"])
    parser.add_argument("--all", action="store_true",
                        help="Every dataset with long chains, split across pages.")
    parser.add_argument("--rows-per-figure", type=int, default=4)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--plot-draws", type=int, default=1500)
    parser.add_argument("--figure-dir", type=Path, default=figures)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.store_root = args.store_root.resolve()

    if not args.all:
        pairs = [(d, DISPLAY.get(d, d.removeprefix("fixed100_"))) for d in args.datasets]
        draw(args, pairs, args.figure_dir / "section3_long_chains_worst2.png")
        return 0

    pairs = ordered_datasets(args)
    print(f"[figure] {len(pairs)} datasets, {args.rows_per_figure} per page", flush=True)
    for page, start in enumerate(range(0, len(pairs), args.rows_per_figure), start=1):
        chunk = pairs[start:start + args.rows_per_figure]
        draw(args, chunk, args.figure_dir / f"appendix_long_chains_worst2_{page}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
