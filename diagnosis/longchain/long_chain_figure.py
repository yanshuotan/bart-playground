#!/usr/bin/env python
"""Worst-direction and PCA panels for the stored long Default chains.

Default: the two-dataset figure for Section 3. With `--all`, every dataset that
has long chains, ordered by its separation index and split across pages, for the
appendix.

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
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chains import long_predictions, long_runs, separation_index, worst_direction  # noqa: E402

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
    """One figure: a row per dataset, worst-direction trace beside PCA axes."""
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(11.0, 3.2 * rows), layout="constrained",
                             squeeze=False)
    for row, (dataset, label) in enumerate(pairs):
        draws = long_predictions(args.store_root, dataset, args.run, args.burn)
        index, _ = separation_index(draws, args.blocks)
        projected = worst_direction(draws)

        trace_ax, pca_ax = axes[row]
        iterations = np.arange(args.burn, args.burn + projected.shape[1])
        for chain in range(projected.shape[0]):
            trace_ax.plot(iterations, projected[chain], color=CHAIN_COLORS[chain],
                          linewidth=0.5, alpha=0.8, label=f"chain {chain + 1}")
        letter = PANELS[first_panel + 2 * row]
        trace_ax.set(title=f"({letter}) {label}, worst direction ($I={index:.2f}$)",
                     xlabel="stored draw", ylabel="projected prediction")
        if row == 0:
            trace_ax.legend(ncol=4, fontsize=7.5, loc="upper center")

        flat = draws.reshape(-1, draws.shape[-1])
        pca = PCA(n_components=2, random_state=0).fit(flat)
        coords = pca.transform(flat).reshape(draws.shape[0], draws.shape[1], 2)
        idx = spaced(coords.shape[1], args.plot_draws)
        for chain in range(coords.shape[0]):
            pca_ax.scatter(coords[chain, idx, 0], coords[chain, idx, 1], s=7, alpha=0.18,
                           color=CHAIN_COLORS[chain], edgecolors="none")
            center = coords[chain].mean(axis=0)
            pca_ax.scatter(*center, marker="X", s=90, color=CHAIN_COLORS[chain],
                           edgecolor="black", linewidth=0.6, zorder=3)
        pca_ax.set(title=f"({PANELS[first_panel + 2 * row + 1]}) {label}, PCA axes",
                   xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                   ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        for ax in (trace_ax, pca_ax):
            ax.grid(alpha=0.2)
        print(f"[figure] {label}: separation index {index:.2f}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"[figure] {output}", flush=True)


def ordered_datasets(args) -> list[tuple[str, str]]:
    """Every dataset with long chains, ordered by separation index when known."""
    available = [d.name for d in sorted(args.store_root.glob("fixed100_*"))
                 if long_runs(args.store_root, d.name)]
    # order by the run-level table, which keys on the raw store names
    runs = args.store_root.parent / "longchain" / "tables" / "budget_comparison_runs.csv"
    if runs.is_file():
        worst = pd.read_csv(runs).groupby("dataset")["long_index"].min()
        available.sort(key=lambda d: worst.get(d, float("inf")))
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
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--plot-draws", type=int, default=1500)
    parser.add_argument("--figure-dir", type=Path, default=figures)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.store_root = args.store_root.resolve()

    if not args.all:
        pairs = [(d, DISPLAY.get(d, d.removeprefix("fixed100_"))) for d in args.datasets]
        draw(args, pairs, args.figure_dir / "section3_long_chains.png")
        return 0

    pairs = ordered_datasets(args)
    print(f"[figure] {len(pairs)} datasets, {args.rows_per_figure} per page", flush=True)
    for page, start in enumerate(range(0, len(pairs), args.rows_per_figure), start=1):
        chunk = pairs[start:start + args.rows_per_figure]
        draw(args, chunk, args.figure_dir / f"appendix_long_chains_{page}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
