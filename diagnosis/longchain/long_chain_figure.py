#!/usr/bin/env python
"""The Section 3 figure for the long Default chains.

Two datasets, one per row, both at 10^7 iterations: the four chains projected
onto their worst between/within direction, and the same draws on their own PCA
axes. One row returns to the null of Equation (5) and the other does not.

    .venv/Scripts/python diagnosis/longchain/long_chain_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chains import long_predictions, separation_index, worst_direction  # noqa: E402

CHAIN_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def spaced(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


def main() -> int:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=script.parent.parent / "store")
    parser.add_argument("--datasets", nargs=2,
                        default=["fixed100_Abalone", "fixed100_SeoulBike"])
    parser.add_argument("--labels", nargs=2, default=["Abalone", "SeoulBike"])
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--plot-draws", type=int, default=1500)
    parser.add_argument("--output", type=Path,
                        default=Path(r"C:\Users\ztykk\Phd\BART\paper\figures\outline_example"
                                     r"\section3_long_chains.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), layout="constrained")
    panels = "abcd"
    for row, (dataset, label) in enumerate(zip(args.datasets, args.labels)):
        draws = long_predictions(args.store_root.resolve(), dataset, args.run, args.burn)
        index, _ = separation_index(draws, args.blocks)
        projected = worst_direction(draws)

        trace_ax, pca_ax = axes[row]
        iterations = np.arange(args.burn, args.burn + projected.shape[1])
        for chain in range(projected.shape[0]):
            trace_ax.plot(iterations, projected[chain], color=CHAIN_COLORS[chain],
                          linewidth=0.5, alpha=0.8, label=f"chain {chain + 1}")
        trace_ax.set(title=f"({panels[2 * row]}) {label}, worst direction ($I={index:.2f}$)",
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
        pca_ax.set(title=f"({panels[2 * row + 1]}) {label}, PCA axes",
                   xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                   ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        for ax in (trace_ax, pca_ax):
            ax.grid(alpha=0.2)
        print(f"[figure] {label}: separation index {index:.2f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"[figure] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
