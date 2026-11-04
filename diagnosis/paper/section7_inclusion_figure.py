#!/usr/bin/env python3
"""Per-chain inclusion probabilities of every predictor for one DART run.

For each record (production Default, MTMH+PT and the thinned long Default
record) the figure plots, for every predictor, the fraction of the 7,000
post-warm-up draws of each chain whose ensemble splits on that predictor at
least once. The first five predictors carry the Friedman signal; the rest are
nuisance inputs. The dashed line marks the median-probability threshold 0.5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
sys.path.insert(0, str(DIAG))
from core import load_with_shape  # noqa: E402
from section7_feature_use import ACTIVE, histogram_path  # noqa: E402

RECORDS = [("default", "Default"), ("mtmh_pt", "MTMH+PT"),
           ("default_long", r"Default, thinned $10^7$-iteration record")]
CHAIN_STYLE = [dict(color="#1b9e77", marker="o"), dict(color="#d95f02", marker="s"),
               dict(color="#7570b3", marker="^"), dict(color="#e7298a", marker="D")]


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store")
    parser.add_argument("--figure-dir", type=Path, default=HERE / "figures" / "feature_use")
    parser.add_argument("--dataset", default="fixed100_FriedmanSparseDir_p100")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--label-above", type=float, default=0.5,
                        help="Label nuisance predictors whose inclusion probability "
                             "exceeds this value in some chain.")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(supplied)


def inclusion(store_root: Path, dataset: str, run: int, method: str, burn: int) -> np.ndarray:
    path = histogram_path(store_root, dataset, run, method)
    if path is None:
        raise FileNotFoundError(f"no {method} split counts for {dataset} run {run}")
    counts = load_with_shape(path)[:, burn:, :]
    if counts.shape[1] != 7000:
        raise ValueError(f"{path.name}: {counts.shape[1]} retained draws, expected 7000")
    return (counts > 0).mean(axis=1)                     # (chain, predictor)


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(RECORDS), 1, figsize=(7.2, 6.0), sharex=True, sharey=True)
    for ax, (method, title) in zip(axes, RECORDS):
        pip = inclusion(args.store_root, args.dataset, args.run, method, args.burn)
        chains, predictors = pip.shape
        index = np.arange(1, predictors + 1)
        ax.axvspan(0.5, ACTIVE + 0.5, color="0.88", linewidth=0)
        ax.axhline(0.5, color="0.35", linestyle=(0, (4, 2)), linewidth=0.9)
        for c in range(chains):
            ax.scatter(index, pip[c], s=13, linewidths=0.4, edgecolors="white",
                       label=f"chain {c + 1}", zorder=3, **CHAIN_STYLE[c])
        for v in range(ACTIVE, predictors):
            if pip[:, v].max() > args.label_above:
                ax.annotate(f"$x_{{{v + 1}}}$", (v + 1, pip[:, v].max()),
                            textcoords="offset points", xytext=(5, -3), fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlim(0, predictors + 1)
        ax.set_ylabel("inclusion probability", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        print(f"[inclusion] {method}: max nuisance per chain "
              f"{np.round(pip[:, ACTIVE:].max(axis=1), 2).tolist()}", flush=True)
    axes[-1].set_xlabel("predictor index", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    name = f"inclusion_{args.dataset}_run{args.run:03d}.png"
    fig.savefig(args.figure_dir / name, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[inclusion] wrote {args.figure_dir / name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
