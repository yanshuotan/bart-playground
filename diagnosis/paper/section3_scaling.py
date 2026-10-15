#!/usr/bin/env python3
"""Agreement against training-set size and noise level, from the sweep's draws.

The ratio summaries alone cannot say why agreement changes with $n$: a larger
$\\Rhat$ can mean chains that sit further apart or a posterior that has narrowed
around them. The within- and between-chain terms are therefore reported
separately, on the response scale, alongside the ratio they form.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
from core import load_with_shape, pointwise_rhat_summary  # noqa: E402

TAG = re.compile(r"^scal_(?P<source>[A-Za-z]+)_n(?P<n>\d+)(?:_sd(?P<sd>\d+))?$")


def parse_args(supplied: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=DIAG / "store_scaling")
    parser.add_argument("--table-dir", type=Path, default=HERE / "tables")
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--figure-dir", type=Path,
                        default=HERE / "figures" / "scaling")
    parser.add_argument("--figure-only", action="store_true",
                        help="Draw from the stored table without rescoring draws.")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(supplied)


def score(draws: np.ndarray) -> dict[str, float]:
    summary = pointwise_rhat_summary(draws)
    chain_means = draws.mean(axis=1)
    within = draws.var(axis=1, ddof=1).mean(axis=0)
    between = chain_means.var(axis=0, ddof=1)
    chains, retained, points = draws.shape
    alone = xr.DataArray(draws.transpose(1, 0, 2).reshape(1, -1, chains * points),
                         dims=("chain", "draw", "point"), name="prediction")
    single = az.ess(alone, method="bulk")["prediction"].values.reshape(chains, points)
    return {
        "rhat_median": summary["median"],
        "rhat_q90": summary["q90"],
        "within_median": float(np.median(within)),
        "between_median": float(np.median(between)),
        "between_share_median": float(np.median(between / (between + within))),
        "rel_ess_single_median": float(np.median(single.mean(axis=0) / retained)),
    }


SOURCES = {"CCPP": "#1f77b4", "CalHousing": "#d62728", "Friedman": "#2ca02c"}


def draw(args: argparse.Namespace, frame: pd.DataFrame) -> None:
    """Agreement against size and noise, with the two variance terms beneath."""
    keys = ["source", "n_train", "noise"]
    mean = frame.drop(columns=["run"]).groupby(keys, dropna=False).mean().reset_index()
    size = mean[mean.noise.isna() | mean.noise.eq(1.0)]
    noise = mean[mean.source.eq("Friedman") & mean.n_train.eq(2000)].sort_values("noise")

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), layout="constrained")
    top_left, top_right, low_left, low_right = axes.flat

    for source, colour in SOURCES.items():
        part = size[size.source.eq(source)].sort_values("n_train")
        if part.empty:
            continue
        top_left.plot(part.n_train, part.rhat_median, marker="o", color=colour,
                      linewidth=1.5, markersize=4.5, label=source)
        low_left.plot(part.n_train, part.within_median,
                      marker="o", color=colour, linewidth=1.5, markersize=4)
        low_left.plot(part.n_train, part.between_median,
                      marker="s", color=colour, linewidth=1.5, markersize=4,
                      linestyle=(0, (4, 2)))
    top_left.set(xscale="log", xlabel="training rows", ylabel=r"median $\widehat{R}$",
                 title="(a) Agreement against training-set size")
    top_left.legend(fontsize=8.5, frameon=False)
    low_left.set(xscale="log", yscale="log", xlabel="training rows",
                 ylabel="prediction variance (response units squared)",
                 title="(c) Within (solid) and between (dashed), original scale")

    top_right.plot(noise.noise, noise.rhat_median, marker="o", color="#7f3f9f",
                   linewidth=1.5, markersize=4.5)
    top_right.set(xscale="log", xlabel="noise standard deviation",
                  ylabel=r"median $\widehat{R}$",
                  title="(b) Agreement against noise, Friedman at $n=2000$")
    low_right.plot(noise.noise, noise.within_median, marker="o",
                   color="#7f3f9f", linewidth=1.5, markersize=4, label="within")
    low_right.plot(noise.noise, noise.between_median, marker="s",
                   color="#7f3f9f", linewidth=1.5, markersize=4,
                   linestyle=(0, (4, 2)), label="between")
    low_right.set(xscale="log", yscale="log", xlabel="noise standard deviation",
                  ylabel="prediction variance (response units squared)",
                  title="(d) Within and between variance, original scale")
    low_right.legend(fontsize=8.5, frameon=False)
    for ax in (top_right, low_right):
        ax.set_xticks([0.5, 1, 2, 5])
        ax.set_xticklabels(["0.5", "1", "2", "5"])
        ax.xaxis.set_minor_locator(NullLocator())
    for ax in axes.flat:
        ax.grid(alpha=0.25, linewidth=0.5)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_dir / "scaling_curves.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {args.figure_dir / 'scaling_curves.png'}")


def main(supplied: list[str] | None = None) -> int:
    args = parse_args(supplied)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    if args.figure_only:
        frame = pd.read_csv(args.table_dir / "scaling_run_level.csv")
        draw(args, frame)
        return 0

    rows = []
    for folder in sorted(p for p in args.store_root.iterdir() if p.is_dir()):
        matched = TAG.match(folder.name)
        if not matched:
            continue
        source = matched.group("source")
        n_train = int(matched.group("n"))
        noise = int(matched.group("sd")) / 100 if matched.group("sd") else np.nan
        for path in sorted((folder / "preds").glob("*__default__preds.csv")):
            run = int(re.search(r"run(\d+)", path.name).group(1))
            draws = load_with_shape(path).transpose(0, 2, 1)[:, args.burn:, :]
            row = {"source": source, "n_train": n_train, "noise": noise,
                   "run": run, "draws": draws.shape[1]}
            row.update(score(draws))
            rows.append(row)
            del draws
        print(f"[scaling] {folder.name}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.table_dir / "scaling_run_level.csv", index=False)
    draw(args, frame)
    keys = ["source", "n_train", "noise"]
    summary = frame.drop(columns=["run"]).groupby(keys, dropna=False).agg(["mean", "std"])
    summary.to_csv(args.table_dir / "scaling_summary.csv")

    pd.set_option("display.width", 220)
    mean = frame.drop(columns=["run"]).groupby(keys, dropna=False).mean().reset_index()
    columns = ["rhat_median", "rhat_q90", "within_median", "between_median",
               "between_share_median", "rel_ess_single_median"]
    print()
    print("size sweep (noise fixed where it applies)")
    size = mean[mean.noise.isna() | mean.noise.eq(1.0)].sort_values(["source", "n_train"])
    print(size[keys + columns].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print()
    print("noise sweep, Friedman at n=2000")
    noise = mean[mean.source.eq("Friedman") & mean.n_train.eq(2000)].sort_values("noise")
    print(noise[keys + columns].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print()
    print(f"tables: {args.table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
