#!/usr/bin/env python
"""Pointwise R-hat for the Default sampler at two computational budgets.

The short chains run 10^4 iterations and the long chains 10^7, and both are
stored as roughly 10^4 draws. Modern rank-normalized split R-hat is computed at
each fixed test point using the same number of post-burn draws per chain, so the
two budgets differ only in the span of iterations behind the stored draws.

    .venv/Scripts/python diagnosis/paper/section3_long_budget.py table
    .venv/Scripts/python diagnosis/paper/section3_long_budget.py figures
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core import (dataset_tag, input_signature, load_with_shape,  # noqa: E402
                  long_predictions, long_runs, pointwise_rhat_summary, run_inputs,
                  worst_directions)

DISPLAY = {
    "fixed100_Abalone": "Abalone",
    "fixed100_Airfoil": "Airfoil",
    "fixed100_CPUAct": "CPUAct",
    "fixed100_CCPP": "CCPP",
    "fixed100_CalHousing_subsample5000": "CalHousing",
    "fixed100_Concrete": "Concrete",
    "fixed100_Friedman": "Friedman",
    "fixed100_FriedmanSparseDir_p20": "Friedman-S p20",
    "fixed100_FriedmanSparseDir_p100": "Friedman-S p100",
    "fixed100_FriedmanSparseDir_p200": "Friedman-S p200",
    "fixed100_SeoulBike": "SeoulBike",
    "fixed100_Superconduct_subsample5000": "Superconduct",
}


def short_default(store: Path, dataset: str, run: int, burn: int) -> np.ndarray:
    tag = dataset_tag(store, dataset)
    path = store / dataset / "preds" / f"{tag}__run{run:03d}__default__preds.csv"
    return load_with_shape(path).transpose(0, 2, 1)[:, burn:, :]


def thin(draws: np.ndarray, count: int) -> np.ndarray:
    """Evenly spaced draws spanning the whole chain, so the span is preserved."""
    if draws.shape[1] <= count:
        return draws
    return draws[:, np.linspace(0, draws.shape[1] - 1, count, dtype=int), :]


def table_main(argv=None) -> int:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=script.parent.parent / "store")
    parser.add_argument("--out-dir", type=Path, default=script.parent)
    parser.add_argument("--short-burn", type=int, default=3000)
    parser.add_argument("--long-burn", type=int, default=3000)
    parser.add_argument("--force", action="store_true",
                        help="Recompute every run instead of reusing the cache.")
    args = parser.parse_args(argv)

    store = args.store_root.resolve()
    cache_path = args.out_dir / "tables" / "budget_comparison_cache.csv"
    fingerprint = f"metric=rank_rhat_v1;short_burn={args.short_burn};long_burn={args.long_burn}"
    cached = {}
    if cache_path.is_file() and not args.force:
        frame = pd.read_csv(cache_path)
        frame = frame[frame["settings"] == fingerprint]  # stale settings drop out
        cached = {(r.dataset, int(r.run)): r._asdict() for r in frame.itertuples(index=False)}
    rows = []
    for directory in sorted(store.glob("fixed100_*")):
        dataset = directory.name
        if not sorted((directory / "preds").glob("*__default_long__preds.csv")):
            continue
        tag = dataset_tag(store, dataset)
        for path in sorted((directory / "preds").glob(f"{tag}__run*__default__preds.csv")):
            run = int(re.search(r"__run(\d+)__", path.name).group(1))
            # a replaced draw file changes this, so its cached row is recomputed
            inputs = input_signature({}, run_inputs(store, dataset, run, ["default", "default_long"]))
            if (dataset, run) in cached and cached[(dataset, run)].get("inputs") == inputs:
                hit = cached[(dataset, run)]
                rows.append({k: hit[k] for k in (
                    "dataset", "run", "draws_per_chain",
                    "short_rhat_median", "short_rhat_q90", "short_rhat_max",
                    "long_rhat_median", "long_rhat_q90", "long_rhat_max", "inputs",
                )})
                print(f"[budget] {dataset} run {run:03d}  cached", flush=True)
                continue
            short = short_default(store, dataset, run, args.short_burn)
            long = long_predictions(store, dataset, run, args.long_burn)
            count = min(short.shape[1], long.shape[1])
            short_rhat = pointwise_rhat_summary(thin(short, count))
            long_rhat = pointwise_rhat_summary(thin(long, count))
            rows.append({
                "dataset": dataset, "run": run, "draws_per_chain": count,
                "short_rhat_median": short_rhat["median"],
                "short_rhat_q90": short_rhat["q90"],
                "short_rhat_max": short_rhat["maximum"],
                "long_rhat_median": long_rhat["median"],
                "long_rhat_q90": long_rhat["q90"],
                "long_rhat_max": long_rhat["maximum"],
                "inputs": inputs,
            })
            print(f"[budget] {dataset} run {run:03d}  "
                  f"short median/q90 {short_rhat['median']:.3f}/{short_rhat['q90']:.3f}  "
                  f"long {long_rhat['median']:.3f}/{long_rhat['q90']:.3f}", flush=True)

    runs = pd.DataFrame(rows)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    runs.assign(settings=fingerprint).to_csv(cache_path, index=False)
    runs = runs.drop(columns="inputs")
    runs.to_csv(args.out_dir / "tables" / "budget_comparison_runs.csv", index=False)

    grouped = runs.groupby("dataset").agg(
        runs=("run", "count"), draws=("draws_per_chain", "min"),
        short_median_lo=("short_rhat_median", "min"), short_median_hi=("short_rhat_median", "max"),
        long_median_lo=("long_rhat_median", "min"), long_median_hi=("long_rhat_median", "max"),
        short_q90_lo=("short_rhat_q90", "min"), short_q90_hi=("short_rhat_q90", "max"),
        long_q90_lo=("long_rhat_q90", "min"), long_q90_hi=("long_rhat_q90", "max"),
    ).reset_index().sort_values("long_q90_hi")
    grouped["dataset"] = grouped["dataset"].map(DISPLAY)
    grouped.to_csv(args.out_dir / "tables" / "budget_comparison_datasets.csv", index=False)

    lines = [
        "# Pointwise R-hat at two computational budgets",
        "",
        "Modern rank-normalized split R-hat is calculated at each fixed test point and",
        "summarized by its median and 90th percentile. Both budgets use the same number",
        "of post-burn draws per chain, so they differ only in the iterations those draws",
        "span: 10^4 for the short chains and 10^7 for the long ones.",
        "",
        "| dataset | runs | draws/chain | short median | long median | short q90 | long q90 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in grouped.iterrows():
        lines.append(f"| {row.dataset} | {row.runs} | {row.draws} | "
                     f"{row.short_median_lo:.3f}-{row.short_median_hi:.3f} | "
                     f"{row.long_median_lo:.3f}-{row.long_median_hi:.3f} | "
                     f"{row.short_q90_lo:.3f}-{row.short_q90_hi:.3f} | "
                     f"{row.long_q90_lo:.3f}-{row.long_q90_hi:.3f} |")
    lines.append("")
    (args.out_dir / "budget_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0



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
    "fixed100_Superconduct_subsample5000": "Superconduct",
}


def spaced(size: int, count: int) -> np.ndarray:
    return np.linspace(0, size - 1, min(size, count), dtype=int)


def draw_long_panels(args, pairs: list[tuple[str, str]], output: Path, first_panel: int = 0) -> None:
    """One figure: a row per dataset, worst-direction trace beside a 2D scatter."""
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(11.0, 3.2 * rows), layout="constrained",
                             squeeze=False)
    for row, (dataset, label) in enumerate(pairs):
        draws = long_predictions(args.store_root, dataset, args.run, args.burn)
        result = worst_directions(draws, n_directions=2)
        projected = result["projected"]
        eigenvalues = result["eigenvalues"]

        trace_ax, scatter_ax = axes[row]
        iterations = np.arange(args.burn, args.burn + projected.shape[1])
        for chain in range(projected.shape[0]):
            trace_ax.plot(iterations, projected[chain, :, 0], color=CHAIN_COLORS[chain],
                          linewidth=0.5, alpha=0.8, label=f"chain {chain + 1}")
        trace_ax.axvline(args.burn + int(result["fit_draws"]), color="0.35",
                         linestyle=(0, (4, 2)), linewidth=1.0)
        letter = PANELS[first_panel + 2 * row]
        trace_ax.set(title=f"({letter}) {label}, worst direction",
                     xlabel="stored draw", ylabel="projected prediction")
        if row == 0:
            trace_ax.legend(ncol=4, fontsize=7.5, loc="upper center")

        # the fitted first half is left out of the scatter; the trace shows both halves
        coords = projected[:, int(result["fit_draws"]):, :]
        panel_title = (f"({PANELS[first_panel + 2 * row + 1]}) {label}, "
                       "top two directions, held half")
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
        print(f"[figure] {label}: lambda1/lambda2 {eigenvalues[0]:.3g}/{eigenvalues[1]:.3g}",
              flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(f"[figure] {output}", flush=True)


def ordered_datasets(args) -> list[tuple[str, str]]:
    """Every dataset with long chains, ordered by long-chain q90 R-hat."""
    available = [d.name for d in sorted(args.store_root.glob("fixed100_*"))
                 if long_runs(args.store_root, d.name)]
    # order by the run-level table, which keys on the raw store names
    runs = Path(__file__).resolve().parent / "tables" / "budget_comparison_runs.csv"
    if runs.is_file():
        ordering = pd.read_csv(runs).groupby("dataset")["long_rhat_q90"].mean()
        available = [d for d in available if d in ordering.index]
        available.sort(key=lambda d: ordering.get(d, float("inf")))
    return [(d, DISPLAY.get(d, d.removeprefix("fixed100_"))) for d in available]


def figures_main(argv=None) -> int:
    script = Path(__file__).resolve()
    figures = script.parent / "figures" / "long_budget"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=script.parent.parent / "store")
    parser.add_argument("--datasets", nargs="+",
                        default=["fixed100_Concrete", "fixed100_SeoulBike"])
    parser.add_argument("--all", action="store_true",
                        help="Every dataset with long chains, split across pages.")
    parser.add_argument("--rows-per-figure", type=int, default=4)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--burn", type=int, default=3000)
    parser.add_argument("--plot-draws", type=int, default=1500)
    parser.add_argument("--figure-dir", type=Path, default=figures)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    args.store_root = args.store_root.resolve()

    if not args.all:
        pairs = [(d, DISPLAY.get(d, d.removeprefix("fixed100_"))) for d in args.datasets]
        draw_long_panels(args, pairs, args.figure_dir / "section3_long_chains_worst2.png")
        return 0

    pairs = ordered_datasets(args)
    print(f"[figure] {len(pairs)} datasets, {args.rows_per_figure} per page", flush=True)
    for page, start in enumerate(range(0, len(pairs), args.rows_per_figure), start=1):
        chunk = pairs[start:start + args.rows_per_figure]
        draw_long_panels(args, chunk, args.figure_dir / f"appendix_long_chains_worst2_{page}.png")
    return 0



def main(supplied: list[str] | None = None) -> int:
    """`table` builds the two-budget table; `figures` draws from its output."""
    argv = list(sys.argv[1:] if supplied is None else supplied)
    part = argv[0] if argv and not argv[0].startswith("-") else "table"
    rest = argv[1:] if argv and not argv[0].startswith("-") else argv
    if part == "figures":
        return figures_main(rest)
    if part != "table":
        raise SystemExit(f"unknown part {part!r}; expected 'table' or 'figures'")
    return table_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
